"""Claude (Anthropic) subscription OAuth helpers.

This provider lets an Odysseus admin sign in with their Claude Pro/Max
**subscription** (the same OAuth grant `claude setup-token` / the Claude Code
CLI use) instead of a pay-per-token Anthropic API key. It stores the OAuth
refresh token server-side and resolves a fresh access token at request time.

Auth differs from the API-key Anthropic provider in exactly one way: requests
carry ``Authorization: Bearer <access_token>`` plus the
``anthropic-beta: oauth-2025-04-20`` header instead of ``x-api-key``. Everything
else (the /v1/messages payload, response parsing, streaming) is shared with the
existing Anthropic path in ``src/llm_core.py``.

The flow is the standard OAuth 2.0 Authorization Code + PKCE grant with a manual
code paste (Anthropic shows a ``<code>#<state>`` string after sign-in), so it
does not use the device-flow scaffolding the ChatGPT subscription provider uses.

Note on terms of service: a subscription is intended for first-party Claude
surfaces (Claude apps, Claude Code). Routing a third-party app through it is a
grey area and the account owner accepts that risk by connecting here. The
mechanism is symmetric with the existing ChatGPT Subscription provider.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import threading
from datetime import timedelta
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

CLAUDE_SUBSCRIPTION_PROVIDER = "claude-subscription"

# Real Anthropic API root that requests are sent to.
ANTHROPIC_API_BASE = (
    os.getenv("CLAUDE_SUBSCRIPTION_API_BASE", "").strip().rstrip("/")
    or "https://api.anthropic.com"
)
# Sentinel base_url stored on the ModelEndpoint / ProviderAuthSession. The
# trailing ``/oauth`` marker is what makes ``_detect_provider`` classify this
# endpoint as ``claude-subscription`` (OAuth bearer) rather than the API-key
# ``anthropic`` provider — both live on the same host. It is stripped back to
# ANTHROPIC_API_BASE when the real /v1/messages or /v1/models URL is built.
DEFAULT_CLAUDE_SUBSCRIPTION_BASE_URL = f"{ANTHROPIC_API_BASE}/oauth"

# Public OAuth client used by the Claude Code CLI / `claude setup-token`.
CLAUDE_OAUTH_CLIENT_ID = (
    os.getenv("CLAUDE_OAUTH_CLIENT_ID", "").strip()
    or "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
)
# Endpoints + scopes match the Claude Code CLI exactly (extracted from the
# binary): authorize, token, and redirect all live on platform.claude.com.
# claude.ai/oauth/authorize does NOT recognize these scopes (it returns
# "Solicitação OAuth inválida / Escopo desconhecido") — the subscription OAuth
# for this client goes through platform.claude.com (the Console/platform host).
CLAUDE_OAUTH_AUTHORIZE_URL = "https://platform.claude.com/oauth/authorize"
CLAUDE_OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CLAUDE_OAUTH_REDIRECT_URI = "https://platform.claude.com/oauth/code/callback"
# Inference scopes only. `org:create_api_key` is the CLI's "create an API key"
# mode and is rejected as an unknown scope on the subscription authorize flow
# (the real subscription token is granted user:inference + user:profile, never
# org:create_api_key).
CLAUDE_OAUTH_SCOPES = "user:inference user:profile"

# Beta header that authorizes OAuth-bearer access to /v1/messages.
CLAUDE_OAUTH_BETA = "oauth-2025-04-20"
ANTHROPIC_VERSION = "2023-06-01"

# Refresh the access token this many seconds before it actually expires.
CLAUDE_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 300

_AUTH_REFRESH_LOCKS: dict[str, threading.Lock] = {}
_AUTH_REFRESH_LOCKS_GUARD = threading.Lock()


def _database_handles():
    from core.database import ProviderAuthSession, SessionLocal, utcnow_naive

    return ProviderAuthSession, SessionLocal, utcnow_naive


def _refresh_lock_for(auth_id: str) -> threading.Lock:
    with _AUTH_REFRESH_LOCKS_GUARD:
        lock = _AUTH_REFRESH_LOCKS.get(auth_id)
        if lock is None:
            lock = threading.Lock()
            _AUTH_REFRESH_LOCKS[auth_id] = lock
        return lock


class ClaudeSubscriptionError(RuntimeError):
    """Base error for Claude subscription provider failures."""


class ClaudeSubscriptionReauthRequired(ClaudeSubscriptionError):
    """Stored OAuth credentials are invalid or expired beyond refresh."""


class ClaudeSubscriptionRateLimited(ClaudeSubscriptionError):
    """Upstream quota/rate limit; reconnecting will not fix it."""


class ClaudeSubscriptionAuthNotFound(ClaudeSubscriptionError):
    """No matching owner-scoped auth session exists."""


def is_claude_subscription_base(url: str) -> bool:
    """True for the Claude-subscription sentinel base (``…anthropic.com/oauth``).

    Checked before the plain ``anthropic.com`` host match in ``_detect_provider``
    so an OAuth-backed endpoint is not misread as the API-key provider.
    """
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url or "")
        host = (parsed.hostname or "").lower().rstrip(".")
        path = (parsed.path or "").rstrip("/")
    except Exception:
        return False
    if not (host == "anthropic.com" or host.endswith(".anthropic.com")):
        return False
    return path.endswith("/oauth")


# ── OAuth bearer headers (shared with src/llm_core anthropic path) ──

def claude_oauth_headers(access_token: Optional[str]) -> Dict[str, str]:
    """Headers for an OAuth-bearer Anthropic request (no x-api-key)."""
    headers = {
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": CLAUDE_OAUTH_BETA,
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers


# ── PKCE / authorization-code flow ──

def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def generate_pkce() -> Dict[str, str]:
    """Return a fresh PKCE verifier/challenge and an anti-CSRF state."""
    verifier = _b64url(os.urandom(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    state = secrets.token_urlsafe(24)
    return {"code_verifier": verifier, "code_challenge": challenge, "state": state}


def build_authorize_url(code_challenge: str, state: str) -> str:
    """Build the Claude OAuth authorize URL for the manual-code grant."""
    from urllib.parse import urlencode

    params = {
        "code": "true",
        "client_id": CLAUDE_OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": CLAUDE_OAUTH_REDIRECT_URI,
        "scope": CLAUDE_OAUTH_SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{CLAUDE_OAUTH_AUTHORIZE_URL}?{urlencode(params)}"


def split_pasted_code(pasted: str) -> tuple[str, Optional[str]]:
    """Split the ``<code>#<state>`` string Anthropic shows after sign-in."""
    pasted = (pasted or "").strip()
    if "#" in pasted:
        code, _, state = pasted.partition("#")
        return code.strip(), (state.strip() or None)
    return pasted, None


def _raise_for_oauth_response(response: httpx.Response, action: str) -> None:
    if response.status_code < 400:
        return
    code = ""
    message = f"Claude Subscription {action} failed with HTTP {response.status_code}."
    try:
        payload = response.json()
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict):
            code = str(err.get("type") or err.get("code") or "").strip()
            msg = err.get("message")
            if msg:
                message = f"Claude Subscription {action} failed: {msg}"
        elif isinstance(err, str):
            code = err.strip()
            desc = payload.get("error_description") or payload.get("message")
            if desc:
                message = f"Claude Subscription {action} failed: {desc}"
    except Exception:
        pass
    if response.status_code == 429:
        raise ClaudeSubscriptionRateLimited(
            "Claude Subscription quota or rate limit was reached. Credentials are still valid."
        )
    if response.status_code in (400, 401, 403) or code in {
        "invalid_grant",
        "invalid_token",
        "invalid_request",
        "unauthorized",
    }:
        raise ClaudeSubscriptionReauthRequired(message)
    raise ClaudeSubscriptionError(message)


def _token_request(payload: Dict[str, Any], action: str, timeout: float = 20.0) -> Dict[str, Any]:
    response = httpx.post(
        CLAUDE_OAUTH_TOKEN_URL,
        json=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=timeout,
        follow_redirects=True,
    )
    _raise_for_oauth_response(response, action)
    try:
        data = response.json()
    except Exception as exc:
        raise ClaudeSubscriptionError(f"Claude Subscription {action} returned invalid JSON.") from exc
    if not isinstance(data, dict) or not data.get("access_token"):
        raise ClaudeSubscriptionReauthRequired(f"Claude Subscription {action} did not return an access token.")
    return data


def exchange_authorization_code(code: str, state: Optional[str], code_verifier: str) -> Dict[str, Any]:
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": CLAUDE_OAUTH_REDIRECT_URI,
        "client_id": CLAUDE_OAUTH_CLIENT_ID,
        "code_verifier": code_verifier,
    }
    if state:
        payload["state"] = state
    return _token_request(payload, "token exchange")


def refresh_oauth_tokens(refresh_token: str) -> Dict[str, Any]:
    if not refresh_token:
        raise ClaudeSubscriptionReauthRequired(
            "Claude Subscription is missing a refresh token. Reconnect the provider."
        )
    return _token_request(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLAUDE_OAUTH_CLIENT_ID,
        },
        "token refresh",
    )


# ── Model discovery ──

def fetch_available_models(access_token: str, timeout: float = 12.0) -> List[str]:
    """List Claude chat model IDs available to this subscription via /v1/models."""
    if not access_token:
        return []
    try:
        response = httpx.get(
            f"{ANTHROPIC_API_BASE}/v1/models?limit=100",
            headers=claude_oauth_headers(access_token),
            timeout=timeout,
        )
        if response.status_code != 200:
            return []
        data = response.json()
    except Exception:
        return []
    entries = data.get("data", []) if isinstance(data, dict) else []
    ordered: List[str] = []
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if not isinstance(mid, str) or not mid.startswith("claude"):
            continue
        if mid not in seen:
            ordered.append(mid)
            seen.add(mid)
    return ordered


# ── Runtime credential resolution (refresh-aware) ──

def _access_token_is_expiring(expires_at, utcnow_naive, skew_seconds: int) -> bool:
    """True when there's no stored expiry or it's within ``skew_seconds`` of now."""
    if not expires_at:
        return True
    try:
        return expires_at <= (utcnow_naive() + timedelta(seconds=int(skew_seconds)))
    except Exception:
        return True


def resolve_runtime_credentials(
    auth_id: str, owner: Optional[str] = None, *, force_refresh: bool = False
) -> Dict[str, Any]:
    ProviderAuthSession, SessionLocal, utcnow_naive = _database_handles()
    db = SessionLocal()
    try:
        q = db.query(ProviderAuthSession).filter(
            ProviderAuthSession.id == auth_id,
            ProviderAuthSession.provider == CLAUDE_SUBSCRIPTION_PROVIDER,
        )
        if owner:
            q = q.filter(ProviderAuthSession.owner == owner)
        row = q.first()
        if row is None:
            raise ClaudeSubscriptionAuthNotFound(
                "Claude Subscription credentials were not found for this user."
            )

        expiring = force_refresh or _access_token_is_expiring(
            getattr(row, "expires_at", None), utcnow_naive, CLAUDE_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
        )
        if expiring:
            with _refresh_lock_for(auth_id):
                db.refresh(row)
                expiring = force_refresh or _access_token_is_expiring(
                    getattr(row, "expires_at", None), utcnow_naive,
                    CLAUDE_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
                )
                if expiring:
                    refreshed = refresh_oauth_tokens(row.refresh_token or "")
                    row.access_token = refreshed["access_token"]
                    if refreshed.get("refresh_token"):
                        row.refresh_token = refreshed["refresh_token"]
                    expires_in = refreshed.get("expires_in")
                    if isinstance(expires_in, (int, float)) and expires_in > 0:
                        row.expires_at = utcnow_naive() + timedelta(seconds=int(expires_in))
                    row.last_refresh = utcnow_naive()
                    db.commit()
                    db.refresh(row)

        return {
            "provider": CLAUDE_SUBSCRIPTION_PROVIDER,
            "base_url": (row.base_url or DEFAULT_CLAUDE_SUBSCRIPTION_BASE_URL).rstrip("/"),
            "api_key": row.access_token or "",
            "auth_mode": row.auth_mode or "claude",
        }
    finally:
        db.close()


def to_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, ClaudeSubscriptionRateLimited):
        return HTTPException(429, str(exc))
    if isinstance(exc, (ClaudeSubscriptionReauthRequired, ClaudeSubscriptionAuthNotFound)):
        return HTTPException(401, f"{exc} Reconnect the provider.")
    return HTTPException(502, str(exc))
