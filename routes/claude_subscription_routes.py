"""Claude Subscription OAuth (PKCE) setup routes.

Two admin-only endpoints drive the manual authorization-code flow:

  POST /api/claude-subscription/start     -> { authorize_url, state }
  POST /api/claude-subscription/complete  -> { id, name, base_url, models }

The PKCE verifier never leaves this process; only the access/refresh tokens are
persisted (encrypted at rest via ProviderAuthSession).
"""

import json
import logging
import threading
import time
import uuid
from datetime import timedelta
from typing import Dict, Optional

from fastapi import APIRouter, Form, HTTPException, Request

from core.database import ModelEndpoint, ProviderAuthSession, SessionLocal, utcnow_naive
from core.middleware import require_admin
from src import claude_subscription
from src.auth_helpers import get_current_user

logger = logging.getLogger(__name__)

_PENDING_TTL_SECONDS = 900


class _PendingPkceStore:
    """Thread-safe in-memory PKCE verifier store, keyed by the OAuth ``state``."""

    def __init__(self):
        self._pending: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def _prune(self) -> None:
        now = time.time()
        for k in [k for k, v in self._pending.items() if v.get("expires_at", 0) < now]:
            self._pending.pop(k, None)

    def add(self, state: str, code_verifier: str, owner: Optional[str]) -> None:
        with self._lock:
            self._prune()
            self._pending[state] = {
                "code_verifier": code_verifier,
                "owner": owner,
                "expires_at": time.time() + _PENDING_TTL_SECONDS,
            }

    def take(self, state: str) -> Optional[Dict]:
        with self._lock:
            self._prune()
            return self._pending.pop(state, None)


_PENDING = _PendingPkceStore()


def _provision_endpoint(tokens: Dict, owner: Optional[str]) -> Dict:
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not access_token or not refresh_token:
        raise ValueError("Claude token response was missing access_token or refresh_token")

    base = claude_subscription.DEFAULT_CLAUDE_SUBSCRIPTION_BASE_URL
    models = claude_subscription.fetch_available_models(access_token)
    if not models:
        raise ValueError(
            "Claude Subscription connected, but no usable Claude models were discovered for this account."
        )

    expires_in = tokens.get("expires_in")
    expires_at = None
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expires_at = utcnow_naive() + timedelta(seconds=int(expires_in))

    db = SessionLocal()
    try:
        auth = (
            db.query(ProviderAuthSession)
            .filter(
                ProviderAuthSession.provider == claude_subscription.CLAUDE_SUBSCRIPTION_PROVIDER,
                ProviderAuthSession.owner == owner,
            )
            .first()
        )
        if auth is None:
            auth = ProviderAuthSession(
                id=str(uuid.uuid4())[:8],
                provider=claude_subscription.CLAUDE_SUBSCRIPTION_PROVIDER,
                owner=owner,
                label="Claude Subscription",
                base_url=base,
                auth_mode="claude",
            )
            db.add(auth)
        auth.base_url = base
        auth.access_token = access_token
        auth.refresh_token = refresh_token
        auth.expires_at = expires_at
        auth.last_refresh = utcnow_naive()
        auth.auth_mode = "claude"

        ep = (
            db.query(ModelEndpoint)
            .filter(
                ModelEndpoint.base_url == base,
                ModelEndpoint.provider_auth_id == auth.id,
                ModelEndpoint.owner == owner,
            )
            .first()
        )
        if ep is None:
            ep = ModelEndpoint(
                id=str(uuid.uuid4())[:8],
                name="Claude Subscription",
                base_url=base,
                model_type="llm",
                endpoint_kind="api",
                owner=owner,
            )
            db.add(ep)
        ep.name = "Claude Subscription"
        ep.base_url = base
        ep.api_key = None
        ep.provider_auth_id = auth.id
        ep.is_enabled = True
        ep.supports_tools = True
        ep.model_type = "llm"
        ep.endpoint_kind = "api"
        ep.model_refresh_mode = "manual"
        ep.cached_models = json.dumps(models)
        db.commit()
        result = {
            "id": ep.id,
            "name": ep.name,
            "base_url": ep.base_url,
            "models": models,
        }
    finally:
        db.close()

    try:
        from routes.model_routes import _invalidate_models_cache

        _invalidate_models_cache()
    except Exception:
        pass
    return result


def setup_claude_subscription_routes() -> APIRouter:
    router = APIRouter(prefix="/api/claude-subscription", tags=["claude-subscription"])

    @router.post("/start")
    def start(request: Request):
        require_admin(request)
        pkce = claude_subscription.generate_pkce()
        _PENDING.add(pkce["state"], pkce["code_verifier"], get_current_user(request) or None)
        authorize_url = claude_subscription.build_authorize_url(pkce["code_challenge"], pkce["state"])
        return {"authorize_url": authorize_url, "state": pkce["state"]}

    @router.post("/complete")
    def complete(request: Request, code: str = Form(...), state: str = Form(None)):
        require_admin(request)
        pasted_code, pasted_state = claude_subscription.split_pasted_code(code)
        flow_state = (state or "").strip() or pasted_state
        if not flow_state:
            raise HTTPException(400, "Missing OAuth state. Restart the connection.")
        pending = _PENDING.take(flow_state)
        if not pending:
            raise HTTPException(400, "Unknown or expired login session. Restart the connection.")
        try:
            tokens = claude_subscription.exchange_authorization_code(
                pasted_code, pasted_state or flow_state, pending["code_verifier"]
            )
            result = _provision_endpoint(tokens, pending["owner"])
        except Exception as exc:
            logger.exception("Claude Subscription endpoint provisioning failed")
            raise claude_subscription.to_http_exception(exc)
        return {"status": "authorized", "endpoint": result}

    return router
