"""Tests for the Claude (Anthropic) subscription OAuth provider.

These import the real helpers from ``src.claude_subscription`` /
``src.llm_core`` / ``src.endpoint_resolver`` so the provider wiring (detection,
auth headers, URL building) is actually exercised. OAuth network calls are
monkeypatched — no real requests are made.
"""

import json

import pytest

from src import claude_subscription as cs
from src import llm_core
from src.endpoint_resolver import build_chat_url, build_models_url, build_headers

SENTINEL = cs.DEFAULT_CLAUDE_SUBSCRIPTION_BASE_URL  # https://api.anthropic.com/oauth


class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


# ── Provider detection ──

class TestDetection:
    def test_is_claude_subscription_base_positive(self):
        assert cs.is_claude_subscription_base(SENTINEL)
        assert cs.is_claude_subscription_base("https://api.anthropic.com/oauth/")

    def test_is_claude_subscription_base_negative(self):
        assert not cs.is_claude_subscription_base("https://api.anthropic.com")
        assert not cs.is_claude_subscription_base("https://api.anthropic.com/v1")
        assert not cs.is_claude_subscription_base("https://api.openai.com/oauth")

    def test_detect_provider_subscription_vs_anthropic(self):
        assert llm_core._detect_provider(SENTINEL) == "claude-subscription"
        assert llm_core._detect_provider("https://api.anthropic.com") == "anthropic"
        assert llm_core._detect_provider("https://api.anthropic.com/v1") == "anthropic"

    def test_is_anthropic_like(self):
        assert llm_core._is_anthropic_like("anthropic")
        assert llm_core._is_anthropic_like("claude-subscription")
        assert not llm_core._is_anthropic_like("openai")


# ── URL building ──

class TestUrls:
    def test_normalize_strips_oauth_sentinel(self):
        assert llm_core._normalize_anthropic_url(SENTINEL) == "https://api.anthropic.com/v1/messages"
        # plain anthropic still works
        assert llm_core._normalize_anthropic_url("https://api.anthropic.com") == "https://api.anthropic.com/v1/messages"

    def test_build_chat_url_keeps_subscription_routing(self):
        chat_url = build_chat_url(SENTINEL)
        # The chat URL must still detect as claude-subscription downstream...
        assert llm_core._detect_provider(chat_url) == "claude-subscription"
        # ...and normalize to the real messages endpoint at request time.
        assert llm_core._normalize_anthropic_url(chat_url) == "https://api.anthropic.com/v1/messages"

    def test_build_models_url(self):
        assert build_models_url(SENTINEL) == "https://api.anthropic.com/v1/models"


# ── Auth headers ──

class TestHeaders:
    def test_oauth_headers_keep_bearer_and_add_beta(self):
        h = llm_core._build_anthropic_headers({"Authorization": "Bearer TOK"}, oauth=True)
        assert h["Authorization"] == "Bearer TOK"
        assert h["anthropic-beta"] == "oauth-2025-04-20"
        assert h["anthropic-version"] == "2023-06-01"
        assert "x-api-key" not in h

    def test_apikey_headers_convert_bearer(self):
        h = llm_core._build_anthropic_headers({"Authorization": "Bearer KEY"})
        assert h["x-api-key"] == "KEY"
        assert "Authorization" not in h
        assert "anthropic-beta" not in h

    def test_oauth_headers_do_not_duplicate_incoming_beta(self):
        h = llm_core._build_anthropic_headers(
            {"Authorization": "Bearer TOK", "anthropic-beta": "something-else"}, oauth=True
        )
        assert h["anthropic-beta"] == "oauth-2025-04-20"

    def test_build_headers_for_subscription(self):
        h = build_headers("ACCESS", SENTINEL)
        assert h["Authorization"] == "Bearer ACCESS"
        assert h["anthropic-beta"] == "oauth-2025-04-20"
        assert h["anthropic-version"] == "2023-06-01"
        assert "x-api-key" not in h

    def test_claude_oauth_headers_helper(self):
        h = cs.claude_oauth_headers("t")
        assert h["Authorization"] == "Bearer t"
        assert h["anthropic-beta"] == "oauth-2025-04-20"
        assert cs.claude_oauth_headers(None).get("Authorization") is None


# ── PKCE / authorize ──

class TestPkce:
    def test_generate_pkce_distinct(self):
        a = cs.generate_pkce()
        b = cs.generate_pkce()
        assert a["code_verifier"] != b["code_verifier"]
        assert a["state"] != b["state"]
        assert a["code_challenge"] and a["code_verifier"]

    def test_authorize_url_contains_params(self):
        pkce = cs.generate_pkce()
        url = cs.build_authorize_url(pkce["code_challenge"], pkce["state"])
        assert cs.CLAUDE_OAUTH_CLIENT_ID in url
        assert "code_challenge_method=S256" in url
        assert pkce["state"] in url
        assert "code=true" in url

    def test_split_pasted_code(self):
        assert cs.split_pasted_code("abc#xyz") == ("abc", "xyz")
        assert cs.split_pasted_code("  abc#xyz  ") == ("abc", "xyz")
        assert cs.split_pasted_code("plain") == ("plain", None)


# ── Token exchange / refresh request shapes (monkeypatched httpx) ──

class TestTokenRequests:
    def test_exchange_authorization_code_shape(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None, follow_redirects=None):
            captured["url"] = url
            captured["json"] = json
            return _FakeResp(200, {"access_token": "A", "refresh_token": "R", "expires_in": 28800})

        monkeypatch.setattr(cs.httpx, "post", fake_post)
        out = cs.exchange_authorization_code("CODE", "STATE", "VERIFIER")
        assert out["access_token"] == "A"
        assert captured["url"] == cs.CLAUDE_OAUTH_TOKEN_URL
        body = captured["json"]
        assert body["grant_type"] == "authorization_code"
        assert body["code"] == "CODE"
        assert body["code_verifier"] == "VERIFIER"
        assert body["client_id"] == cs.CLAUDE_OAUTH_CLIENT_ID
        assert body["redirect_uri"] == cs.CLAUDE_OAUTH_REDIRECT_URI

    def test_refresh_oauth_tokens_shape(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None, follow_redirects=None):
            captured["json"] = json
            return _FakeResp(200, {"access_token": "A2", "expires_in": 28800})

        monkeypatch.setattr(cs.httpx, "post", fake_post)
        out = cs.refresh_oauth_tokens("REFRESH")
        assert out["access_token"] == "A2"
        body = captured["json"]
        assert body["grant_type"] == "refresh_token"
        assert body["refresh_token"] == "REFRESH"
        assert body["client_id"] == cs.CLAUDE_OAUTH_CLIENT_ID

    def test_refresh_without_token_raises_reauth(self):
        with pytest.raises(cs.ClaudeSubscriptionReauthRequired):
            cs.refresh_oauth_tokens("")

    def test_exchange_rate_limited_maps(self, monkeypatch):
        def fake_post(url, json=None, headers=None, timeout=None, follow_redirects=None):
            return _FakeResp(429, {"error": {"type": "rate_limit_error", "message": "slow"}})

        monkeypatch.setattr(cs.httpx, "post", fake_post)
        with pytest.raises(cs.ClaudeSubscriptionRateLimited):
            cs.exchange_authorization_code("C", "S", "V")


# ── Model discovery (monkeypatched httpx) ──

class TestModelDiscovery:
    def test_fetch_available_models_filters_claude(self, monkeypatch):
        payload = {
            "data": [
                {"id": "claude-opus-4-8"},
                {"id": "claude-haiku-4-5-20251001"},
                {"id": "not-a-claude-model"},
                {"id": "claude-sonnet-4-6"},
            ]
        }

        def fake_get(url, headers=None, timeout=None):
            assert "/v1/models" in url
            assert headers.get("anthropic-beta") == "oauth-2025-04-20"
            return _FakeResp(200, payload)

        monkeypatch.setattr(cs.httpx, "get", fake_get)
        models = cs.fetch_available_models("ACCESS")
        assert models == ["claude-opus-4-8", "claude-haiku-4-5-20251001", "claude-sonnet-4-6"]

    def test_fetch_available_models_empty_on_error(self, monkeypatch):
        def fake_get(url, headers=None, timeout=None):
            return _FakeResp(401, {})

        monkeypatch.setattr(cs.httpx, "get", fake_get)
        assert cs.fetch_available_models("ACCESS") == []
        assert cs.fetch_available_models("") == []


# ── Expiry decision (pure) ──

class TestExpiry:
    def test_access_token_is_expiring(self):
        from datetime import datetime, timedelta

        def now():
            return datetime(2026, 1, 1, 12, 0, 0)

        # No expiry stored -> treat as expiring.
        assert cs._access_token_is_expiring(None, now, 300) is True
        # Far future -> not expiring.
        assert cs._access_token_is_expiring(now() + timedelta(hours=2), now, 300) is False
        # Within skew -> expiring.
        assert cs._access_token_is_expiring(now() + timedelta(seconds=60), now, 300) is True
        # Past -> expiring.
        assert cs._access_token_is_expiring(now() - timedelta(seconds=1), now, 300) is True
