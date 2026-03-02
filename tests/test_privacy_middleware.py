"""
tests/test_privacy_middleware.py
Unit tests for privacy/middleware.py — PrivacyMiddleware.

All router.route() calls are monkeypatched so no model is loaded.

Run with:
    pytest tests/test_privacy_middleware.py -v
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import router  # imported early so monkeypatch can target it
from privacy.middleware import PrivacyMiddleware


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class TestSessionManagement:

    def test_create_session_returns_64_hex_token(self):
        mw = PrivacyMiddleware()
        token = mw.create_session()
        assert len(token) == 64
        assert all(c in "0123456789abcdef" for c in token)

    def test_session_exists_after_create(self):
        mw = PrivacyMiddleware()
        token = mw.create_session()
        assert mw.session_exists(token) is True

    def test_unknown_token_does_not_exist(self):
        mw = PrivacyMiddleware()
        assert mw.session_exists("not-a-real-token") is False

    def test_destroy_session_removes_it(self):
        mw = PrivacyMiddleware()
        token = mw.create_session()
        mw.destroy_session(token)
        assert mw.session_exists(token) is False

    def test_destroy_nonexistent_session_is_safe(self):
        mw = PrivacyMiddleware()
        mw.destroy_session("nonexistent")  # must not raise

    def test_tokens_are_unique(self):
        """100 sequential creates must produce 100 distinct tokens."""
        mw = PrivacyMiddleware()
        tokens = {mw.create_session() for _ in range(100)}
        assert len(tokens) == 100

    def test_opaque_identifier_does_not_appear_in_token(self):
        """The raw opaque_identifier must not be recoverable from the token."""
        mw = PrivacyMiddleware()
        secret = b"super-secret-user-id"
        token = mw.create_session(opaque_identifier=secret)
        assert secret.decode() not in token
        assert secret.hex() not in token


# ---------------------------------------------------------------------------
# Session TTL / eviction
# ---------------------------------------------------------------------------

class TestSessionEviction:

    def test_evict_expired_removes_old_session(self):
        mw = PrivacyMiddleware(max_age_seconds=60)
        token = mw.create_session()
        # Back-date to simulate an expired session.
        mw._sessions[token]["created_at"] -= 120
        mw._evict_expired()
        assert mw.session_exists(token) is False

    def test_evict_expired_keeps_fresh_session(self):
        mw = PrivacyMiddleware(max_age_seconds=3600)
        token = mw.create_session()
        mw._evict_expired()
        assert mw.session_exists(token) is True

    def test_evict_only_removes_expired(self):
        mw = PrivacyMiddleware(max_age_seconds=60)
        fresh = mw.create_session()
        stale = mw.create_session()
        mw._sessions[stale]["created_at"] -= 120
        mw._evict_expired()
        assert mw.session_exists(fresh) is True
        assert mw.session_exists(stale) is False

    def test_create_session_triggers_eviction(self):
        mw = PrivacyMiddleware(max_age_seconds=60)
        stale = mw.create_session()
        mw._sessions[stale]["created_at"] -= 120
        # Creating a new session should evict the stale one.
        mw.create_session()
        assert mw.session_exists(stale) is False

    def test_handle_triggers_eviction(self, monkeypatch):
        mw = PrivacyMiddleware(max_age_seconds=60)
        stale = mw.create_session()
        mw._sessions[stale]["created_at"] -= 120
        fresh = mw.create_session()
        monkeypatch.setattr(router, "route", lambda prompt, expert=None: "ok")
        mw.handle(fresh, "hello")
        assert mw.session_exists(stale) is False


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------

class TestHandle:

    def test_handle_rejects_invalid_token(self):
        mw = PrivacyMiddleware()
        result = mw.handle("bad-token", "Hello")
        assert result == "Invalid or expired session."

    def test_handle_rejects_expired_token(self):
        mw = PrivacyMiddleware(max_age_seconds=60)
        token = mw.create_session()
        mw._sessions[token]["created_at"] -= 120
        result = mw.handle(token, "Hello")
        assert result == "Invalid or expired session."

    def test_handle_blocks_on_failed_credit_check(self):
        mw = PrivacyMiddleware()
        token = mw.create_session()
        result = mw.handle(token, "Hello", credit_check=lambda t, c: False)
        assert result == "Insufficient credits."

    def test_handle_passes_when_credit_check_succeeds(self, monkeypatch):
        mw = PrivacyMiddleware()
        token = mw.create_session()
        monkeypatch.setattr(router, "route", lambda prompt, expert=None: "ok")
        result = mw.handle(token, "Hello", credit_check=lambda t, c: True)
        assert result == "ok"

    def test_handle_calls_router_route(self, monkeypatch):
        mw = PrivacyMiddleware()
        token = mw.create_session()
        monkeypatch.setattr(router, "route", lambda prompt, expert=None: "mocked response")
        assert mw.handle(token, "test prompt", expert="base") == "mocked response"

    def test_handle_passes_expert_to_router(self, monkeypatch):
        mw = PrivacyMiddleware()
        token = mw.create_session()
        captured = {}

        def fake_route(prompt, expert=None):
            captured["expert"] = expert
            return "ok"

        monkeypatch.setattr(router, "route", fake_route)
        mw.handle(token, "prompt", expert="coder")
        assert captured["expert"] == "coder"

    def test_handle_passes_prompt_to_router(self, monkeypatch):
        mw = PrivacyMiddleware()
        token = mw.create_session()
        captured = {}

        def fake_route(prompt, expert=None):
            captured["prompt"] = prompt
            return "ok"

        monkeypatch.setattr(router, "route", fake_route)
        mw.handle(token, "my secret prompt", expert="base")
        assert captured["prompt"] == "my secret prompt"

    def test_handle_without_credit_check_still_routes(self, monkeypatch):
        """credit_check=None means no gate — request proceeds unconditionally."""
        mw = PrivacyMiddleware()
        token = mw.create_session()
        monkeypatch.setattr(router, "route", lambda prompt, expert=None: "free response")
        result = mw.handle(token, "hello")
        assert result == "free response"

    def test_handle_returns_router_response_verbatim(self, monkeypatch):
        mw = PrivacyMiddleware()
        token = mw.create_session()
        monkeypatch.setattr(router, "route", lambda prompt, expert=None: "  exact output  ")
        result = mw.handle(token, "q")
        assert result == "  exact output  "
