"""
tests/test_telegram_utils.py
Unit tests for pure utility functions in interfaces/telegram_bot.py.

Tests cover the privacy layer (user-ID hashing), session management,
input validation helpers, and response truncation — all without
touching the Telegram API.

Run with:
    pytest tests/test_telegram_utils.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

# Import the module; this does NOT start the bot.
import interfaces.telegram_bot as bot


# ---------------------------------------------------------------------------
# User-ID hashing (privacy)
# ---------------------------------------------------------------------------

class TestHashUserId:

    def test_returns_bytes(self):
        assert isinstance(bot._hash_user_id(12345), bytes)

    def test_same_id_same_day_is_deterministic(self):
        assert bot._hash_user_id(99) == bot._hash_user_id(99)

    def test_different_ids_produce_different_hashes(self):
        assert bot._hash_user_id(1) != bot._hash_user_id(2)

    def test_raw_id_not_in_digest(self):
        digest = bot._hash_user_id(12345)
        assert b"12345" not in digest
        assert str(12345).encode() not in digest

    def test_output_is_32_bytes(self):
        # SHA-256 digest is always 32 bytes.
        assert len(bot._hash_user_id(42)) == 32


# ---------------------------------------------------------------------------
# Session & expert state management
# ---------------------------------------------------------------------------

class TestSessionState:

    def setup_method(self):
        """Isolate each test by clearing shared state dicts."""
        bot._user_sessions.clear()
        bot._user_experts.clear()
        bot._user_locks.clear()

    def test_get_or_create_session_returns_string(self):
        token = bot._get_or_create_session(1001)
        assert isinstance(token, str)
        assert len(token) == 64  # HMAC-SHA256 hex

    def test_get_or_create_session_is_idempotent(self):
        assert bot._get_or_create_session(1001) == bot._get_or_create_session(1001)

    def test_different_users_get_different_sessions(self):
        assert bot._get_or_create_session(1) != bot._get_or_create_session(2)

    def test_get_active_expert_defaults_to_none(self):
        assert bot._get_active_expert(1001) is None

    def test_set_and_get_active_expert(self):
        bot._set_active_expert(1001, "coder")
        assert bot._get_active_expert(1001) == "coder"

    def test_clear_active_expert(self):
        bot._set_active_expert(1001, "writer")
        bot._set_active_expert(1001, None)
        assert bot._get_active_expert(1001) is None

    def test_expert_is_per_user(self):
        bot._set_active_expert(1001, "coder")
        bot._set_active_expert(1002, "analyst")
        assert bot._get_active_expert(1001) == "coder"
        assert bot._get_active_expert(1002) == "analyst"

    def test_get_user_lock_returns_asyncio_lock(self):
        import asyncio
        lock = bot._get_user_lock(1001)
        assert isinstance(lock, asyncio.Lock)

    def test_get_user_lock_is_idempotent(self):
        assert bot._get_user_lock(1001) is bot._get_user_lock(1001)

    def test_different_users_get_different_locks(self):
        assert bot._get_user_lock(1) is not bot._get_user_lock(2)


# ---------------------------------------------------------------------------
# Response truncation
# ---------------------------------------------------------------------------

class TestTruncate:

    def test_short_response_unchanged(self):
        text = "Hello world"
        assert bot._truncate(text) == text

    def test_long_response_is_truncated(self):
        text = "x" * 5000
        result = bot._truncate(text)
        assert len(result) <= bot._MAX_RESPONSE_LEN + 50  # truncation suffix adds a few chars
        assert "[… response truncated]" in result

    def test_exact_limit_not_truncated(self):
        text = "a" * bot._MAX_RESPONSE_LEN
        assert bot._truncate(text) == text

    def test_custom_limit(self):
        text = "abcde"
        result = bot._truncate(text, limit=3)
        assert result.startswith("abc")
        assert "truncated" in result


# ---------------------------------------------------------------------------
# Expert keyboard structure
# ---------------------------------------------------------------------------

class TestExpertKeyboard:

    def test_contains_auto_route_button(self):
        manifest = {"experts": {"coder": {"enabled": True}}}
        kb = bot._expert_keyboard(manifest)
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert "expert:__auto__" in all_data

    def test_enabled_experts_appear_as_buttons(self):
        manifest = {
            "experts": {
                "coder":   {"enabled": True},
                "analyst": {"enabled": True},
                "writer":  {"enabled": False},  # disabled — must NOT appear
            }
        }
        kb = bot._expert_keyboard(manifest)
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert "expert:coder" in all_data
        assert "expert:analyst" in all_data
        assert "expert:writer" not in all_data

    def test_disabled_expert_not_in_keyboard(self):
        manifest = {"experts": {"base": {"enabled": False}}}
        kb = bot._expert_keyboard(manifest)
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert "expert:base" not in all_data

    def test_empty_manifest_has_only_auto_route(self):
        kb = bot._expert_keyboard({"experts": {}})
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert all_data == ["expert:__auto__"]
