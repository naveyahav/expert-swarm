"""
privacy/middleware.py
Privacy-by-Design middleware — GDPR Article 5(1)(c) data minimisation
and Article 25 privacy-by-default.

Design guarantees:
  - No prompt text, response text, or user identity is ever written to
    disk or emitted to any log handler.
  - Session tokens are ephemeral HMAC digests derived from a process-
    lifetime secret; they are never stored and cannot be reversed to a
    user identity.
  - The structured audit record contains only: timestamp, expert name,
    credit cost, and response length in characters — never content.
  - All session state lives in-process memory only; a process restart
    is a complete data wipe.
"""

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable

# Audit logger — emits only the content-free AuditRecord fields.
# Configure this handler to write to a SIEM or append-only store if needed.
_audit_log = logging.getLogger("expertswarm.audit")


@dataclass
class AuditRecord:
    """
    The only persistent record of a request.
    Contains no prompt text, no response text, no user identity.
    """
    timestamp: float
    session_token: str        # opaque ephemeral token — not a user ID
    expert_name: str
    credit_cost: int
    response_length: int      # character count only


class PrivacyMiddleware:
    """
    Wraps router.route() with data-minimisation guarantees.

    Usage:
        middleware = PrivacyMiddleware()
        session = middleware.create_session()
        response = middleware.handle(session, prompt, expert="coder")
    """

    # Process-lifetime secret for HMAC session token derivation.
    # Generated fresh on every process start — tokens from previous
    # runs are automatically invalidated.
    _PROCESS_SECRET: bytes = os.urandom(32)

    def __init__(self, credit_cost_per_request: int = 1):
        self._cost = credit_cost_per_request
        # In-memory session store: token → metadata dict.
        # No user identity is held here — only the token itself.
        self._sessions: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(self, opaque_identifier: bytes = b"") -> str:
        """
        Derive an ephemeral session token.

        Args:
            opaque_identifier: An arbitrary caller-supplied byte string
                               (e.g. a hashed Telegram user ID). It is
                               used only as HMAC input and is never stored.

        Returns:
            A 64-hex-char session token that is safe to share with the
            caller but cannot be reversed to the source identifier.
        """
        nonce = os.urandom(16)
        token = hmac.new(
            self._PROCESS_SECRET,
            opaque_identifier + nonce,
            hashlib.sha256,
        ).hexdigest()
        # Store only the token itself — no link back to opaque_identifier.
        self._sessions[token] = {"created_at": time.time()}
        return token

    def session_exists(self, token: str) -> bool:
        return token in self._sessions

    def destroy_session(self, token: str) -> None:
        """Explicitly wipe a session from memory."""
        self._sessions.pop(token, None)

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def handle(
        self,
        session_token: str,
        prompt: str,
        expert: str | None = None,
        credit_check: Callable[[str, int], bool] | None = None,
    ) -> str:
        """
        Route a prompt through the privacy gate.

        Args:
            session_token:  Token from create_session().
            prompt:         Raw user input — held in memory only for the
                            duration of this call, never logged.
            expert:         Optional expert override passed to router.route().
            credit_check:   Optional callable(token, cost) → bool. If
                            provided, the request is blocked when it returns
                            False. Wires in the CreditLedger gate.

        Returns:
            The model response string. The caller is responsible for
            displaying it; this layer does not log or store it.
        """
        if not self.session_exists(session_token):
            return "Invalid or expired session."

        if credit_check is not None and not credit_check(session_token, self._cost):
            return "Insufficient credits."

        # Import here to keep startup fast and avoid circular imports.
        import router
        response = router.route(prompt, expert=expert)

        # Emit an audit record — content-free by design.
        record = AuditRecord(
            timestamp=time.time(),
            session_token=session_token,
            expert_name=expert or "auto",
            credit_cost=self._cost,
            response_length=len(response),
        )
        _audit_log.info(
            "request | expert=%s cost=%d resp_len=%d token=%s",
            record.expert_name,
            record.credit_cost,
            record.response_length,
            record.session_token[:8] + "…",   # log only first 8 chars
        )

        return response
