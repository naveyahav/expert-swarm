"""
credits/ledger.py
Credit ledger and payment backend abstraction.

Backend selection (via EXPERTSWARM_BACKEND environment variable):
    "mock"   → MockBackend (default) — in-memory, resets on restart
    "sqlite" → SQLiteBackend         — persists to EXPERTSWARM_DB_PATH
    "stripe" → StripeBackend         — SQLite + Stripe payment verification

Design:
  - Credits are keyed by ephemeral session token, never by user identity.
  - PaymentBackend is a full ABC: all implementations must provide
    verify_and_claim(), mint(), and balance().
  - CreditLedger is backend-agnostic: no isinstance guards, no coupling
    to any concrete implementation.
  - check_and_deduct() is the hard gate: must return True before any
    expert is allowed to run.
"""

import logging
import os
import threading
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

# Default credit amounts.
DEMO_MINT_AMOUNT = 10   # credits granted on session creation / /start
COST_PER_REQUEST = 1    # credits deducted per router.route() call


# ---------------------------------------------------------------------------
# Payment backend abstraction
# ---------------------------------------------------------------------------

class PaymentBackend(ABC):
    """
    Full abstract interface for a credit wallet backend.

    Concrete implementations:
        MockBackend    — in-memory (tests / local dev)
        SQLiteBackend  — persistent file-based (single-node production)
        StripeBackend  — SQLiteBackend + Stripe payment verification
    """

    @abstractmethod
    def verify_and_claim(self, session_token: str, amount: int) -> bool:
        """
        Atomically verify that *amount* credits are available for
        *session_token* and deduct them.
        Returns True on success, False if balance is insufficient.
        """

    @abstractmethod
    def mint(self, session_token: str, amount: int) -> int:
        """Add *amount* credits to *session_token*. Return new balance."""

    @abstractmethod
    def balance(self, session_token: str) -> int:
        """Return current credit balance for *session_token*."""


# ---------------------------------------------------------------------------
# Mock backend (in-memory — default for local dev and tests)
# ---------------------------------------------------------------------------

class MockBackend(PaymentBackend):
    """
    Thread-safe in-memory backend. Balances reset on process restart.
    Use for local development, CI tests, and demos.
    """

    def __init__(self) -> None:
        self._balances: dict[str, int] = {}
        self._lock = threading.Lock()

    def balance(self, session_token: str) -> int:
        return self._balances.get(session_token, 0)

    def mint(self, session_token: str, amount: int) -> int:
        """Add *amount* credits. Returns new balance."""
        with self._lock:
            self._balances[session_token] = (
                self._balances.get(session_token, 0) + amount
            )
            new_balance = self._balances[session_token]
        log.info(
            "Minted %d credits for token %s… → balance %d",
            amount, session_token[:8], new_balance,
        )
        return new_balance

    def verify_and_claim(self, session_token: str, amount: int) -> bool:
        with self._lock:
            current = self._balances.get(session_token, 0)
            if current < amount:
                return False
            self._balances[session_token] = current - amount
        log.info(
            "Claimed %d credits for token %s… → balance %d",
            amount, session_token[:8], self._balances[session_token],
        )
        return True


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _default_backend() -> PaymentBackend:
    """
    Instantiate the backend specified by EXPERTSWARM_BACKEND.

        mock   (default) → MockBackend()
        sqlite           → SQLiteBackend()
        stripe           → StripeBackend()
    """
    name = os.environ.get("EXPERTSWARM_BACKEND", "mock").lower()
    if name == "sqlite":
        from credits.sqlite_backend import SQLiteBackend
        return SQLiteBackend()
    if name == "stripe":
        from credits.stripe_backend import StripeBackend
        return StripeBackend()
    return MockBackend()


# ---------------------------------------------------------------------------
# Credit ledger — the hard gate
# ---------------------------------------------------------------------------

class CreditLedger:
    """
    Backend-agnostic credit ledger.

    The constructor accepts any PaymentBackend. When none is provided,
    the backend is selected from EXPERTSWARM_BACKEND (default: MockBackend).

    Usage:
        ledger = CreditLedger()                         # env-driven
        ledger = CreditLedger(backend=MockBackend())    # explicit
        ledger = CreditLedger(backend=SQLiteBackend())  # explicit persistent
    """

    def __init__(self, backend: PaymentBackend | None = None) -> None:
        self._backend = backend if backend is not None else _default_backend()
        log.info("CreditLedger using backend: %s", type(self._backend).__name__)

    @property
    def backend(self) -> PaymentBackend:
        return self._backend

    def mint(self, session_token: str, amount: int = DEMO_MINT_AMOUNT) -> int:
        """Top up credits for a session. Returns new balance."""
        return self._backend.mint(session_token, amount)

    def balance(self, session_token: str) -> int:
        """Return current credit balance for a session."""
        return self._backend.balance(session_token)

    def check_and_deduct(self, session_token: str, cost: int = COST_PER_REQUEST) -> bool:
        """
        Hard gate — must return True before any expert runs.

        Atomically verifies sufficient balance and deducts *cost* credits.
        Returns False (blocking the request) if balance is insufficient.
        """
        allowed = self._backend.verify_and_claim(session_token, cost)
        if not allowed:
            log.warning(
                "Credit gate blocked request for token %s… (cost=%d)",
                session_token[:8], cost,
            )
        return allowed
