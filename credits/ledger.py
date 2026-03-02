"""
credits/ledger.py
Non-custodial simulated credit system.

Design:
  - Credits are keyed by ephemeral session token (never by user identity).
  - The operator never holds payment details — the PaymentBackend ABC
    makes it straightforward to swap MockBackend for a real on-chain
    verifier (e.g. Lightning, ERC-20 balance check) without changing
    the gate logic.
  - check_and_deduct() is the hard gate: it must return True before any
    expert is allowed to run. Same philosophy as validate_expert().
  - All balances live in-process memory only; restart resets everything.
"""

import logging
import threading
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

# Default credit amounts — adjust for your research scenario.
DEMO_MINT_AMOUNT = 10        # credits granted on /start
COST_PER_REQUEST = 1         # credits deducted per router.route() call


# ---------------------------------------------------------------------------
# Payment backend abstraction
# ---------------------------------------------------------------------------

class PaymentBackend(ABC):
    """
    Abstract base for payment verification.

    Swap MockBackend for a real implementation to integrate with an
    on-chain balance check (Lightning invoice, ERC-20 transfer event, etc.)
    without touching CreditLedger or the gate logic.
    """

    @abstractmethod
    def verify_and_claim(self, session_token: str, amount: int) -> bool:
        """
        Verify that *amount* credits are available for *session_token*
        and atomically claim (deduct) them.

        Returns True if the claim succeeded, False otherwise.
        """


class MockBackend(PaymentBackend):
    """
    In-memory mock backend — simulates a non-custodial credit wallet.
    Balances reset on process restart (ephemeral by design).
    """

    def __init__(self) -> None:
        self._balances: dict[str, int] = {}
        self._lock = threading.Lock()

    def balance(self, session_token: str) -> int:
        return self._balances.get(session_token, 0)

    def mint(self, session_token: str, amount: int) -> int:
        """Add *amount* credits to *session_token*. Returns new balance."""
        with self._lock:
            self._balances[session_token] = self._balances.get(session_token, 0) + amount
            new_balance = self._balances[session_token]
        log.info("Minted %d credits for token %s… → balance %d", amount, session_token[:8], new_balance)
        return new_balance

    def verify_and_claim(self, session_token: str, amount: int) -> bool:
        with self._lock:
            current = self._balances.get(session_token, 0)
            if current < amount:
                return False
            self._balances[session_token] = current - amount
        log.info("Claimed %d credits for token %s… → balance %d", amount, session_token[:8], self._balances[session_token])
        return True


# ---------------------------------------------------------------------------
# Credit ledger — the hard gate
# ---------------------------------------------------------------------------

class CreditLedger:
    """
    Wraps a PaymentBackend and exposes the check_and_deduct gate used by
    PrivacyMiddleware. Defaults to MockBackend for research/simulation use.
    """

    def __init__(self, backend: PaymentBackend | None = None) -> None:
        self._backend = backend or MockBackend()

    @property
    def backend(self) -> PaymentBackend:
        return self._backend

    def mint(self, session_token: str, amount: int = DEMO_MINT_AMOUNT) -> int:
        """Top up credits for a session. Returns new balance."""
        if not isinstance(self._backend, MockBackend):
            raise TypeError("mint() is only available on MockBackend.")
        return self._backend.mint(session_token, amount)

    def balance(self, session_token: str) -> int:
        """Return current credit balance for a session."""
        if not isinstance(self._backend, MockBackend):
            raise TypeError("balance() is only available on MockBackend.")
        return self._backend.balance(session_token)

    def check_and_deduct(self, session_token: str, cost: int = COST_PER_REQUEST) -> bool:
        """
        Hard gate — must return True before any expert runs.

        Atomically verifies that *session_token* has at least *cost*
        credits and deducts them. Returns False (blocking the request)
        if the balance is insufficient.
        """
        allowed = self._backend.verify_and_claim(session_token, cost)
        if not allowed:
            log.warning("Credit gate blocked request for token %s… (cost=%d)", session_token[:8], cost)
        return allowed
