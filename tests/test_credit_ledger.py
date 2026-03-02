"""
tests/test_credit_ledger.py
Unit tests for credits/ledger.py — MockBackend and CreditLedger.

Run with:
    pytest tests/test_credit_ledger.py -v
"""

import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credits.ledger import (
    CreditLedger,
    MockBackend,
    DEMO_MINT_AMOUNT,
    COST_PER_REQUEST,
)


# ---------------------------------------------------------------------------
# MockBackend
# ---------------------------------------------------------------------------

class TestMockBackend:

    def test_initial_balance_is_zero(self):
        assert MockBackend().balance("tok") == 0

    def test_mint_returns_new_balance(self):
        backend = MockBackend()
        result = backend.mint("tok", 10)
        assert result == 10

    def test_mint_is_additive(self):
        backend = MockBackend()
        backend.mint("tok", 5)
        backend.mint("tok", 7)
        assert backend.balance("tok") == 12

    def test_mint_is_per_token(self):
        backend = MockBackend()
        backend.mint("tok_a", 10)
        assert backend.balance("tok_b") == 0

    def test_verify_and_claim_sufficient(self):
        backend = MockBackend()
        backend.mint("tok", 10)
        assert backend.verify_and_claim("tok", 3) is True
        assert backend.balance("tok") == 7

    def test_verify_and_claim_insufficient(self):
        backend = MockBackend()
        backend.mint("tok", 2)
        assert backend.verify_and_claim("tok", 5) is False
        assert backend.balance("tok") == 2  # unchanged — no partial deduction

    def test_verify_and_claim_exact_balance(self):
        backend = MockBackend()
        backend.mint("tok", 3)
        assert backend.verify_and_claim("tok", 3) is True
        assert backend.balance("tok") == 0

    def test_verify_and_claim_zero_balance(self):
        backend = MockBackend()
        assert backend.verify_and_claim("tok", 1) is False

    def test_concurrent_deductions_are_atomic(self):
        """
        Thread-safety: only one of two concurrent claims against a balance of 1
        should succeed. The lock in MockBackend must prevent both from passing.
        """
        backend = MockBackend()
        backend.mint("tok", 1)
        results = []

        def claim():
            results.append(backend.verify_and_claim("tok", 1))

        t1 = threading.Thread(target=claim)
        t2 = threading.Thread(target=claim)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results.count(True) == 1
        assert results.count(False) == 1
        assert backend.balance("tok") == 0


# ---------------------------------------------------------------------------
# CreditLedger
# ---------------------------------------------------------------------------

class TestCreditLedger:

    def test_mint_and_balance(self):
        ledger = CreditLedger()
        ledger.mint("tok", 10)
        assert ledger.balance("tok") == 10

    def test_check_and_deduct_sufficient_returns_true(self):
        ledger = CreditLedger()
        ledger.mint("tok", 5)
        assert ledger.check_and_deduct("tok", 1) is True
        assert ledger.balance("tok") == 4

    def test_check_and_deduct_insufficient_returns_false(self):
        ledger = CreditLedger()
        assert ledger.check_and_deduct("tok", 1) is False

    def test_check_and_deduct_does_not_overdraw(self):
        ledger = CreditLedger()
        ledger.mint("tok", 1)
        ledger.check_and_deduct("tok", 1)  # empties it
        assert ledger.check_and_deduct("tok", 1) is False
        assert ledger.balance("tok") == 0

    def test_demo_mint_amount_is_ten(self):
        assert DEMO_MINT_AMOUNT == 10

    def test_cost_per_request_is_one(self):
        assert COST_PER_REQUEST == 1

    def test_mint_works_on_any_backend(self):
        """mint() must work on any PaymentBackend, not just MockBackend."""
        from credits.sqlite_backend import SQLiteBackend
        ledger = CreditLedger(backend=SQLiteBackend(db_path=":memory:"))
        ledger.mint("tok", 10)
        assert ledger.balance("tok") == 10

    def test_backend_property_returns_instance(self):
        ledger = CreditLedger()
        assert isinstance(ledger.backend, MockBackend)
