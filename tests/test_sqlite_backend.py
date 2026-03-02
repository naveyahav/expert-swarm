"""
tests/test_sqlite_backend.py
Unit tests for credits/sqlite_backend.py — SQLiteBackend.

All tests use db_path=":memory:" so no disk I/O occurs and each test
gets a fully isolated, fresh database.

Run with:
    pytest tests/test_sqlite_backend.py -v
"""

import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from credits.sqlite_backend import SQLiteBackend


def fresh() -> SQLiteBackend:
    """Return a new in-memory SQLiteBackend instance."""
    return SQLiteBackend(db_path=":memory:")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestSQLiteInit:

    def test_init_creates_tables(self):
        """Tables must exist after construction."""
        b = fresh()
        with b._connect() as conn:
            tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "balances" in tables
        assert "transactions" in tables

    def test_wal_mode_enabled_on_disk(self, tmp_path):
        """WAL journal mode is set for disk-based databases."""
        db_path = str(tmp_path / "test.db")
        b = SQLiteBackend(db_path=db_path)
        with b._connect() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_memory_mode_for_in_memory_db(self):
        """SQLite does not support WAL for :memory: databases; mode is 'memory'."""
        b = fresh()
        with b._connect() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "memory"

    def test_disk_path_creates_parent_dir(self, tmp_path):
        db_path = str(tmp_path / "subdir" / "credits.db")
        b = SQLiteBackend(db_path=db_path)
        assert Path(db_path).exists()


# ---------------------------------------------------------------------------
# balance()
# ---------------------------------------------------------------------------

class TestBalance:

    def test_unknown_token_balance_is_zero(self):
        assert fresh().balance("tok") == 0

    def test_balance_after_mint(self):
        b = fresh()
        b.mint("tok", 10)
        assert b.balance("tok") == 10

    def test_balance_is_per_token(self):
        b = fresh()
        b.mint("a", 5)
        assert b.balance("b") == 0


# ---------------------------------------------------------------------------
# mint()
# ---------------------------------------------------------------------------

class TestMint:

    def test_mint_returns_new_balance(self):
        b = fresh()
        assert b.mint("tok", 10) == 10

    def test_mint_is_additive(self):
        b = fresh()
        b.mint("tok", 5)
        b.mint("tok", 7)
        assert b.balance("tok") == 12

    def test_mint_zero_raises(self):
        with pytest.raises(ValueError):
            fresh().mint("tok", 0)

    def test_mint_negative_raises(self):
        with pytest.raises(ValueError):
            fresh().mint("tok", -1)

    def test_mint_records_transaction(self):
        b = fresh()
        b.mint("tok", 10)
        history = b.transaction_history("tok")
        assert len(history) == 1
        assert history[0]["kind"] == "mint"
        assert history[0]["amount"] == 10

    def test_mint_creates_row_on_first_call(self):
        b = fresh()
        b.mint("new_tok", 3)
        assert b.balance("new_tok") == 3


# ---------------------------------------------------------------------------
# verify_and_claim()
# ---------------------------------------------------------------------------

class TestVerifyAndClaim:

    def test_sufficient_balance_returns_true(self):
        b = fresh()
        b.mint("tok", 10)
        assert b.verify_and_claim("tok", 3) is True
        assert b.balance("tok") == 7

    def test_insufficient_balance_returns_false(self):
        b = fresh()
        b.mint("tok", 2)
        assert b.verify_and_claim("tok", 5) is False
        assert b.balance("tok") == 2  # unchanged

    def test_exact_balance_succeeds(self):
        b = fresh()
        b.mint("tok", 3)
        assert b.verify_and_claim("tok", 3) is True
        assert b.balance("tok") == 0

    def test_zero_balance_fails(self):
        assert fresh().verify_and_claim("tok", 1) is False

    def test_deduct_records_transaction(self):
        b = fresh()
        b.mint("tok", 10)
        b.verify_and_claim("tok", 3)
        history = b.transaction_history("tok")
        kinds = [r["kind"] for r in history]
        assert "deduct" in kinds

    def test_concurrent_claims_are_atomic(self):
        """
        Thread-safety: only one of two concurrent claims against
        a balance of 1 should succeed.
        """
        b = fresh()
        b.mint("tok", 1)
        results = []

        def claim():
            results.append(b.verify_and_claim("tok", 1))

        t1 = threading.Thread(target=claim)
        t2 = threading.Thread(target=claim)
        t1.start(); t2.start()
        t1.join();  t2.join()

        assert results.count(True)  == 1
        assert results.count(False) == 1
        assert b.balance("tok") == 0


# ---------------------------------------------------------------------------
# transaction_history()
# ---------------------------------------------------------------------------

class TestTransactionHistory:

    def test_history_newest_first(self):
        b = fresh()
        b.mint("tok", 5)
        b.mint("tok", 3)
        history = b.transaction_history("tok")
        assert len(history) == 2
        assert history[0]["amount"] == 3  # newest first

    def test_history_respects_limit(self):
        b = fresh()
        for _ in range(10):
            b.mint("tok", 1)
        assert len(b.transaction_history("tok", limit=5)) == 5

    def test_history_empty_for_unknown_token(self):
        assert fresh().transaction_history("ghost") == []


# ---------------------------------------------------------------------------
# CreditLedger integration with SQLiteBackend
# ---------------------------------------------------------------------------

class TestCreditLedgerWithSQLite:

    def test_ledger_uses_sqlite_backend(self):
        from credits.ledger import CreditLedger
        ledger = CreditLedger(backend=SQLiteBackend(db_path=":memory:"))
        ledger.mint("tok", 10)
        assert ledger.balance("tok") == 10

    def test_ledger_check_and_deduct_with_sqlite(self):
        from credits.ledger import CreditLedger
        ledger = CreditLedger(backend=SQLiteBackend(db_path=":memory:"))
        ledger.mint("tok", 5)
        assert ledger.check_and_deduct("tok", 1) is True
        assert ledger.balance("tok") == 4

    def test_ledger_backend_selection_via_env(self, monkeypatch, tmp_path):
        """EXPERTSWARM_BACKEND=sqlite must produce a SQLiteBackend."""
        import credits.ledger as ledger_mod
        monkeypatch.setenv("EXPERTSWARM_BACKEND", "sqlite")
        monkeypatch.setenv("EXPERTSWARM_DB_PATH", str(tmp_path / "test.db"))
        backend = ledger_mod._default_backend()
        assert isinstance(backend, SQLiteBackend)

    def test_ledger_default_backend_is_mock(self, monkeypatch):
        """Without EXPERTSWARM_BACKEND, MockBackend is used."""
        from credits.ledger import MockBackend, _default_backend
        monkeypatch.delenv("EXPERTSWARM_BACKEND", raising=False)
        assert isinstance(_default_backend(), MockBackend)
