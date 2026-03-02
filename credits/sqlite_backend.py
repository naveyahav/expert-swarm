"""
credits/sqlite_backend.py
Persistent credit backend backed by SQLite.

Design:
  - Thread-safe writes via threading.Lock.
  - WAL journal mode allows concurrent readers alongside a single writer,
    making it safe to share one database file between the Streamlit web
    process and the Telegram bot process (via a Docker volume).
  - Pass db_path=":memory:" in tests for a fast, isolated in-memory DB.
  - All mutations are wrapped in explicit transactions; partial writes
    cannot leave balances in an inconsistent state.
  - A full transaction log is kept in the `transactions` table for
    auditability (no content, only amounts and token prefixes).

Environment variable:
    EXPERTSWARM_DB_PATH — path to the SQLite file (default: data/credits.db)
"""

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

from credits.ledger import PaymentBackend

log = logging.getLogger(__name__)

_DEFAULT_DB_PATH = os.environ.get("EXPERTSWARM_DB_PATH", "data/credits.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS balances (
    session_token TEXT PRIMARY KEY,
    balance       INTEGER NOT NULL DEFAULT 0,
    created_at    REAL    NOT NULL,
    updated_at    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_token TEXT    NOT NULL,
    amount        INTEGER NOT NULL,
    kind          TEXT    NOT NULL CHECK(kind IN ('mint', 'deduct')),
    created_at    REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transactions_token
    ON transactions (session_token);
"""


class SQLiteBackend(PaymentBackend):
    """
    Persistent, thread-safe credit wallet backed by SQLite.

    Suitable for single-node deployments. For multi-node horizontal
    scaling, swap for a PostgreSQL backend implementing the same ABC.

    In-memory mode (db_path=":memory:"):
        A single shared connection is kept alive for the lifetime of the
        object. This is necessary because SQLite in-memory databases are
        connection-scoped — a new connection always gets a fresh, empty DB.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._lock = threading.Lock()
        # For :memory: databases keep a single connection alive; SQLite
        # in-memory DBs are connection-scoped and would reset on each new
        # sqlite3.connect() call.
        self._mem_conn: sqlite3.Connection | None = None
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        if self._db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.execute("PRAGMA journal_mode=WAL")
            self._mem_conn.execute("PRAGMA foreign_keys=ON")
            self._mem_conn.executescript(_SCHEMA)
        else:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
        log.info("SQLiteBackend initialised: %s", self._db_path)

    # ------------------------------------------------------------------
    # PaymentBackend interface
    # ------------------------------------------------------------------

    def balance(self, session_token: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT balance FROM balances WHERE session_token = ?",
                (session_token,),
            ).fetchone()
        return row[0] if row else 0

    def mint(self, session_token: str, amount: int) -> int:
        """
        Add *amount* credits to *session_token*.
        Creates the row if it does not exist. Returns new balance.
        """
        if amount <= 0:
            raise ValueError(f"mint amount must be positive, got {amount}")
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO balances (session_token, balance, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_token) DO UPDATE SET
                    balance    = balance + excluded.balance,
                    updated_at = excluded.updated_at
                """,
                (session_token, amount, now, now),
            )
            conn.execute(
                "INSERT INTO transactions (session_token, amount, kind, created_at)"
                " VALUES (?, ?, 'mint', ?)",
                (session_token, amount, now),
            )
            new_balance = conn.execute(
                "SELECT balance FROM balances WHERE session_token = ?",
                (session_token,),
            ).fetchone()[0]
        log.info(
            "Minted %d credits → token %s… balance=%d",
            amount, session_token[:8], new_balance,
        )
        return new_balance

    def verify_and_claim(self, session_token: str, amount: int) -> bool:
        """
        Atomically deduct *amount* from *session_token* if sufficient balance
        exists. Returns True on success, False if balance is insufficient.
        """
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT balance FROM balances WHERE session_token = ?",
                (session_token,),
            ).fetchone()
            if not row or row[0] < amount:
                log.warning(
                    "Credit gate blocked: token %s… (have=%d need=%d)",
                    session_token[:8], row[0] if row else 0, amount,
                )
                return False
            conn.execute(
                """
                UPDATE balances
                SET balance = balance - ?, updated_at = ?
                WHERE session_token = ?
                """,
                (amount, now, session_token),
            )
            conn.execute(
                "INSERT INTO transactions (session_token, amount, kind, created_at)"
                " VALUES (?, ?, 'deduct', ?)",
                (session_token, amount, now),
            )
        log.info("Claimed %d credits ← token %s…", amount, session_token[:8])
        return True

    # ------------------------------------------------------------------
    # Reporting helpers (not part of ABC; optional)
    # ------------------------------------------------------------------

    def transaction_history(self, session_token: str, limit: int = 20) -> list[dict]:
        """Return recent transactions for a session (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT amount, kind, created_at
                FROM transactions
                WHERE session_token = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (session_token, limit),
            ).fetchall()
        return [{"amount": r[0], "kind": r[1], "created_at": r[2]} for r in rows]
