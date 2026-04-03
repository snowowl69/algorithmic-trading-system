"""
SQLite database layer for persistent trade storage.

Stores every trade, position snapshot and equity curve point so nothing is
lost between sessions.  Uses raw sqlite3 — no external ORM dependency.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.logging_config import get_logger
from config.settings import DB_PATH

log = get_logger(__name__)


class TradeDatabase:
    """Thread-safe SQLite wrapper for trade data."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path or DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    # ── connection per thread ─────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self._path), check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    # ── schema ────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL,
                side        TEXT    NOT NULL,
                qty         REAL    NOT NULL,
                entry_price REAL    NOT NULL,
                exit_price  REAL,
                pnl         REAL,
                pnl_pct     REAL,
                strategy    TEXT,
                opened_at   TEXT    NOT NULL,
                closed_at   TEXT,
                status      TEXT    NOT NULL DEFAULT 'open'
            );

            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT    NOT NULL,
                equity    REAL    NOT NULL,
                cash      REAL    NOT NULL,
                pnl       REAL    NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS positions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL UNIQUE,
                side        TEXT    NOT NULL,
                qty         REAL    NOT NULL,
                entry_price REAL    NOT NULL,
                stop_loss   REAL,
                take_profit REAL,
                strategy    TEXT,
                opened_at   TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
            CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        """)
        conn.commit()
        log.info("Trade database ready at %s", self._path)

    # ── Trade CRUD ────────────────────────────────────────────────────

    def insert_trade(self, trade: Dict[str, Any]) -> int:
        conn = self._conn()
        cur = conn.execute(
            """INSERT INTO trades
               (ticker, side, qty, entry_price, exit_price, pnl, pnl_pct,
                strategy, opened_at, closed_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade["ticker"],
                trade.get("side", "buy"),
                trade.get("qty", 0),
                trade.get("entry", trade.get("entry_price", 0)),
                trade.get("exit", trade.get("exit_price")),
                trade.get("pnl"),
                trade.get("pnl_pct"),
                trade.get("strategy", "unknown"),
                trade.get("opened_at", datetime.now(timezone.utc).isoformat()),
                trade.get("closed_at"),
                trade.get("status", "closed"),
            ),
        )
        conn.commit()
        return cur.lastrowid

    def insert_trades_bulk(self, trades: List[Dict[str, Any]]) -> int:
        conn = self._conn()
        rows = []
        for t in trades:
            rows.append((
                t["ticker"],
                t.get("side", "buy"),
                t.get("qty", 0),
                t.get("entry", t.get("entry_price", 0)),
                t.get("exit", t.get("exit_price")),
                t.get("pnl"),
                t.get("pnl_pct"),
                t.get("strategy", "unknown"),
                t.get("opened_at", datetime.now(timezone.utc).isoformat()),
                t.get("closed_at"),
                t.get("status", "closed"),
            ))
        conn.executemany(
            """INSERT INTO trades
               (ticker, side, qty, entry_price, exit_price, pnl, pnl_pct,
                strategy, opened_at, closed_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        log.info("Inserted %d trades into database", len(rows))
        return len(rows)

    def get_all_trades(self) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_closed_trades(self) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM trades WHERE status='closed' ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trades_by_ticker(self, ticker: str) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM trades WHERE ticker=? ORDER BY id", (ticker,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trades_by_strategy(self, strategy: str) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM trades WHERE strategy=? ORDER BY id", (strategy,)
        ).fetchall()
        return [dict(r) for r in rows]

    def trade_count(self) -> int:
        conn = self._conn()
        return conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

    # ── Equity snapshots ──────────────────────────────────────────────

    def insert_equity_snapshot(
        self, equity: float, cash: float, pnl: float = 0.0
    ) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT INTO equity_snapshots (timestamp, equity, cash, pnl) VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), equity, cash, pnl),
        )
        conn.commit()

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM equity_snapshots ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Position snapshots ────────────────────────────────────────────

    def upsert_position(self, pos: Dict[str, Any]) -> None:
        conn = self._conn()
        conn.execute(
            """INSERT INTO positions
               (ticker, side, qty, entry_price, stop_loss, take_profit, strategy, opened_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(ticker) DO UPDATE SET
                   side=excluded.side, qty=excluded.qty,
                   entry_price=excluded.entry_price,
                   stop_loss=excluded.stop_loss,
                   take_profit=excluded.take_profit,
                   strategy=excluded.strategy""",
            (
                pos["ticker"],
                pos.get("side", "buy"),
                pos.get("qty", 0),
                pos.get("entry_price", 0),
                pos.get("stop_loss"),
                pos.get("take_profit"),
                pos.get("strategy", "unknown"),
                pos.get("opened_at", datetime.now(timezone.utc).isoformat()),
            ),
        )
        conn.commit()

    def remove_position(self, ticker: str) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM positions WHERE ticker=?", (ticker,))
        conn.commit()

    def get_open_positions(self) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM positions ORDER BY ticker").fetchall()
        return [dict(r) for r in rows]

    # ── Utility ───────────────────────────────────────────────────────

    def clear_all(self) -> None:
        conn = self._conn()
        conn.executescript("""
            DELETE FROM trades;
            DELETE FROM equity_snapshots;
            DELETE FROM positions;
        """)
        conn.commit()
        log.info("Database cleared")

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
