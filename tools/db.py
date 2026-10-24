"""SQLite persistence layer (the 'persistent data' requirement)."""
import sqlite3
from contextlib import contextmanager

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS expenses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         TEXT    NOT NULL,
    amount       REAL    NOT NULL,
    currency     TEXT    NOT NULL DEFAULT 'LKR',
    amount_lkr   REAL    NOT NULL,
    category     TEXT    NOT NULL,
    description  TEXT    NOT NULL DEFAULT '',
    created_at   TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date);
CREATE TABLE IF NOT EXISTS budgets (
    category       TEXT PRIMARY KEY,
    monthly_limit  REAL NOT NULL
);
"""


@contextmanager
def connect():
    """Open the database (created on first use) and commit/rollback automatically."""
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def build_filters(start, end, category=None):
    clauses, params = [], []
    if start:
        clauses.append("date >= ?")
        params.append(start)
    if end:
        clauses.append("date <= ?")
        params.append(end)
    if category:
        clauses.append("category = ?")
        params.append(category)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params
