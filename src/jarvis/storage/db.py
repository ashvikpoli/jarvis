"""SQLite database: one file (data/jarvis.db), WAL mode, ordered migrations.

Calls are synchronous. Local SQLite operations take well under a millisecond, so
running them on the event loop is fine at this scale and keeps the code simple.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

# Append-only. Never edit a migration that has shipped; add a new one.
MIGRATIONS: list[str] = [
    # 1 — conversations
    """
    CREATE TABLE conversations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        title       TEXT,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    );
    CREATE TABLE messages (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id   INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role              TEXT NOT NULL,
        content           TEXT NOT NULL,
        thinking          TEXT NOT NULL DEFAULT '',
        created_at        TEXT NOT NULL,
        model             TEXT,
        prompt_tokens     INTEGER,
        completion_tokens INTEGER,
        ttft_ms           REAL,
        answer_ttft_ms    REAL,
        interrupted       INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX idx_messages_conversation ON messages(conversation_id, id);
    CREATE INDEX idx_conversations_updated ON conversations(updated_at);
    """,
]


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI may call us from worker threads; the lock
        # serialises access so that's safe.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                self.conn.execute("PRAGMA journal_mode = WAL")
        self.migrate()

    @property
    def schema_version(self) -> int:
        with self.lock:
            self.conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
            row = self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            return row[0] or 0

    def migrate(self) -> int:
        """Apply any pending migrations. Safe to call repeatedly."""
        with self.lock:
            current = self.schema_version
            for version, sql in enumerate(MIGRATIONS[current:], start=current + 1):
                # Schema change + version bump in one transaction: all or nothing.
                self.conn.executescript(
                    f"BEGIN;\n{sql}\nINSERT INTO schema_version VALUES ({int(version)});\nCOMMIT;"
                )
            return self.schema_version

    def close(self) -> None:
        with self.lock:
            self.conn.close()
