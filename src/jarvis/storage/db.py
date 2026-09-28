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
    # 2 — tools: tool calls on messages, audit log, notes, reminders
    """
    ALTER TABLE messages ADD COLUMN tool_calls TEXT;     -- JSON list of ToolCall
    ALTER TABLE messages ADD COLUMN tool_name TEXT;
    ALTER TABLE messages ADD COLUMN tool_call_id TEXT;

    CREATE TABLE tool_calls (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id  INTEGER,
        call_id          TEXT,
        tool             TEXT NOT NULL,
        arguments        TEXT NOT NULL,   -- JSON
        risk             TEXT NOT NULL,
        decision         TEXT NOT NULL,   -- allowed | approved | approved_session | denied
        ok               INTEGER NOT NULL,
        output           TEXT,
        duration_ms      REAL,
        created_at       TEXT NOT NULL
    );
    CREATE INDEX idx_tool_calls_created ON tool_calls(created_at);

    CREATE TABLE notes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        text        TEXT NOT NULL,
        created_at  TEXT NOT NULL
    );

    CREATE TABLE reminders (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        text        TEXT NOT NULL,
        due_at      TEXT NOT NULL,        -- UTC ISO-8601
        status      TEXT NOT NULL DEFAULT 'pending',  -- pending | fired | cancelled
        created_at  TEXT NOT NULL,
        fired_at    TEXT
    );
    CREATE INDEX idx_reminders_due ON reminders(status, due_at);
    """,
    # 3 — long-term memory
    """
    CREATE TABLE memories (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        text             TEXT NOT NULL,
        category         TEXT NOT NULL DEFAULT 'other',  -- profile|preference|person|project|other
        source           TEXT NOT NULL DEFAULT 'told',   -- told (explicit) | learned (extracted)
        conversation_id  INTEGER,
        created_at       TEXT NOT NULL,
        updated_at       TEXT NOT NULL,
        last_used_at     TEXT,
        use_count        INTEGER NOT NULL DEFAULT 0,
        embedding        BLOB,
        embed_model      TEXT
    );
    CREATE INDEX idx_memories_category ON memories(category);
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
