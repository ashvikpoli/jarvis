"""Conversation persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator

from jarvis.core.types import Message, Role, ToolCall, Usage
from jarvis.storage.db import Database


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class Conversation(BaseModel):
    id: int
    title: str | None
    created_at: str
    updated_at: str
    message_count: int = 0


class StoredMessage(BaseModel):
    id: int
    conversation_id: int
    role: Role
    content: str
    thinking: str = ""
    created_at: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_ms: float | None = None
    answer_ttft_ms: float | None = None
    interrupted: bool = False
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_name: str | None = None
    tool_call_id: str | None = None

    @field_validator("tool_calls", mode="before")
    @classmethod
    def _parse_tool_calls(cls, v):
        if v is None or v == "":
            return []
        return json.loads(v) if isinstance(v, str) else v

    def to_message(self) -> Message:
        return Message(
            role=self.role,
            content=self.content,
            tool_calls=self.tool_calls,
            tool_name=self.tool_name,
            tool_call_id=self.tool_call_id,
        )


_CONV_SELECT = """
    SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
    FROM conversations c
"""


class ConversationStore:
    def __init__(self, db: Database):
        self.db = db

    def create(self, title: str | None = None) -> Conversation:
        now = _now()
        with self.db.lock, self.db.conn:
            cur = self.db.conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
        return self.get(cur.lastrowid)  # type: ignore[return-value]

    def get(self, conversation_id: int) -> Conversation | None:
        with self.db.lock:
            row = self.db.conn.execute(
                _CONV_SELECT + " WHERE c.id = ?", (conversation_id,)
            ).fetchone()
        return Conversation(**dict(row)) if row else None

    def latest(self) -> Conversation | None:
        items = self.list(limit=1)
        return items[0] if items else None

    def list(self, limit: int = 20) -> list[Conversation]:
        with self.db.lock:
            rows = self.db.conn.execute(
                _CONV_SELECT + " ORDER BY c.updated_at DESC, c.id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Conversation(**dict(r)) for r in rows]

    def rename(self, conversation_id: int, title: str) -> None:
        with self.db.lock, self.db.conn:
            self.db.conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), conversation_id),
            )

    def delete(self, conversation_id: int) -> bool:
        with self.db.lock, self.db.conn:
            cur = self.db.conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cur.rowcount > 0

    def add_message(
        self,
        conversation_id: int,
        role: Role,
        content: str,
        *,
        thinking: str = "",
        model: str | None = None,
        usage: Usage | None = None,
        interrupted: bool = False,
        tool_calls: list[ToolCall] | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
    ) -> StoredMessage:
        now = _now()
        u = usage or Usage()
        calls_json = json.dumps([c.model_dump() for c in tool_calls]) if tool_calls else None
        with self.db.lock, self.db.conn:
            cur = self.db.conn.execute(
                """INSERT INTO messages (conversation_id, role, content, thinking, created_at,
                       model, prompt_tokens, completion_tokens, ttft_ms, answer_ttft_ms,
                       interrupted, tool_calls, tool_name, tool_call_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    conversation_id,
                    role,
                    content,
                    thinking,
                    now,
                    model,
                    u.prompt_tokens,
                    u.completion_tokens,
                    u.ttft_ms,
                    u.answer_ttft_ms,
                    int(interrupted),
                    calls_json,
                    tool_name,
                    tool_call_id,
                ),
            )
            self.db.conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
            )
            row = self.db.conn.execute(
                "SELECT * FROM messages WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
        return StoredMessage(**dict(row))

    def add(self, conversation_id: int, message: Message, **kw) -> StoredMessage:
        """Store a core Message (convenience for the agent loop)."""
        return self.add_message(
            conversation_id,
            message.role,
            message.content,
            tool_calls=message.tool_calls,
            tool_name=message.tool_name,
            tool_call_id=message.tool_call_id,
            **kw,
        )

    def messages(self, conversation_id: int) -> list[StoredMessage]:
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
            ).fetchall()
        return [StoredMessage(**dict(r)) for r in rows]
