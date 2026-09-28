"""ToolRegistry, PermissionPolicy and the audit log."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

from jarvis.storage import Database
from jarvis.tools.base import Risk, Tool, ToolContext


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def specs(self, ctx: ToolContext) -> list[dict[str, Any]]:
        return [t.spec(ctx) for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)


# ------------------------------------------------------------------ permissions


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


ApprovalAnswer = Literal["once", "session", "deny"]


@dataclass
class ApprovalRequest:
    tool: Tool
    args: BaseModel
    risk: Risk
    description: str

    @property
    def can_allow_for_session(self) -> bool:
        return self.risk != Risk.DESTRUCTIVE


Approver = Callable[[ApprovalRequest], Awaitable[ApprovalAnswer]]


DEFAULT_RISK_DECISIONS = {
    Risk.SAFE: Decision.ALLOW,
    Risk.CHANGE: Decision.ASK,
    Risk.DESTRUCTIVE: Decision.ASK,
}


class PermissionPolicy:
    """Decides allow/ask/deny for a tool call.

    Order: per-tool override → risk default. Destructive calls can never be
    auto-allowed by "always this session"; a config override is the only way.
    """

    def __init__(
        self,
        risk_defaults: Mapping[Risk | str, Decision | str] | None = None,
        overrides: Mapping[str, Decision | str] | None = None,
    ):
        self.risk_defaults = dict(DEFAULT_RISK_DECISIONS)
        for k, v in (risk_defaults or {}).items():
            self.risk_defaults[Risk(k)] = Decision(v)
        self.overrides = {k: Decision(v) for k, v in (overrides or {}).items()}
        self.session_allowed: set[str] = set()

    def decide(
        self, tool: Tool, args: BaseModel, ctx: ToolContext | None = None
    ) -> tuple[Decision, Risk]:
        risk = tool.risk_in(args, ctx)
        # Escalated risk (e.g. overwrite=True) ignores a lower-risk "allow" override.
        override = self.overrides.get(tool.name)
        if override is not None and (risk == tool.risk or override != Decision.ALLOW):
            return override, risk
        if risk != Risk.DESTRUCTIVE and tool.name in self.session_allowed:
            return Decision.ALLOW, risk
        return self.risk_defaults[risk], risk

    def allow_for_session(self, tool_name: str) -> None:
        self.session_allowed.add(tool_name)


# ------------------------------------------------------------------ audit


class AuditEntry(BaseModel):
    id: int
    conversation_id: int | None
    call_id: str | None
    tool: str
    arguments: str
    risk: str
    decision: str
    ok: bool
    output: str | None
    duration_ms: float | None
    created_at: str


class ToolAuditLog:
    def __init__(self, db: Database):
        self.db = db

    def record(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        risk: Risk,
        decision: str,
        ok: bool,
        output: str,
        duration_ms: float | None = None,
        conversation_id: int | None = None,
        call_id: str | None = None,
    ) -> None:
        with self.db.lock, self.db.conn:
            self.db.conn.execute(
                """INSERT INTO tool_calls (conversation_id, call_id, tool, arguments, risk,
                       decision, ok, output, duration_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    conversation_id,
                    call_id,
                    tool,
                    json.dumps(arguments, default=str),
                    str(risk),
                    decision,
                    int(ok),
                    output[:2000],
                    duration_ms,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )

    def recent(self, limit: int = 30) -> list[AuditEntry]:
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT * FROM tool_calls ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [AuditEntry(**dict(r)) for r in rows]
