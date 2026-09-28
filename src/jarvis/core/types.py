"""Core data types shared by every layer."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


def _call_id() -> str:
    return uuid.uuid4().hex[:12]


class ToolCall(BaseModel):
    """A model's request to run a tool."""

    id: str = Field(default_factory=_call_id)
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant asking for tools
    tool_name: str | None = None  # on role="tool": which tool produced this result
    tool_call_id: str | None = None  # on role="tool": which call this answers

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, content: str, name: str, call_id: str | None = None) -> Message:
        return cls(role="tool", content=content, tool_name=name, tool_call_id=call_id)


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_ms: float | None = None  # time to first token of any kind (incl. thinking)
    answer_ttft_ms: float | None = None  # time to first ANSWER token — what voice cares about
    total_ms: float | None = None

    @property
    def tokens_per_s(self) -> float | None:
        first = self.ttft_ms if self.ttft_ms is not None else self.answer_ttft_ms
        if self.completion_tokens and self.total_ms and first is not None:
            gen_ms = self.total_ms - first
            return self.completion_tokens / (gen_ms / 1000) if gen_ms > 0 else None
        return None


ToolEventKind = Literal["call", "result", "denied", "error"]


class ToolEvent(BaseModel):
    """Progress report from the agent loop, for UIs to display."""

    kind: ToolEventKind
    call: ToolCall
    output: str = ""  # result text / reason (for result, denied, error)


class StreamChunk(BaseModel):
    """One piece of a streamed response.

    Usually exactly one of: text, thinking, tool_calls, event — or done=True.
    """

    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    event: ToolEvent | None = None
    done: bool = False
    usage: Usage | None = None


class LoadedModel(BaseModel):
    name: str
    size_bytes: int | None = None
    vram_bytes: int | None = None


class ProviderHealth(BaseModel):
    ok: bool
    detail: str = ""
    version: str | None = None
