"""Core data types shared by every layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    role: Role
    content: str

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role="assistant", content=content)


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


class StreamChunk(BaseModel):
    """One piece of a streamed response. Exactly one of text/thinking is usually set."""

    text: str = ""
    thinking: str = ""
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
