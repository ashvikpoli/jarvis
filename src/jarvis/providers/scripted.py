"""ScriptedProvider: a fake model that plays back pre-written turns (for tests).

ScriptedProvider([
    [ToolCall(name="files_list", arguments={})],   # 1st call: ask for a tool
    "Here are your files.",                        # 2nd call: answer
])
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from jarvis.core.types import Message, ProviderHealth, StreamChunk, ToolCall, Usage
from jarvis.providers.base import ModelProvider

Turn = str | list[ToolCall]


class ScriptedProvider(ModelProvider):
    name = "scripted"

    def __init__(self, turns: Sequence[Turn] = (), *, default: str = "OK."):
        self.turns = list(turns)
        self.default = default
        self.calls: list[dict[str, Any]] = []  # what each stream_chat received

    async def stream_chat(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        options: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.calls.append({"messages": list(messages), "tools": tools, "options": options})
        turn = self.turns.pop(0) if self.turns else self.default
        if isinstance(turn, str):
            for i, word in enumerate(turn.split(" ")):
                yield StreamChunk(text=word if i == 0 else " " + word)
            n = len(turn.split())
        else:
            yield StreamChunk(tool_calls=list(turn))
            n = 5
        yield StreamChunk(
            done=True,
            usage=Usage(
                prompt_tokens=10, completion_tokens=n, ttft_ms=5, answer_ttft_ms=5, total_ms=50
            ),
        )

    async def embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        from jarvis.memory.vectors import toy_embed

        return [toy_embed(t) for t in texts]

    async def list_models(self) -> list[str]:
        return ["main"]

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True)
