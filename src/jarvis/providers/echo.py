"""Offline fake provider: echoes the last user message word by word.

Useful for tests, for demoing the pipeline without a GPU, and later for testing
voice latency without an LLM in the loop.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from jarvis.core.types import Message, ProviderHealth, StreamChunk, Usage
from jarvis.providers.base import ModelProvider


class EchoProvider(ModelProvider):
    name = "echo"

    def __init__(self, delay_s: float = 0.02):
        self.delay_s = delay_s

    async def stream_chat(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        start = time.perf_counter()
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        words = f"You said: {last_user}".split(" ")
        ttft = None
        for i, word in enumerate(words):
            await asyncio.sleep(self.delay_s)
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            yield StreamChunk(text=word if i == 0 else " " + word)
        yield StreamChunk(
            done=True,
            usage=Usage(
                prompt_tokens=sum(len(m.content.split()) for m in messages),
                completion_tokens=len(words),
                ttft_ms=ttft,
                total_ms=(time.perf_counter() - start) * 1000,
            ),
        )

    async def list_models(self) -> list[str]:
        return ["echo"]

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="offline echo provider", version="1")
