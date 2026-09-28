"""The contract every model backend implements (Ollama today; cloud APIs later)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from jarvis.core.types import LoadedModel, Message, ProviderHealth, StreamChunk


class ProviderError(RuntimeError):
    """Raised when a provider can't fulfil a request (unreachable, unknown model, ...)."""


class ModelProvider(ABC):
    name: str

    @abstractmethod
    def stream_chat(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        options: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion.

        `tools` are function specs ({"type": "function", "function": {...}}); if the
        model wants to call some, they arrive as a chunk with `tool_calls` set.

        Contract: yields text/thinking/tool_calls chunks, then exactly one chunk with done=True
        (carrying Usage if available). Must stop promptly and release resources if
        the consuming task is cancelled.
        """

    @abstractmethod
    async def list_models(self) -> list[str]: ...

    async def loaded_models(self) -> list[LoadedModel]:
        return []

    async def load(  # noqa: B027 - optional hook
        self, model: str, *, options: dict[str, Any] | None = None
    ) -> None:
        """Warm a model into memory. No-op for providers that don't need it."""

    async def unload(self, model: str) -> None:  # noqa: B027 - optional hook
        """Free a model's memory. No-op for providers that don't need it."""

    @abstractmethod
    async def health(self) -> ProviderHealth: ...

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network clients etc."""
