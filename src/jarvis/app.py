"""JarvisApp — the composition root.

This is the ONE place that builds concrete objects and wires them together.
Everything else receives its dependencies through its constructor, which keeps
components testable and swappable without a DI framework.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from jarvis.core.config import Settings, load_settings
from jarvis.core.types import Message, StreamChunk
from jarvis.providers import EchoProvider, ModelManager, ModelProvider, OllamaProvider


class JarvisApp:
    def __init__(self, settings: Settings, models: ModelManager):
        self.settings = settings
        self.models = models

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        providers: Mapping[str, ModelProvider] | None = None,
    ) -> JarvisApp:
        settings = settings or load_settings()
        if providers is None:
            providers = {
                "ollama": OllamaProvider(settings.providers.ollama),
                "echo": EchoProvider(),
            }
        return cls(settings, ModelManager(providers, settings.models.roles))

    async def chat_stream(
        self,
        messages: Sequence[Message],
        role: str = "chat",
        *,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a reply. Adds the system prompt unless the caller supplied one.

        Phase 1 replaces this with ConversationManager (history, context window,
        prompt builder); callers won't need to change.
        """
        msgs = list(messages)
        if not msgs or msgs[0].role != "system":
            msgs.insert(0, Message.system(self.settings.app.system_prompt))
        async for chunk in self.models.stream_chat(msgs, role, options=options):
            yield chunk

    async def aclose(self) -> None:
        await self.models.aclose()
