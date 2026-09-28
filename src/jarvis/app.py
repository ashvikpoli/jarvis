"""JarvisApp — the composition root.

This is the ONE place that builds concrete objects and wires them together.
Everything else receives its dependencies through its constructor, which keeps
components testable and swappable without a DI framework.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from jarvis.conversation import ConversationManager, PromptBuilder
from jarvis.core.config import Settings, load_settings
from jarvis.core.types import Message, StreamChunk
from jarvis.providers import EchoProvider, ModelManager, ModelProvider, OllamaProvider
from jarvis.storage import ConversationStore, Database


class JarvisApp:
    def __init__(
        self,
        settings: Settings,
        models: ModelManager,
        db: Database,
        prompts: PromptBuilder,
    ):
        self.settings = settings
        self.models = models
        self.db = db
        self.prompts = prompts
        self.store = ConversationStore(db)
        self.conversations = ConversationManager(models, self.store, prompts)

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        providers: Mapping[str, ModelProvider] | None = None,
        *,
        db_path: Path | str | None = None,
    ) -> JarvisApp:
        settings = settings or load_settings()
        if providers is None:
            providers = {
                "ollama": OllamaProvider(settings.providers.ollama),
                "echo": EchoProvider(),
            }
        models = ModelManager(
            providers,
            settings.models.roles,
            role_options=settings.models.options,
            default_options={"num_ctx": settings.conversation.context_tokens},
        )
        conv = settings.conversation
        prompts = PromptBuilder(
            settings.app.system_prompt,
            context_tokens=conv.context_tokens,
            reserve_tokens=conv.reserve_tokens,
            include_datetime=conv.include_datetime,
        )
        db = Database(db_path or settings.app.data_dir / "jarvis.db")
        return cls(settings, models, db, prompts)

    async def chat_stream(
        self,
        messages: Sequence[Message],
        role: str = "chat",
        *,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stateless chat: the caller supplies the whole history; nothing is saved.

        Uses the same PromptBuilder as conversations, so evals test what you talk to.
        """
        built = self.prompts.build(messages)
        async for chunk in self.models.stream_chat(built.messages, role, options=options):
            yield chunk

    async def aclose(self) -> None:
        await self.models.aclose()
        self.db.close()
