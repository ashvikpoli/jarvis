"""ConversationManager: the stateful chat path (CLI today, voice in Phase 4)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from jarvis.conversation.prompt import PromptBuilder
from jarvis.core.types import Message, StreamChunk, Usage
from jarvis.providers import ModelManager
from jarvis.storage import Conversation, ConversationStore


class ConversationNotFound(LookupError):
    pass


def make_title(text: str, limit: int = 60) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


class ConversationManager:
    def __init__(
        self,
        models: ModelManager,
        store: ConversationStore,
        prompts: PromptBuilder,
        *,
        role: str = "chat",
    ):
        self.models = models
        self.store = store
        self.prompts = prompts
        self.role = role

    def new(self, title: str | None = None) -> Conversation:
        return self.store.create(title)

    def get(self, conversation_id: int) -> Conversation:
        conv = self.store.get(conversation_id)
        if conv is None:
            raise ConversationNotFound(f"No conversation with id {conversation_id}")
        return conv

    def history(self, conversation_id: int) -> list[Message]:
        return [m.to_message() for m in self.store.messages(conversation_id) if m.content]

    async def send(
        self,
        conversation_id: int,
        text: str,
        *,
        role: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Save the user message, stream the reply, save the reply.

        If the consumer stops early (Ctrl+C, barge-in, client disconnect) the partial
        reply is saved with interrupted=True. Consume with `contextlib.aclosing` so
        that happens immediately rather than whenever the generator is collected.
        """
        role = role or self.role
        conv = self.get(conversation_id)
        provider, model = self.models.resolve(role)  # fail fast before saving anything

        self.store.add_message(conversation_id, "user", text)
        if not conv.title:
            self.store.rename(conversation_id, make_title(text))

        built = self.prompts.build(self.history(conversation_id))
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        usage: Usage | None = None
        completed = False
        try:
            async for chunk in self.models.stream_chat(built.messages, role, options=options):
                text_parts.append(chunk.text)
                thinking_parts.append(chunk.thinking)
                if chunk.done:
                    usage, completed = chunk.usage, True
                yield chunk
        finally:
            content = "".join(text_parts)
            if content or completed:
                self.store.add_message(
                    conversation_id,
                    "assistant",
                    content,
                    thinking="".join(thinking_parts),
                    model=f"{provider.name}:{model}",
                    usage=usage,
                    interrupted=not completed,
                )
