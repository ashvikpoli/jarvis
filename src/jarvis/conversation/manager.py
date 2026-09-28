"""ConversationManager: the stateful chat path (CLI today, voice in Phase 4)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from jarvis.agent import AgentLoop
from jarvis.conversation.prompt import PromptBuilder
from jarvis.core.types import Message, StreamChunk, Usage
from jarvis.providers import ModelManager
from jarvis.storage import Conversation, ConversationStore
from jarvis.tools import Approver, PermissionPolicy, ToolContext, ToolRegistry


class ConversationNotFound(LookupError):
    pass


def make_title(text: str, limit: int = 60) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


class _NullLauncher:
    def open(self, target: str) -> None:
        raise RuntimeError("no launcher configured")


class _NullNotifier:
    def notify(self, title: str, message: str) -> None:
        pass


class ConversationManager:
    def __init__(
        self,
        models: ModelManager,
        store: ConversationStore,
        prompts: PromptBuilder,
        *,
        loop: AgentLoop | None = None,
        context_factory: Callable[[int | None], ToolContext] | None = None,
        role: str = "chat",
    ):
        self.models = models
        self.store = store
        self.prompts = prompts
        self.loop = loop or AgentLoop(models, ToolRegistry(), PermissionPolicy())
        self.context_factory = context_factory or (
            lambda cid: ToolContext(
                db=store.db,
                file_roots=[],
                launcher=_NullLauncher(),
                notifier=_NullNotifier(),
                conversation_id=cid,
            )
        )
        self.role = role

    def new(self, title: str | None = None) -> Conversation:
        return self.store.create(title)

    def get(self, conversation_id: int) -> Conversation:
        conv = self.store.get(conversation_id)
        if conv is None:
            raise ConversationNotFound(f"No conversation with id {conversation_id}")
        return conv

    def history(self, conversation_id: int) -> list[Message]:
        return [
            m.to_message()
            for m in self.store.messages(conversation_id)
            if m.content or m.tool_calls or m.role == "tool"
        ]

    async def send(
        self,
        conversation_id: int,
        text: str,
        *,
        role: str | None = None,
        options: dict[str, Any] | None = None,
        approver: Approver | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Save the user message, run the agent loop, save everything it produces.

        If the consumer stops early (Ctrl+C, barge-in, client disconnect) the partial
        reply is saved with interrupted=True. Consume with `contextlib.aclosing` so
        that happens immediately rather than whenever the generator is collected.
        Tools that need approval are denied when no `approver` is given.
        """
        role = role or self.role
        conv = self.get(conversation_id)
        provider, model = self.models.resolve(role)  # fail fast before saving anything
        model_ref = f"{provider.name}:{model}"

        self.store.add_message(conversation_id, "user", text)
        if not conv.title:
            self.store.rename(conversation_id, make_title(text))

        ctx = self.context_factory(conversation_id)
        tool_tokens = self.prompts.tokens_for_tools(self.loop.registry.specs(ctx))
        built = self.prompts.build(self.history(conversation_id), extra_tokens=tool_tokens)

        def on_message(
            msg: Message,
            *,
            thinking: str = "",
            usage: Usage | None = None,
            interrupted: bool = False,
        ) -> None:
            self.store.add(
                conversation_id,
                msg,
                thinking=thinking,
                model=model_ref if msg.role == "assistant" else None,
                usage=usage,
                interrupted=interrupted,
            )

        stream = self.loop.run(
            built.messages,
            ctx,
            approver,
            role=role,
            options=options,
            on_message=on_message,
        )
        try:
            async for chunk in stream:
                yield chunk
        finally:
            await stream.aclose()
