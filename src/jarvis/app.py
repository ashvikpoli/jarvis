"""JarvisApp — the composition root.

This is the ONE place that builds concrete objects and wires them together.
Everything else receives its dependencies through its constructor, which keeps
components testable and swappable without a DI framework.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from jarvis.agent import AgentLoop
from jarvis.conversation import ConversationManager, PromptBuilder
from jarvis.core.config import Settings, load_settings
from jarvis.core.types import Message, StreamChunk
from jarvis.providers import EchoProvider, ModelManager, ModelProvider, OllamaProvider
from jarvis.storage import ConversationStore, Database
from jarvis.tools import (
    Approver,
    Launcher,
    Notifier,
    PermissionPolicy,
    ReminderScheduler,
    SystemLauncher,
    ToolAuditLog,
    ToolContext,
    ToolRegistry,
    default_registry,
    resolve_root,
)
from jarvis.tools.apps import AppCatalog
from jarvis.tools.notify import LogNotifier, MultiNotifier, default_notifier

log = logging.getLogger(__name__)


class JarvisApp:
    def __init__(
        self,
        settings: Settings,
        models: ModelManager,
        db: Database,
        prompts: PromptBuilder,
        *,
        registry: ToolRegistry,
        policy: PermissionPolicy,
        launcher: Launcher,
        notifier: MultiNotifier,
        file_roots: list[Path],
        apps: AppCatalog | None = None,
    ):
        self.settings = settings
        self.models = models
        self.db = db
        self.prompts = prompts
        self.registry = registry
        self.policy = policy
        self.launcher = launcher
        self.notifier = notifier
        self.file_roots = file_roots
        self.store = ConversationStore(db)
        self.audit = ToolAuditLog(db)
        self.loop = AgentLoop(
            models, registry, policy, self.audit, max_steps=settings.tools.max_steps
        )
        self.conversations = ConversationManager(
            models, self.store, prompts, loop=self.loop, context_factory=self.tool_context
        )
        self.scheduler = ReminderScheduler(db, notifier)
        self.apps = apps or AppCatalog()

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        providers: Mapping[str, ModelProvider] | None = None,
        *,
        db_path: Path | str | None = None,
        launcher: Launcher | None = None,
        notifier: Notifier | None = None,
        file_roots: list[Path] | None = None,
        apps: AppCatalog | None = None,
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

        tcfg = settings.tools
        registry = ToolRegistry()
        if tcfg.enabled:
            for tool in default_registry().tools:
                if tool.name not in tcfg.disabled:
                    registry.register(tool)
        perms = tcfg.permissions
        policy = PermissionPolicy(
            {"safe": perms.safe, "change": perms.change, "destructive": perms.destructive},
            perms.effective_overrides,
        )
        if file_roots is None:
            file_roots = []
            for spec in tcfg.files.roots:
                root = resolve_root(spec)
                if root.is_dir():
                    file_roots.append(root)
                else:
                    log.info("file root %r (%s) doesn't exist; skipping", spec, root)
        base_notifier = default_notifier() if tcfg.notifications else MultiNotifier(LogNotifier())
        if notifier is not None:
            base_notifier = MultiNotifier(notifier)

        return cls(
            settings,
            models,
            db,
            prompts,
            registry=registry,
            policy=policy,
            launcher=launcher or SystemLauncher(),
            notifier=base_notifier,
            file_roots=file_roots,
            apps=apps,
        )

    def tool_context(self, conversation_id: int | None = None) -> ToolContext:
        return ToolContext(
            db=self.db,
            file_roots=self.file_roots,
            launcher=self.launcher,
            notifier=self.notifier,
            conversation_id=conversation_id,
            max_read_chars=self.settings.tools.files.max_read_chars,
            apps=self.apps,
        )

    async def chat_stream(
        self,
        messages: Sequence[Message],
        role: str = "chat",
        *,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stateless chat, no tools: the caller supplies the whole history; nothing saved.

        Uses the same PromptBuilder as conversations, so evals test what you talk to.
        """
        built = self.prompts.build(messages)
        async for chunk in self.models.stream_chat(built.messages, role, options=options):
            yield chunk

    async def agent_stream(
        self,
        messages: Sequence[Message],
        ctx: ToolContext,
        approver: Approver | None = None,
        *,
        role: str = "chat",
        options: dict[str, Any] | None = None,
        on_message=None,
    ) -> AsyncIterator[StreamChunk]:
        """Stateless chat WITH tools (used by evals). Nothing is saved to conversations."""
        tool_tokens = self.prompts.tokens_for_tools(self.registry.specs(ctx))
        built = self.prompts.build(messages, extra_tokens=tool_tokens)
        kw = {"on_message": on_message} if on_message else {}
        async for chunk in self.loop.run(
            built.messages, ctx, approver, role=role, options=options, **kw
        ):
            yield chunk

    async def aclose(self) -> None:
        self.scheduler.stop()
        await self.models.aclose()
        self.db.close()
