"""JarvisApp — the composition root.

This is the ONE place that builds concrete objects and wires them together.
Everything else receives its dependencies through its constructor, which keeps
components testable and swappable without a DI framework.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from jarvis.agent import AgentLoop
from jarvis.conversation import ConversationManager, PromptBuilder
from jarvis.core.config import Settings, load_settings
from jarvis.core.types import Message, StreamChunk
from jarvis.memory.extractor import Change, MemoryExtractor, MemoryWorker
from jarvis.memory.service import Embedder, MemoryService
from jarvis.memory.store import MemoryStore
from jarvis.providers import (
    EchoProvider,
    ModelManager,
    ModelProvider,
    OllamaProvider,
    ProviderError,
)
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
from jarvis.tools.base import FilePolicy
from jarvis.tools.notify import LogNotifier, MultiNotifier, default_notifier
from jarvis.vault.index import QUERY_PREFIX, VaultIndex
from jarvis.vault.sync import Report, VaultSync, VaultWorker

log = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "ollama:qwen3-embedding:0.6b"
DEFAULT_VISION_MODEL = "ollama:qwen3-vl:8b-instruct"  # plain :8b is the slow thinking one
SUBJECTS_DONE = "meta:subjects-assigned"


DESKTOP_SAFE = ("screen_look", "ui_elements", "windows_list")


def _make_desktop(settings: Settings, models: ModelManager):
    """The real Windows desktop, or None (other OS, turned off, or it can't start)."""
    import sys

    d = settings.desktop
    if not d.enabled or sys.platform != "win32":
        return None
    try:
        from jarvis.desktop import Desktop
        from jarvis.desktop.guard import Guard
        from jarvis.desktop.vision import Vision
        from jarvis.desktop.windows import WindowsBackend, init_thread, own_windows

        return Desktop(
            WindowsBackend(),
            Vision(models),
            Guard(takeover_px=d.takeover_px),
            blocked_apps=d.blocked_apps,
            no_typing_apps=d.no_typing_apps,
            max_side=d.max_image_side,
            element_limit=d.element_limit,
            initializer=init_thread,
            own_windows=own_windows(),
        )
    except Exception as e:  # missing package, no desktop session …
        log.warning("screen control unavailable: %s", e)
        return None


def resolve_vault(spec: str) -> Path:
    """ "Documents/obsidian_notes" → the real (OneDrive-aware) Documents folder + rest."""
    raw = spec.strip().replace("\\", "/")
    first, _, rest = raw.partition("/")
    if first.lower() in ("documents", "desktop", "downloads", "pictures", "music", "videos"):
        base = resolve_root(first)
        return (base / rest).resolve() if rest else base
    return resolve_root(spec)


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
        file_policy: FilePolicy | None = None,
        provider_factory: Callable[[], Mapping[str, ModelProvider]] | None = None,
        vault_root: Path | None = None,
        desktop=None,
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
        # ---- long-term memory
        mcfg = settings.memory
        self.memory_store = MemoryStore(db)
        self.memory = self._memory_service(models)
        self.memory.enabled = mcfg.enabled
        self.memory_listeners: list[Callable[[list[Change]], None]] = []
        self._provider_factory = provider_factory
        self.memory_worker = MemoryWorker(self._make_extractor, self._memory_changed)

        self.conversations = ConversationManager(
            models,
            self.store,
            prompts,
            loop=self.loop,
            context_factory=self.tool_context,
            after_turn=self.learn,
        )
        self.scheduler = ReminderScheduler(db, notifier)
        self.apps = apps or AppCatalog()
        self.file_policy = file_policy or FilePolicy()
        # ---- Obsidian vault
        vcfg = settings.vault
        self.vault_root = vault_root
        self.vault_index: VaultIndex | None = None
        if vault_root is not None and vcfg.search:
            self.vault_index = VaultIndex(
                db,
                Embedder(models, query_prefix=QUERY_PREFIX),
                conversations=vcfg.search_conversations,
            )
        if self.vault_index is None and registry.get("vault_search"):
            registry.unregister("vault_search")
        self.vault_worker: VaultWorker | None = None
        self.vault_listeners: list[Callable[[Report], None]] = []
        # ---- screen & mouse/keyboard
        self.desktop = desktop
        if not settings.desktop.enabled:
            for t in [t for t in registry.tools if t.group == "desktop" or t.name in DESKTOP_SAFE]:
                registry.unregister(t.name)

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
        desktop=None,
    ) -> JarvisApp:
        settings = settings or load_settings()
        if providers is None:

            def provider_factory() -> dict[str, ModelProvider]:
                return {"ollama": OllamaProvider(settings.providers.ollama), "echo": EchoProvider()}

            providers = provider_factory()
        else:
            fixed = providers

            def provider_factory() -> Mapping[str, ModelProvider]:
                return fixed

        roles = dict(settings.models.roles)
        roles.setdefault("embed", DEFAULT_EMBED_MODEL)
        roles.setdefault("memory", roles["chat"])  # same model → no VRAM swap
        # A vision chat model sees screenshots itself (one model in VRAM); else qwen3-vl.
        roles.setdefault(
            "vision", roles["chat"] if "-vl" in roles["chat"] else DEFAULT_VISION_MODEL
        )
        models = ModelManager(
            providers,
            roles,
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
        fcfg = tcfg.files

        def _paths(specs: list[str]) -> list[Path]:
            out = []
            for spec in specs:
                expanded = os.path.expandvars(spec)
                if "%" in expanded:  # variable not set on this OS
                    continue
                out.append(Path(expanded).expanduser().resolve())
            return out

        file_policy = FilePolicy(
            read_anywhere=fcfg.read_anywhere,
            outside_writes=fcfg.outside_writes,
            no_access=_paths(fcfg.no_access),
            # Jarvis may never edit its own settings or database: that would let it
            # grant itself permissions.
            no_write=_paths(fcfg.no_write)
            + [settings.config_dir.resolve(), settings.app.data_dir.resolve()],
        )
        vault_root = None
        if settings.vault.enabled and settings.vault.path:
            candidate = resolve_vault(settings.vault.path)
            if candidate.is_dir():
                vault_root = candidate
            else:
                log.info("vault folder %s doesn't exist; Obsidian sync off", candidate)
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
            file_policy=file_policy,
            provider_factory=provider_factory,
            vault_root=vault_root,
            desktop=desktop if desktop is not None else _make_desktop(settings, models),
        )

    # ------------------------------------------------------------ memory

    def _memory_service(self, models: ModelManager) -> MemoryService:
        mcfg = self.settings.memory
        kw = {} if mcfg.query_prefix is None else {"query_prefix": mcfg.query_prefix}
        return MemoryService(
            self.memory_store,
            Embedder(models, **kw),
            top_k=mcfg.top_k,
            min_score=mcfg.min_score,
            core_limit=mcfg.core_limit,
            dedupe_threshold=mcfg.dedupe_threshold,
        )

    def _make_extractor(self) -> MemoryExtractor:
        """Called on the memory thread: fresh model clients for that thread's loop."""
        providers = self._provider_factory() if self._provider_factory else self.models.providers
        models = ModelManager(
            providers,
            {r: str(ref) for r, ref in self.models.roles.items()},
            role_options=self.models.role_options,
            default_options=self.models.default_options,
        )
        return MemoryExtractor(models, self._memory_service(models))

    def _memory_changed(self, changes: list[Change]) -> None:
        for listener in list(self.memory_listeners):
            listener(changes)

    def learn(self, conversation_id: int | None, user_text: str, context: list[Message]) -> None:
        """After a completed turn: let the background worker look for facts to remember."""
        if self.memory.enabled and self.settings.memory.auto_learn:
            self.memory_worker.submit(user_text, context, conversation_id)

    async def remember(self, text: str) -> list[Change]:
        """Explicit "remember this": phrase it like other memories, save verbatim if needed."""
        extractor = MemoryExtractor(self.models, self.memory)
        try:
            changes = await extractor.extract(text, force=True)
        except ProviderError:
            changes = []
        if not changes:
            mem, created = await self.memory.save(text, source="told")
            changes = [Change("add" if created else "update", mem.id, mem.text)]
        return changes

    # ------------------------------------------------------------ vault

    def make_vault_sync(self) -> VaultSync:
        """A VaultSync with its own model clients (it runs on the vault thread)."""
        assert self.vault_root is not None
        v = self.settings.vault
        extractor = self._make_extractor()
        index = None
        if v.search:
            index = VaultIndex(
                self.db,
                Embedder(extractor.models, query_prefix=QUERY_PREFIX),
                conversations=v.search_conversations,
            )
        return VaultSync(
            self.vault_root,
            self.db,
            extractor.service,
            quiet_s=v.quiet_s,
            conversations=v.conversations,
            journal=v.journal,
            inbox=v.inbox,
            user_name=v.user_name,
            extractor=extractor if v.inbox else None,
            index=index,
        )

    async def prepare_vault(self, sync: VaultSync) -> int:
        """Once: give memories from before the vault a subject, so they get pages."""
        from jarvis.memory.extractor import assign_subjects

        with self.db.lock:
            done = self.db.conn.execute(
                "SELECT 1 FROM vault_files WHERE key = ?", (SUBJECTS_DONE,)
            ).fetchone()
        if done:
            return 0
        try:
            n = await assign_subjects(sync.memory.embedder.models, sync.memory.store)
        except ProviderError as e:
            log.info("couldn't sort memories into pages yet: %s", e)
            return 0
        sync._save_row(SUBJECTS_DONE, "", "meta")
        return n

    def start_vault(self) -> bool:
        """Start syncing the vault in the background (no-op if there's no vault)."""
        if self.vault_root is None:
            return False
        if self.vault_worker is None:
            self.vault_worker = VaultWorker(
                self.make_vault_sync,
                interval_s=self.settings.vault.poll_s,
                on_report=self._vault_report,
                prepare=self.prepare_vault,
            )
            self.vault_worker.start()
        return True

    def start_desktop(self) -> bool:
        """Listen for the Esc Esc panic key (chat / voice / serve)."""
        if self.desktop is None:
            return False
        self.desktop.guard.start()
        return True

    def _vault_report(self, report: Report) -> None:
        if report.changes:
            self._memory_changed(report.changes)
        for listener in list(self.vault_listeners):
            listener(report)

    def tool_context(self, conversation_id: int | None = None) -> ToolContext:
        return ToolContext(
            db=self.db,
            file_roots=self.file_roots,
            launcher=self.launcher,
            notifier=self.notifier,
            conversation_id=conversation_id,
            max_read_chars=self.settings.tools.files.max_read_chars,
            apps=self.apps,
            files=self.file_policy,
            memory=self.memory,
            vault=self.vault_index,
            desktop=self.desktop,
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
        system_extra = None
        if ctx.memory is not None:
            recalled = await ctx.memory.prepare(messages)
            messages, system_extra = recalled.messages, recalled.system_extra
        built = self.prompts.build(messages, extra_tokens=tool_tokens, system_extra=system_extra)
        kw = {"on_message": on_message} if on_message else {}
        async for chunk in self.loop.run(
            built.messages, ctx, approver, role=role, options=options, **kw
        ):
            yield chunk

    async def aclose(self) -> None:
        if self.desktop is not None:
            self.desktop.close()
        if self.vault_worker is not None:
            # Let the vault thread finish its cycle before the database closes.
            finished = await asyncio.to_thread(self.vault_worker.stop, 30)
            if not finished:
                log.warning("vault sync still running at shutdown")
        self.scheduler.stop()
        self.memory_worker.close()
        await self.models.aclose()
        self.db.close()
