"""MemoryService: embed, save (with de-duplication), recall into the prompt."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from jarvis.core.types import Message
from jarvis.memory.store import Memory, MemoryStore, ScoredMemory
from jarvis.memory.vectors import normalize
from jarvis.providers import ModelManager, ProviderError

log = logging.getLogger(__name__)

# Qwen3-Embedding works best with an instruction on the query side only.
DEFAULT_QUERY_PREFIX = (
    "Instruct: Given a message from the user, retrieve stored facts about the user "
    "that are relevant to it\nQuery: "
)
CORE_CATEGORIES = ("profile",)


class Embedder:
    """Wraps the `embed` role. Returns None (→ keyword fallback) when unavailable."""

    RETRY_AFTER_S = 60

    def __init__(
        self,
        models: ModelManager,
        role: str = "embed",
        *,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        doc_prefix: str = "",
    ):
        self.models = models
        self.role = role
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self._down_until = 0.0
        self.last_error: str | None = None

    @property
    def model_ref(self) -> str | None:
        ref = self.models.roles.get(self.role)
        return str(ref) if ref else None

    async def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if not texts or self.model_ref is None or time.monotonic() < self._down_until:
            return None
        try:
            vectors = await self.models.embed(texts, self.role)
        except ProviderError as e:
            self.last_error = str(e)
            self._down_until = time.monotonic() + self.RETRY_AFTER_S
            log.info("embeddings unavailable (%s); using keyword recall", e)
            return None
        self.last_error = None
        return [normalize(v) for v in vectors]

    async def docs(self, texts: Sequence[str]) -> list[list[float]] | None:
        return await self._embed([self.doc_prefix + t for t in texts])

    async def query(self, text: str) -> list[float] | None:
        out = await self._embed([self.query_prefix + text])
        return out[0] if out else None


@dataclass
class RecallResult:
    system_extra: str | None
    messages: list[Message]
    used: list[ScoredMemory]
    core: list[Memory]


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        *,
        enabled: bool = True,
        top_k: int = 5,
        min_score: float = 0.5,
        core_limit: int = 15,
        dedupe_threshold: float = 0.9,
    ):
        self.store = store
        self.embedder = embedder
        self.enabled = enabled
        self.top_k = top_k
        self.min_score = min_score
        self.core_limit = core_limit
        self.dedupe_threshold = dedupe_threshold

    # ------------------------------------------------------------ writing

    async def save(
        self,
        text: str,
        *,
        category: str = "other",
        source: str = "told",
        conversation_id: int | None = None,
    ) -> tuple[Memory, bool]:
        """Save a memory, or update the near-identical one. Returns (memory, created)."""
        text = " ".join(text.split())
        vecs = await self.embedder.docs([text])
        vec = vecs[0] if vecs else None
        model = self.embedder.model_ref if vec else None
        dupes = self.store.search(
            text,
            query_vector=vec,
            embed_model=model,
            limit=1,
            min_score=self.dedupe_threshold,
            min_keyword=1.01,  # vectors only
        )
        exact = [m for m in self.store.list() if m.text.lower() == text.lower()]
        existing = exact[0] if exact else (dupes[0].memory if dupes else None)
        if existing is not None:
            updated = self.store.update(
                existing.id, text=text, category=category, embedding=vec, embed_model=model
            )
            return updated or existing, False
        mem = self.store.add(
            text,
            category=category,
            source=source,
            conversation_id=conversation_id,
            embedding=vec,
            embed_model=model,
        )
        return mem, True

    async def update(self, memory_id: int, text: str, category: str | None = None) -> Memory | None:
        vecs = await self.embedder.docs([text])
        return self.store.update(
            memory_id,
            text=" ".join(text.split()),
            category=category,
            embedding=vecs[0] if vecs else None,
            embed_model=self.embedder.model_ref if vecs else None,
        )

    def forget(self, memory_id: int) -> bool:
        return self.store.delete(memory_id)

    async def backfill(self) -> int:
        """Embed memories that have no vector for the current model (e.g. after a switch)."""
        model = self.embedder.model_ref
        if not model:
            return 0
        todo = self.store.missing_embeddings(model)
        if not todo:
            return 0
        vecs = await self.embedder.docs([m.text for m in todo])
        if not vecs:
            return 0
        for m, v in zip(todo, vecs, strict=True):
            self.store.set_embedding(m.id, v, model)
        return len(todo)

    # ------------------------------------------------------------ reading

    def core(self) -> list[Memory]:
        out: list[Memory] = []
        for cat in CORE_CATEGORIES:
            out.extend(self.store.list(category=cat))
        return out[: self.core_limit]

    async def recall(
        self,
        query: str,
        *,
        limit: int | None = None,
        min_score: float | None = None,
        exclude: Sequence[int] = (),
    ) -> list[ScoredMemory]:
        if not query.strip() or not self.store.count():
            return []
        vec = await self.embedder.query(query)
        hits = self.store.search(
            query,
            query_vector=vec,
            embed_model=self.embedder.model_ref,
            limit=(limit or self.top_k) + len(exclude),
            min_score=self.min_score if min_score is None else min_score,
        )
        return [h for h in hits if h.memory.id not in set(exclude)][: limit or self.top_k]

    async def prepare(self, history: Sequence[Message]) -> RecallResult:
        """Add memory to a prompt without disturbing Ollama's prompt cache.

        Core facts go in the system prompt (they rarely change). Memories recalled for
        this message are prepended to the latest user message only — the stored message
        stays as you typed it.
        """
        msgs = list(history)
        if not self.enabled or not self.store.count():
            return RecallResult(None, msgs, [], [])
        core = self.core()
        system_extra = None
        if core:
            lines = "\n".join(f"- (#{m.id}) {m.text}" for m in core)
            system_extra = f"What you remember about the user (long-term memory):\n{lines}"

        idx = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i].role == "user"), None)
        used: list[ScoredMemory] = []
        if idx is not None:
            used = await self.recall(msgs[idx].content, exclude=[m.id for m in core])
            if used:
                lines = "\n".join(f"- (#{s.memory.id}) {s.memory.text}" for s in used)
                block = (
                    "<memory>\nThings you remember about the user that may be relevant "
                    f"(ignore if not):\n{lines}\n</memory>\n\n"
                )
                msgs[idx] = msgs[idx].model_copy(update={"content": block + msgs[idx].content})
        self.store.mark_used([s.memory.id for s in used] + [m.id for m in core])
        return RecallResult(system_extra, msgs, used, core)
