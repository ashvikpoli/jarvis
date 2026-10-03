"""MemoryStore: persistence + search for long-term memories."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from jarvis.memory.vectors import cosine, keyword_score, pack, unpack
from jarvis.storage import Database

Category = Literal["profile", "preference", "person", "project", "other"]
CATEGORIES: tuple[str, ...] = ("profile", "preference", "person", "project", "other")
Source = Literal["told", "learned"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Memory(BaseModel):
    id: int
    text: str
    category: str
    source: str
    conversation_id: int | None = None
    created_at: str
    updated_at: str
    last_used_at: str | None = None
    use_count: int = 0
    embed_model: str | None = None
    subject: str | None = None  # who/what it's about (a person, project, topic); None = the user


class ScoredMemory(BaseModel):
    memory: Memory
    score: float
    how: Literal["vector", "keyword"]


class MemoryStore:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------ CRUD

    def add(
        self,
        text: str,
        *,
        category: str = "other",
        source: str = "told",
        conversation_id: int | None = None,
        embedding: Sequence[float] | None = None,
        embed_model: str | None = None,
        subject: str | None = None,
    ) -> Memory:
        now = _now()
        category = category if category in CATEGORIES else "other"
        with self.db.lock, self.db.conn:
            cur = self.db.conn.execute(
                """INSERT INTO memories (text, category, source, conversation_id, created_at,
                       updated_at, embedding, embed_model, subject)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    text.strip(),
                    category,
                    source,
                    conversation_id,
                    now,
                    now,
                    pack(embedding) if embedding else None,
                    embed_model,
                    self.canonical_subject(subject),
                ),
            )
        return self.get(cur.lastrowid)  # type: ignore[return-value]

    def update(
        self,
        memory_id: int,
        *,
        text: str | None = None,
        category: str | None = None,
        embedding: Sequence[float] | None = None,
        embed_model: str | None = None,
        subject: str | None | object = ...,
    ) -> Memory | None:
        """`subject=...` (the default) keeps it; None makes it about the user."""
        current = self.get(memory_id)
        if current is None:
            return None
        new_cat = category if category in CATEGORIES else current.category
        new_text = (text or current.text).strip()
        sets = ["text = ?", "category = ?", "updated_at = ?"]
        params: list = [new_text, new_cat, _now()]
        if subject is not ...:
            sets.append("subject = ?")
            params.append(self.canonical_subject(subject))  # type: ignore[arg-type]
        if embedding is not None:
            sets += ["embedding = ?", "embed_model = ?"]
            params += [pack(embedding), embed_model]
        elif new_text != current.text:
            sets += ["embedding = NULL", "embed_model = NULL"]  # stale: re-embed later
        with self.db.lock, self.db.conn:
            self.db.conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", (*params, memory_id)
            )
        return self.get(memory_id)

    def delete(self, memory_id: int) -> bool:
        with self.db.lock, self.db.conn:
            cur = self.db.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cur.rowcount > 0

    def clear(self) -> int:
        with self.db.lock, self.db.conn:
            cur = self.db.conn.execute("DELETE FROM memories")
        return cur.rowcount

    def get(self, memory_id: int) -> Memory | None:
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return _row(row) if row else None

    def list(self, *, category: str | None = None, limit: int = 500) -> list[Memory]:
        sql, params = "SELECT * FROM memories", []
        if category:
            sql += " WHERE category = ?"
            params.append(category)
        sql += " ORDER BY category, id LIMIT ?"
        with self.db.lock:
            rows = self.db.conn.execute(sql, (*params, limit)).fetchall()
        return [_row(r) for r in rows]

    def subjects(self) -> list[str]:
        """Everyone/everything memories are about, most-mentioned first."""
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT subject, COUNT(*) n FROM memories WHERE subject IS NOT NULL "
                "GROUP BY subject ORDER BY n DESC, subject"
            ).fetchall()
        return [r["subject"] for r in rows]

    def canonical_subject(self, subject: str | None) -> str | None:
        """Tidy a subject name and reuse an existing spelling ("priya" → "Priya").

        A new name keeps the spelling it was given — it's also a page's file name.
        """
        if not subject:
            return None
        name = " ".join(str(subject).replace("[[", "").replace("]]", "").split()).strip(" .")
        if not name or name.lower() in ("user", "the user", "me", "myself", "none", "null"):
            return None
        for existing in self.subjects():
            if existing.lower() == name.lower():
                return existing
        return name

    def count(self) -> int:
        with self.db.lock:
            return self.db.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def mark_used(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        with self.db.lock, self.db.conn:
            self.db.conn.executemany(
                "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                [(_now(), i) for i in ids],
            )

    def missing_embeddings(self, embed_model: str) -> list[Memory]:
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT * FROM memories WHERE embedding IS NULL OR embed_model IS NOT ?",
                (embed_model,),
            ).fetchall()
        return [_row(r) for r in rows]

    def set_embedding(self, memory_id: int, embedding: Sequence[float], embed_model: str) -> None:
        with self.db.lock, self.db.conn:
            self.db.conn.execute(
                "UPDATE memories SET embedding = ?, embed_model = ? WHERE id = ?",
                (pack(embedding), embed_model, memory_id),
            )

    # ------------------------------------------------------------ search

    def search(
        self,
        query: str,
        *,
        query_vector: Sequence[float] | None = None,
        embed_model: str | None = None,
        limit: int = 5,
        min_score: float = 0.0,
        min_keyword: float = 0.5,
    ) -> list[ScoredMemory]:
        """Rank memories by meaning (vectors from the same model) or, failing that, words.

        Each memory gets the better of its vector score and its keyword score (keyword
        matches are held to `min_keyword`), so exact names still match even when the
        embedding model is missing or a memory hasn't been embedded yet.
        """
        with self.db.lock:
            rows = self.db.conn.execute("SELECT * FROM memories").fetchall()
        scored: list[ScoredMemory] = []
        for r in rows:
            mem = _row(r)
            best, how = 0.0, "keyword"
            vec = (
                unpack(r["embedding"]) if query_vector and r["embed_model"] == embed_model else None
            )
            if vec is not None:
                best, how = cosine(query_vector, vec), "vector"
                passes = best >= min_score
            else:
                passes = False
            kw = keyword_score(query, mem.text)
            if kw >= min_keyword and (not passes or kw > best):
                best, how, passes = max(best, kw), "keyword", True
            if passes:
                scored.append(ScoredMemory(memory=mem, score=round(best, 4), how=how))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:limit]


def _row(r) -> Memory:
    d = dict(r)
    d.pop("embedding", None)
    return Memory(**d)
