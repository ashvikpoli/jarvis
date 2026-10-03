"""Search your vault by meaning: notes are cut into chunks, embedded, and stored in SQLite.

What's indexed: everything you write — Notes/, the "My notes" part of every page,
Journal entries, any other folders you create. Not indexed: Jarvis's own parts of
pages (those facts are already in memory), Home, Reminders, Inbox (until filed) and
— unless enabled — conversation transcripts.
"""

from __future__ import annotations

import logging
import re

from jarvis.memory.service import Embedder
from jarvis.memory.vectors import cosine, keyword_score, pack, unpack
from jarvis.storage import Database
from jarvis.vault.markdown import parse_page

log = logging.getLogger(__name__)

QUERY_PREFIX = (
    "Instruct: Given a question from the user, retrieve passages from their personal "
    "notes that answer it\nQuery: "
)
CHUNK_CHARS = 1200
MAX_PER_CYCLE = 200  # chunks embedded per sync cycle (the rest follow next cycle)
_HEAD = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.M)
SKIP = ("Home.md", "Reminders.md")


def indexable(path: str, *, conversations: bool = False) -> bool:
    if path in SKIP or path.startswith("Inbox/"):
        return False
    if path.startswith("Conversations/"):
        return conversations
    return path.endswith(".md") or path.endswith(".txt")


def page_text(path: str, text: str) -> str:
    """The part of a file worth searching (yours, plus Jarvis's notes)."""
    if not text.startswith("---") and "jarvis:start" not in text:
        return text
    page = parse_page(text)
    if page.region is None:
        return page.before
    keep_region = path.startswith(("Notes/", "Conversations/"))
    region = page.region if keep_region and "🔒" not in page.region else ""
    return "\n".join(x for x in (page.before, region, page.after) if x.strip())


def chunk(path: str, text: str, size: int = CHUNK_CHARS) -> list[tuple[str, str]]:
    """[(heading, text)] — split at headings, then paragraphs."""
    title = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    sections: list[tuple[str, str]] = []
    last, heading = 0, ""
    for m in _HEAD.finditer(text):
        sections.append((heading, text[last : m.start()]))
        heading, last = m.group(2).strip(), m.end()
    sections.append((heading, text[last:]))
    out: list[tuple[str, str]] = []
    for head, body in sections:
        body = body.strip()
        if not body or body in ("(anything you write here — Jarvis reads it, never rewrites it)",):
            continue
        if head.lower() == title.lower():
            head = ""
        cur = ""
        for para in re.split(r"\n\s*\n", body):
            if cur and len(cur) + len(para) > size:
                out.append((head, cur.strip()))
                cur = ""
            cur += para.strip() + "\n\n"
            while len(cur) > size * 1.5:  # one huge paragraph
                out.append((head, cur[:size].strip()))
                cur = cur[size:]
        if cur.strip():
            out.append((head, cur.strip()))
    return out


class VaultIndex:
    def __init__(self, db: Database, embedder: Embedder, *, conversations: bool = False):
        self.db = db
        self.embedder = embedder
        self.conversations = conversations

    def _indexed(self) -> dict[str, float]:
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT path, MAX(mtime) AS mtime FROM vault_chunks GROUP BY path"
            ).fetchall()
        return {r["path"]: r["mtime"] for r in rows}

    def count(self) -> int:
        with self.db.lock:
            return self.db.conn.execute(
                "SELECT COUNT(*) FROM vault_chunks WHERE text != ''"
            ).fetchone()[0]

    async def update(self, files: dict, stop=None) -> int:
        """Re-index files that changed; drop files that are gone. Returns chunks indexed."""
        indexed = self._indexed()
        wanted = {p: f for p, f in files.items() if indexable(p, conversations=self.conversations)}
        gone = [p for p in indexed if p not in wanted]
        if gone:
            with self.db.lock, self.db.conn:
                self.db.conn.executemany(
                    "DELETE FROM vault_chunks WHERE path = ?", [(p,) for p in gone]
                )
        done = 0
        for path, f in sorted(wanted.items()):
            if stop is not None and stop.is_set():
                return done
            if indexed.get(path) == f.mtime:
                continue
            pieces = chunk(path, page_text(path, f.text))
            title = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            vecs = None
            if pieces:
                vecs = await self.embedder.docs(
                    [f"{title}{' › ' + h if h else ''}\n{t}" for h, t in pieces]
                )
            model = self.embedder.model_ref if vecs else None
            with self.db.lock, self.db.conn:
                self.db.conn.execute("DELETE FROM vault_chunks WHERE path = ?", (path,))
                rows = [
                    (path, h, t, f.mtime, pack(vecs[i]) if vecs else None, model)
                    for i, (h, t) in enumerate(pieces)
                ] or [(path, "", "", f.mtime, None, None)]  # remember empty files too
                self.db.conn.executemany(
                    "INSERT INTO vault_chunks (path, heading, text, mtime, embedding, embed_model)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
            done += len(pieces)
            if done >= MAX_PER_CYCLE:
                break
        await self._embed_missing()
        return done

    async def _embed_missing(self, limit: int = 64) -> None:
        model = self.embedder.model_ref
        if not model:
            return
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT id, path, heading, text FROM vault_chunks WHERE text != '' AND "
                "(embedding IS NULL OR embed_model IS NOT ?) LIMIT ?",
                (model, limit),
            ).fetchall()
        if not rows:
            return
        vecs = await self.embedder.docs([f"{r['heading']}\n{r['text']}" for r in rows])
        if not vecs:
            return
        with self.db.lock, self.db.conn:
            self.db.conn.executemany(
                "UPDATE vault_chunks SET embedding = ?, embed_model = ? WHERE id = ?",
                [(pack(v), model, r["id"]) for r, v in zip(rows, vecs, strict=True)],
            )

    async def search(self, query: str, *, limit: int = 5, min_score: float = 0.35) -> list[dict]:
        query = query.strip()
        if not query:
            return []
        qvec = await self.embedder.query(query)
        model = self.embedder.model_ref
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT path, heading, text, embedding, embed_model FROM vault_chunks "
                "WHERE text != ''"
            ).fetchall()
        hits = []
        for r in rows:
            score = 0.0
            if qvec is not None and r["embedding"] is not None and r["embed_model"] == model:
                score = cosine(qvec, unpack(r["embedding"]))
            kw = keyword_score(query, f"{r['path']} {r['heading']} {r['text']}")
            score = max(score, kw if kw >= 0.5 else 0.0)
            if score >= min_score:
                hits.append((score, r))
        hits.sort(key=lambda h: h[0], reverse=True)
        out, seen = [], set()
        for score, r in hits:
            key = (r["path"], r["heading"])
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "page": r["path"].rsplit(".", 1)[0],
                    "section": r["heading"] or None,
                    "text": r["text"][:700],
                    "score": round(score, 2),
                }
            )
            if len(out) >= limit:
                break
        return out
