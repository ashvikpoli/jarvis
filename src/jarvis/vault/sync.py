"""Two-way sync between Jarvis's database and the Obsidian vault.

Every cycle:
  1. read   — files you changed since Jarvis last wrote them → database
              (memory facts, notes, reminders; moved/renamed pages are followed)
  2. inbox  — files dropped in Inbox/ → moved to Notes/, then learned from
  3. write  — database → pages (only where something changed; your parts kept)
  4. index  — your notes chunked + embedded for vault_search

Safety rules (your data first):
- Your edits are read before anything is written. A file you touched in the last
  few seconds — or that can't be read right now (OneDrive, a lock) — is left alone.
- A page with an edit Jarvis hasn't read yet is never overwritten.
- Facts and notes are only forgotten after they've been gone from the vault for
  `grace_s` (a file that vanishes for a moment isn't a deletion), and never when a
  large part of the vault disappears at once.
- Jarvis never writes over a file it didn't create, and never deletes a page that
  has anything of yours on it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jarvis.memory.extractor import Change
from jarvis.memory.service import MemoryService
from jarvis.storage import Database
from jarvis.vault import pages as P
from jarvis.vault.markdown import (
    build_page,
    has_user_content,
    parse_items,
    parse_page,
    safe_name,
    sha,
)

log = logging.getLogger(__name__)

SKIP_DIRS = {".obsidian", ".trash", ".git", ".sync"}
MEMORY_DIRS = tuple(f"{d}/" for d in P.CATEGORY_FOR_FOLDER)
MEMORY_KINDS = {"about", "person", "project", "topic"}
TWO_WAY = MEMORY_KINDS | {"note", "reminders"}
DELETED = "deleted"  # vault_files.hash for pages you deleted on purpose (not re-created)
PAGE_GONE_S = 60.0  # a whole page missing this long (not a OneDrive blip) is written again
MAX_INBOX_BYTES = 300_000
MAX_INBOX_CHUNKS = 12


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def decode(data: bytes) -> str | None:
    """Text from a file: UTF-8 (with/without BOM), UTF-16, or Windows-1252."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            return None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


@dataclass
class FileInfo:
    path: str
    mtime: float
    size: int
    text: str
    hash: str
    key: str | None  # the `jarvis:` frontmatter value, if it's one of ours


def _info(rel: str, mtime: float, size: int, text: str) -> FileInfo:
    key = parse_page(text).frontmatter.get("jarvis") if text.startswith("---") else None
    return FileInfo(rel, mtime, size, text, sha(text), key if isinstance(key, str) else None)


@dataclass
class Report:
    changes: list[Change] = field(default_factory=list)  # memory changes from your edits
    imported: list[str] = field(default_factory=list)  # human-readable import notes
    written: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    filed: list[str] = field(default_factory=list)  # inbox files processed
    errors: list[str] = field(default_factory=list)
    indexed: int = 0
    skipped_busy: list[str] = field(default_factory=list)


@dataclass
class MemPage:
    file: FileInfo
    category: str | None  # None = About me (category comes from the section)
    subject: str | None
    key: str  # the vault_files key this page belongs under


class VaultSync:
    def __init__(
        self,
        root: Path,
        db: Database,
        memory: MemoryService,
        *,
        quiet_s: float = 4.0,
        grace_s: float = 10.0,
        conversations: bool = True,
        journal: bool = True,
        inbox: bool = True,
        user_name: str = "You",
        extractor=None,  # MemoryExtractor, for Inbox/
        index=None,  # VaultIndex
        clock: Callable[[], float] = time.time,
        stop: threading.Event | None = None,
    ):
        self.root = Path(root)
        self.db = db
        self.memory = memory
        self.quiet_s = quiet_s
        self.grace_s = grace_s
        self.conversations = conversations
        self.journal = journal
        self.inbox = inbox
        self.user_name = user_name
        self.extractor = extractor
        self.index = index
        self.clock = clock
        self.stop = stop or threading.Event()
        self._cache: dict[str, FileInfo] = {}
        self._unreadable: set[str] = set()
        self._conv_cache: dict = {}
        self._missing_since: dict[str, float] = {}
        self._strays: dict[str, list[str]] = {}
        self._inbox_skipped: set[str] = set()
        self._copies: set[str] = set()

    # ------------------------------------------------------------ files

    def _abs(self, rel: str) -> Path:
        return self.root / Path(rel)

    def scan(self) -> dict[str, FileInfo]:
        """All Markdown/text files in the vault (cached by mtime + size).

        A file that's listed but can't be read right now keeps its last known content
        and counts as "busy", so it's neither imported nor overwritten nor deleted.
        """
        found: dict[str, FileInfo] = {}
        unreadable: set[str] = set()
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                if not fn.lower().endswith((".md", ".txt")):
                    continue
                full = Path(dirpath) / fn
                rel = full.relative_to(self.root).as_posix()
                cached = self._cache.get(rel)
                try:
                    st = full.stat()
                    if cached and cached.mtime == st.st_mtime and cached.size == st.st_size:
                        found[rel] = cached
                        continue
                    data = full.read_bytes()
                except OSError:
                    if cached:
                        found[rel] = cached
                    unreadable.add(rel)
                    continue
                text = decode(data)
                if text is None:
                    unreadable.add(rel)
                    if cached:
                        found[rel] = cached
                    continue
                found[rel] = _info(rel, st.st_mtime, st.st_size, text.replace("\r\n", "\n"))
        self._cache, self._unreadable = found, unreadable
        return found

    def _busy(self, info: FileInfo | None) -> bool:
        if info is None:
            return False
        return info.path in self._unreadable or self.clock() - info.mtime < self.quiet_s

    def _write(self, rel: str, text: str) -> None:
        path = self._abs(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".jarvis-tmp")
        tmp.write_bytes(text.encode("utf-8"))
        try:
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        st = path.stat()
        self._cache[rel] = _info(rel, st.st_mtime, st.st_size, text)

    def _remove(self, rel: str) -> None:
        try:
            self._abs(rel).unlink()
        except FileNotFoundError:
            pass
        self._cache.pop(rel, None)

    def _gone_long_enough(self, key: str, after: float | None = None) -> bool:
        first = self._missing_since.setdefault(key, self.clock())
        return self.clock() - first >= (self.grace_s if after is None else after)

    # ------------------------------------------------------------ bookkeeping

    def _rows(self) -> dict[str, dict]:
        with self.db.lock:
            rows = self.db.conn.execute("SELECT * FROM vault_files").fetchall()
        return {r["key"]: dict(r) for r in rows}

    def _save_row(self, key: str, path: str, hash_: str, ids=(), version=None, old=None) -> None:
        ids_json = json.dumps(list(ids or []))
        if old and (old["path"], old["hash"], old["ids"], old["version"]) == (
            path,
            hash_,
            ids_json,
            version,
        ):
            return  # unchanged: don't touch the database
        with self.db.lock, self.db.conn:
            self.db.conn.execute(
                "INSERT INTO vault_files (key, path, hash, ids, version, written_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET path = excluded.path, "
                "hash = excluded.hash, ids = excluded.ids, version = excluded.version, "
                "written_at = excluded.written_at",
                (key, path, hash_, ids_json, version, _now_iso()),
            )

    def _drop_row(self, key: str) -> None:
        with self.db.lock, self.db.conn:
            self.db.conn.execute("DELETE FROM vault_files WHERE key = ?", (key,))

    # ------------------------------------------------------------ the cycle

    async def cycle(self) -> Report:
        report = Report()
        self.root.mkdir(parents=True, exist_ok=True)
        files = self.scan()
        rows = self._rows()
        handled: set[str] = set()  # paths whose edits were read this cycle
        pending: set[str] = set()  # paths with edits not read yet → don't overwrite
        try:
            await self._import_memories(files, rows, report, handled, pending)
            self._import_notes(files, rows, report, handled, pending)
            self._import_reminders(files, rows, report, handled, pending)
        except Exception as e:
            log.exception("vault import failed")
            report.errors.append(f"reading the vault: {e}")
            return report  # don't write anything over edits we failed to read
        self._mark_deleted_pages(files, rows)
        if self.inbox and self.extractor is not None and not self.stop.is_set():
            await self._process_inbox(files, report)
        files, rows = self.scan(), self._rows()
        self._export(files, rows, report, handled, pending)
        if self.index is not None and not self.stop.is_set():
            try:
                report.indexed = await self.index.update(self.scan(), stop=self.stop)
            except Exception:
                log.exception("vault indexing failed")
        return report

    # ------------------------------------------------------------ 1. read: memories

    def _healthy(self, files: dict[str, FileInfo]) -> bool:
        """Is the vault really there? (Deletions are only trusted when it is.)

        Home and About me must be present and readable — wherever you've moved them.
        """
        found = {f.key: p for p, f in files.items() if f.key in ("home", "page:" + P.ABOUT)}
        for key, default in (("home", P.HOME), ("page:" + P.ABOUT, P.ABOUT)):
            p = found.get(key) or (default if default in files else None)
            if p is None or p in self._unreadable:
                return False
        return True

    def _memory_pages(self, files: dict[str, FileInfo], rows: dict) -> dict[str, MemPage]:
        """Every file that holds facts: About me, People/…, Projects/…, Topics/…
        (subfolders too), and Jarvis pages you moved elsewhere.

        A second file carrying the same page id while the original still exists is
        a copy (e.g. a OneDrive conflict copy "Priya-DESKTOP-1.md") and is ignored.
        """
        out: dict[str, MemPage] = {}
        for p, f in files.items():
            if not p.endswith(".md"):
                continue
            stem = p.rsplit("/", 1)[-1][:-3]
            folder = p.split("/", 1)[0]
            original = f.key[5:] if f.key and f.key.startswith("page:") else None
            if original is not None:
                owner = rows.get(f.key)
                home = owner["path"] if owner else original
                if home != p and home in files and files[home].key == f.key:
                    if p not in self._copies:
                        self._copies.add(p)
                        log.info("vault: %s looks like a copy of %s; ignoring it", p, home)
                    continue
            if p == P.ABOUT or original == P.ABOUT:
                out[p] = MemPage(f, None, None, "page:" + P.ABOUT)
            elif p.startswith(MEMORY_DIRS):
                cat = P.CATEGORY_FOR_FOLDER[folder]
                out[p] = MemPage(f, cat, stem, "page:" + P.memory_page_path(cat, stem))
            elif original and original.split("/", 1)[0] in P.CATEGORY_FOR_FOLDER:
                cat = P.CATEGORY_FOR_FOLDER[original.split("/", 1)[0]]
                out[p] = MemPage(f, cat, stem, "page:" + P.memory_page_path(cat, stem))
        return out

    async def _import_memories(self, files, rows, report: Report, handled, pending) -> None:
        pages = self._memory_pages(files, rows)
        tracked = {r["path"]: r for k, r in rows.items() if k.startswith("page:")}
        changed: list[MemPage] = []
        for p, mp in pages.items():
            r = tracked.get(p)
            region = parse_page(mp.file.text).region
            edited = (r is not None and r["hash"] != mp.file.hash) or (
                r is None and region is not None
            )
            if not edited:
                continue
            if self._busy(mp.file):
                pending.add(p)
                report.skipped_busy.append(p)
            elif region is not None:
                changed.append(mp)
            else:
                handled.add(p)  # markers deleted: nothing to read; they're put back

        # Every fact still in the vault. Pages we can't read reliably right now
        # (being edited, unreadable, markers deleted) count with what Jarvis wrote.
        present: set[int] = set()
        for p, mp in pages.items():
            region = parse_page(mp.file.text).region
            if region is not None:
                present |= {i.id for i in parse_items(region, loose=True) if i.kind == "m"}
            if (region is None or p in pending or p in self._unreadable) and p in tracked:
                present |= set(json.loads(tracked[p]["ids"] or "[]"))

        svc, store = self.memory, self.memory.store
        for mp in changed:
            region = parse_page(mp.file.text).region or ""
            strays: list[str] = []
            items = parse_items(region, loose=True, strays=strays)
            if strays:
                self._strays[mp.file.path] = strays  # kept just below Jarvis's part
            for item in items:
                if item.kind == "r":
                    continue
                if mp.category is None:  # About me: the section decides
                    cat = P.SECTION_CATEGORY.get(item.section.lower(), "other")
                    subject = None
                else:
                    cat, subject = mp.category, mp.subject
                if item.id is None:  # a line you added
                    mem, created = await svc.save(
                        item.text,
                        category=cat,
                        source="told",
                        subject=subject,
                        same_subject_only=True,
                    )
                    present.add(mem.id)
                    kind = "add" if created else "duplicate"
                    report.changes.append(Change(kind, mem.id, mem.text))
                    continue
                cur = store.get(item.id)
                if cur is None:
                    continue  # already gone from Jarvis; the page will be refreshed
                home = P.memory_page_path(cur.category, cur.subject).lower()
                here = mp.key[5:].lower()
                moved = home != here or (mp.category is None and cur.category != cat)
                if " ".join(item.text.split()) != " ".join(cur.text.split()):
                    mem = await svc.update(
                        cur.id, item.text, cat if moved else None, subject if moved else ...
                    )
                    report.changes.append(Change("update", cur.id, mem.text if mem else item.text))
                elif moved:
                    store.update(cur.id, category=cat, subject=subject)
                    report.imported.append(f"moved #{cur.id} to {mp.file.path}")
            handled.add(mp.file.path)
            ids = [i.id for i in parse_items(region, loose=True) if i.kind == "m"]
            # Remember this version as read, so it's never imported twice.
            self._save_row(mp.key, mp.file.path, mp.file.hash, ids, None, rows.get(mp.key))

        # Facts that were on a page Jarvis wrote and are now nowhere in the vault —
        # plus ones already waiting out their grace period (they're kept off the pages
        # meanwhile, so the page doesn't bring them back).
        waiting = {int(k[1:]) for k in self._missing_since if k.startswith("m")}
        # Only lines deleted from a page that's still there (readable, not mid-edit)
        # count. A page that's missing altogether — moved, OneDrive hiccup, not
        # downloaded yet — never makes Jarvis forget anything.
        if not self._healthy(files):
            return
        candidates = {
            mid
            for p, r in tracked.items()
            if r["hash"] != DELETED
            and p in pages
            and p not in pending
            and p not in self._unreadable
            and parse_page(pages[p].file.text).region is not None
            for mid in json.loads(r["ids"] or "[]")
        } | waiting
        gone = sorted(
            mid for mid in candidates if mid not in present and store.get(mid) is not None
        )
        for mid in waiting - set(gone):
            del self._missing_since[f"m{mid}"]  # you put it back (or it's gone already)
        if not gone:
            return
        total = len({mid for r in tracked.values() for mid in json.loads(r["ids"] or "[]")})
        if len(gone) >= 10 and len(gone) > total / 2:
            # That looks like the vault moved or failed to sync, not like you deleting
            # facts one by one. Forget nothing.
            log.warning("vault: %d facts missing at once — not forgetting them", len(gone))
            report.imported.append(f"⚠ {len(gone)} facts missing from the vault — kept them")
            for mid in gone:
                self._missing_since.pop(f"m{mid}", None)  # and show them again
            return
        for mid in gone:
            if self._gone_long_enough(f"m{mid}") and (cur := store.get(mid)) is not None:
                svc.forget(mid)
                self._missing_since.pop(f"m{mid}", None)
                report.changes.append(Change("delete", mid, cur.text))

    # ------------------------------------------------------------ 1. read: notes

    def _import_notes(self, files, rows, report: Report, handled, pending) -> None:
        by_key = {f.key: f for f in files.values() if f.key}
        healthy = self._healthy(files)
        for key, r in rows.items():
            if not key.startswith("note:") or r["hash"] == DELETED:
                continue
            nid = int(key.split(":", 1)[1])
            f = by_key.get(key)
            if f is None and r["path"] in files and files[r["path"]].key is None:
                f = files[r["path"]]  # its properties were edited away; still this note
            with self.db.lock:
                note = self.db.conn.execute("SELECT * FROM notes WHERE id = ?", (nid,)).fetchone()
            if note is None:
                continue
            if f is None:  # gone from the vault
                if note["secret"] or r["path"] in files or r["path"] in self._unreadable:
                    continue  # a secret's placeholder is simply written again
                if not healthy:
                    continue  # the vault looks unavailable, not edited
                if not self._gone_long_enough(key, PAGE_GONE_S):
                    continue  # a deleted note file counts after a minute, not a blip
                with self.db.lock, self.db.conn:
                    self.db.conn.execute("DELETE FROM notes WHERE id = ?", (nid,))
                self._save_row(key, r["path"], DELETED)
                self._missing_since.pop(key, None)
                report.imported.append(f"deleted note #{nid}")
                continue
            self._missing_since.pop(key, None)
            if f.hash == r["hash"]:
                continue
            if self._busy(f):
                pending.add(f.path)
                continue
            handled.add(f.path)
            if note["secret"]:
                continue  # its placeholder is restored; the secret stays in Jarvis
            region = parse_page(f.text).region
            text = (region or "").strip()
            if text and text != note["text"].strip():
                with self.db.lock, self.db.conn:
                    self.db.conn.execute(
                        "UPDATE notes SET text = ?, updated_at = ? WHERE id = ?",
                        (text, _now_iso(), nid),
                    )
                report.imported.append(f"updated note #{nid}")
            self._save_row(key, f.path, f.hash, json.loads(r["ids"] or "[]"), r["version"], r)

    # ------------------------------------------------------------ 1. read: reminders

    def _import_reminders(self, files, rows, report: Report, handled, pending) -> None:
        r = rows.get("reminders")
        f = files.get(r["path"] if r else P.REMINDERS)
        if f is None or (r and f.hash == r["hash"]):
            return
        if self._busy(f):
            pending.add(f.path)
            return
        region = parse_page(f.text).region
        if region is None:
            return
        handled.add(f.path)
        strays: list[str] = []
        items = parse_items(region, strays=strays)
        if any(s.strip() for s in strays):
            self._strays[f.path] = [s for s in strays if s.strip()]
        seen: set[int] = set()
        for item in items:
            text, due = split_due(item.text)
            if item.kind == "r" and item.id:
                seen.add(item.id)
                with self.db.lock:
                    row = self.db.conn.execute(
                        "SELECT * FROM reminders WHERE id = ?", (item.id,)
                    ).fetchone()
                if row is None:
                    continue
                sets, params = [], []
                if item.checked and row["status"] == "pending":
                    sets.append("status = 'done'")
                    report.imported.append(f"reminder #{item.id} done")
                elif item.checked is False and row["status"] in ("fired", "done"):
                    if due and due > datetime.now(UTC):
                        sets.append("status = 'pending', fired_at = NULL")
                        report.imported.append(f"reminder #{item.id} back on")
                if text and text != row["text"]:
                    sets.append("text = ?")
                    params.append(text)
                if due and _iso(due) != row["due_at"] and row["status"] == "pending":
                    sets.append("due_at = ?")
                    params.append(_iso(due))
                if sets:
                    with self.db.lock, self.db.conn:
                        self.db.conn.execute(
                            f"UPDATE reminders SET {', '.join(sets)} WHERE id = ?",
                            (*params, item.id),
                        )
                    if not any(s.startswith("status") for s in sets):
                        report.imported.append(f"reminder #{item.id} changed")
            elif item.kind is None and text:
                when = due or _tomorrow_9am()
                status = "done" if item.checked else "pending"
                with self.db.lock, self.db.conn:
                    self.db.conn.execute(
                        "INSERT INTO reminders (text, due_at, created_at, status) "
                        "VALUES (?, ?, ?, ?)",
                        (text, _iso(when), _now_iso(), status),
                    )
                report.imported.append(f"new reminder: {text} ({when:%a %d %b %H:%M})")
        for rid in json.loads(r["ids"] if r else "[]"):
            if rid not in seen:
                with self.db.lock, self.db.conn:
                    cur = self.db.conn.execute(
                        "UPDATE reminders SET status = 'cancelled' WHERE id = ? "
                        "AND status = 'pending'",
                        (rid,),
                    )
                if cur.rowcount:
                    report.imported.append(f"cancelled reminder #{rid}")
        # Read: never import this version again, even if writing the page fails.
        self._save_row("reminders", f.path, f.hash, sorted(seen), None, r)

    # ------------------------------------------------------------ 1. read: deleted pages

    def _mark_deleted_pages(self, files, rows) -> None:
        """A conversation/journal page you deleted stays deleted (it isn't re-created)."""
        keys = {f.key for f in files.values() if f.key}
        healthy = self._healthy(files)
        for key, r in rows.items():
            if not key.startswith(("conv:", "day:")) or r["hash"] == DELETED:
                continue
            if key in keys or r["path"] in files:
                self._missing_since.pop(key, None)
            elif healthy and self._gone_long_enough(key, PAGE_GONE_S):
                self._save_row(key, r["path"], DELETED)
                self._missing_since.pop(key, None)

    # ------------------------------------------------------------ 2. inbox

    async def _process_inbox(self, files, report: Report) -> None:
        for p, f in sorted(files.items()):
            if not p.startswith("Inbox/") or p in self._inbox_skipped or self.stop.is_set():
                continue
            if self._busy(f) or not f.text.strip():
                continue
            if f.size > MAX_INBOX_BYTES:
                self._inbox_skipped.add(p)
                report.errors.append(f"{p} is too big to learn from (over 300 KB) — left there")
                continue
            name, ext = p.split("/", 1)[1].rsplit(".", 1)
            dest = self._unique(f"Notes/{safe_name(name)}.{ext}")
            # Move first (your bytes, untouched): if the file is locked, nothing happens
            # and it's tried again next cycle — no duplicates.
            try:
                self._abs(dest).parent.mkdir(parents=True, exist_ok=True)
                os.replace(self._abs(p), self._abs(dest))
            except OSError as e:
                log.info("inbox: can't move %s yet: %s", p, e)
                continue
            self._cache.pop(p, None)
            report.filed.append(f"{p} → {dest}")
            body = parse_page(f.text).before if f.text.startswith("---") else f.text
            for chunk in _chunks(body.strip() or f.text, 1500)[:MAX_INBOX_CHUNKS]:
                try:
                    report.changes += await self.extractor.extract(chunk, force=True)
                except Exception as e:
                    log.info("inbox: couldn't learn from %s: %s", p, e)
                    report.errors.append(f"couldn't learn from {p}: {e}")
                    break

    def _unique(self, rel: str, taken: set[str] | None = None) -> str:
        taken = taken or set()

        def free(candidate: str) -> bool:
            return not self._abs(candidate).exists() and candidate.lower() not in taken

        if free(rel):
            return rel
        stem, ext = rel.rsplit(".", 1)
        n = 2
        while not free(f"{stem} ({n}).{ext}"):
            n += 1
        return f"{stem} ({n}).{ext}"

    # ------------------------------------------------------------ 3. write

    def desired_pages(self) -> list[P.Page]:
        store = self.memory.store
        waiting = {k for k in self._missing_since if k.startswith("m")}
        memories = [m for m in store.list(limit=100_000) if f"m{m.id}" not in waiting]
        subjects = store.subjects()
        conv_pages, mentions = (
            P.conversation_pages(
                self.db, subjects, user_name=self.user_name, cache=self._conv_cache
            )
            if self.conversations
            else ([], {})
        )
        mem_pages = P.memory_pages(memories, mentions)
        note_pgs = P.note_pages(self.db)
        reminders = P.reminders_page(self.db)
        pages = [*mem_pages, *note_pgs, reminders, *conv_pages]
        if self.journal:
            pages += P.journal_pages(self.db, memories, conv_pages)
        pages.append(P.home_page(mem_pages, note_pgs, conv_pages, len(reminders.ids)))
        return pages

    def _place(self, page: P.Page, r, files, by_key, lower, claimed) -> str:
        """Where a page lives: where you put it, else its own path — never on top of
        a file that isn't this page."""
        if page.kind in MEMORY_KINDS:
            existing = lower.get(page.path.lower())  # the same page, any letter case
            if existing:
                return existing
            moved = by_key.get(page.key)
            return moved.path if moved else page.path
        moved = by_key.get(page.key)
        if moved:
            return moved.path
        if r and r["path"] in files and files[r["path"]].key in (None, page.key):
            return r["path"]
        if r and r["path"] not in files and r["path"].lower() not in lower:
            return r["path"]
        mine = {r["path"].lower()} if r else set()
        return self._unique(page.path, (claimed | set(lower)) - mine)

    def _export(self, files, rows, report: Report, handled, pending) -> None:
        pages = self.desired_pages()
        wanted = {p.key for p in pages}
        by_key = {f.key: f for f in files.values() if f.key}
        lower = {p.lower(): p for p in files}
        claimed = {r["path"].lower() for r in rows.values() if r["hash"] != DELETED and r["path"]}
        for page in pages:
            if self.stop.is_set():
                return
            r = rows.get(page.key)
            if (r and r["hash"] == DELETED) or page.key in self._missing_since:
                continue  # deleted by you, or maybe deleted (waiting out the grace period)
            try:
                path = self._place(page, r, files, by_key, lower, claimed)
                f = files.get(path)
                if f is None and r is not None and r["path"] not in files:
                    # The page went missing. Wait a while before writing it again, in
                    # case it's a sync hiccup (nothing is forgotten meanwhile).
                    if not self._gone_long_enough("page-gone:" + page.key, PAGE_GONE_S):
                        continue
                self._missing_since.pop("page-gone:" + page.key, None)
                if path in pending or self._busy(f):
                    report.skipped_busy.append(path)
                    continue
                if (
                    page.kind in TWO_WAY
                    and f is not None
                    and r is not None
                    and r["path"] == path
                    and f.hash != r["hash"]
                    and path not in handled
                ):
                    continue  # an edit of yours Jarvis hasn't read yet: never overwrite it
                if (
                    r
                    and f
                    and r["version"]
                    and r["version"] == page.version
                    and f.hash == r["hash"]
                    and r["path"] == path
                ):
                    continue  # nothing new
                if f is None and self._abs(path).exists():
                    continue  # appeared since the scan; next cycle
                text = build_page(
                    f.text if f else None,
                    title=page.title,
                    region=page.region,
                    props=page.props,
                    user_section=page.user_section,
                    append_after="\n".join(self._strays.pop(path, [])),
                )
                if f is None or text != f.text:
                    self._write(path, text)
                    report.written.append(path)
                    lower[path.lower()] = path
                    claimed.add(path.lower())
                self._save_row(page.key, path, sha(text), page.ids, page.version, r)
            except Exception as e:
                log.exception("vault: couldn't write %s", page.path)
                report.errors.append(f"couldn't write {page.path}: {e}")

        # Pages Jarvis made that have nothing left to show (e.g. all facts forgotten)
        for key, r in rows.items():
            if (
                key in wanted
                or key.startswith("meta:")
                or r["hash"] == DELETED
                or key in ("home", "reminders")
            ):
                continue
            f = files.get(r["path"])
            if f is None:
                self._drop_row(key)
                continue
            if self._busy(f) or r["path"] in pending:
                continue
            try:
                parsed = parse_page(f.text)
                if f.key not in (None, key) and f.key in wanted:
                    self._drop_row(key)  # the file now belongs to another page
                elif not has_user_content(f.text) and f.hash == r["hash"]:
                    self._remove(r["path"])
                    report.deleted.append(r["path"])
                    self._drop_row(key)
                elif key.startswith("page:") and parsed.region and parsed.region.strip():
                    # Keep your notes on the page; just empty Jarvis's part.
                    title = r["path"].rsplit("/", 1)[-1][:-3]
                    text = build_page(f.text, title=title, region="", props={})
                    self._write(r["path"], text)
                    self._save_row(key, r["path"], sha(text), [], None, r)
                elif not key.startswith("page:"):
                    self._drop_row(key)  # yours now; Jarvis lets go of it
            except Exception as e:
                report.errors.append(f"couldn't tidy {r['path']}: {e}")


# ---------------------------------------------------------------- helpers

_DUE = re.compile(
    r"\s*📅\s*(?P<day>\d{4}-\d{2}-\d{2})"
    r"(?:[ T]+(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ampm>[ap]\.?m\.?)?)?\s*$",
    re.I,
)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def _tomorrow_9am() -> datetime:
    now = datetime.now().astimezone()
    return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)


def split_due(text: str) -> tuple[str, datetime | None]:
    """ "Call the bank 📅 2026-10-05 09:00" (or 9am / 6:30 pm) → (text, local time)."""
    m = _DUE.search(text)
    if not m:
        return text.strip(), None
    hour, minute = 9, 0
    if m.group("h"):
        hour, minute = int(m.group("h")), int(m.group("m") or 0)
        ampm = (m.group("ampm") or "").lower().replace(".", "")
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
    try:
        day = datetime.strptime(m.group("day"), "%Y-%m-%d")
        local = day.replace(hour=hour, minute=minute).astimezone()
    except ValueError:
        return text.strip(), None
    return text[: m.start()].strip(), local


def _chunks(text: str, size: int) -> list[str]:
    paras, out, cur = text.split("\n\n"), [], ""
    for p in paras:
        if len(cur) + len(p) > size and cur:
            out.append(cur)
            cur = ""
        cur += p + "\n\n"
    if cur.strip():
        out.append(cur)
    return out


# ---------------------------------------------------------------- background thread


class VaultWorker:
    """Runs sync cycles every few seconds on its own thread and event loop."""

    def __init__(
        self,
        make_sync: Callable[[], VaultSync],
        *,
        interval_s: float = 3.0,
        on_report: Callable[[Report], None] | None = None,
        prepare: Callable[[VaultSync], object] | None = None,
    ):
        self._make = make_sync
        self.interval_s = interval_s
        self.on_report = on_report
        self.prepare = prepare  # e.g. sort old memories into pages, once
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None
        self.last_sync: float | None = None
        self.sync: VaultSync | None = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            self.sync = self._make()
            self.sync.stop = self._stop
            if self.prepare:
                try:
                    loop.run_until_complete(self.prepare(self.sync))
                except Exception:
                    log.exception("vault preparation failed")
            while not self._stop.is_set():
                try:
                    report = loop.run_until_complete(self.sync.cycle())
                    self.last_sync = time.time()
                    self.last_error = report.errors[-1] if report.errors else None
                    if self.on_report and (
                        report.changes or report.filed or report.imported or report.errors
                    ):
                        self.on_report(report)
                except Exception as e:
                    self.last_error = str(e)
                    log.exception("vault sync failed")
                self._stop.wait(self.interval_s)
        finally:
            embedder = getattr(getattr(self.sync, "memory", None), "embedder", None)
            if embedder is not None:
                try:
                    loop.run_until_complete(embedder.models.aclose())
                except Exception:
                    pass
            loop.close()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="vault", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 30) -> bool:
        """Ask the thread to finish its cycle and wait. False if it's still running."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            alive = self._thread.is_alive()
            self._thread = None
            return not alive
        return True
