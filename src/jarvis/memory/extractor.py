"""Learning: pull durable facts out of what the user says, in the background."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from jarvis.core.types import Message
from jarvis.memory.service import MemoryService
from jarvis.memory.store import CATEGORIES
from jarvis.providers import ModelManager, ProviderError

log = logging.getLogger(__name__)

EXTRACT_SYSTEM = """You maintain the long-term memory of a personal assistant called Jarvis.
Read the USER's latest message and decide what, if anything, is worth remembering about the user for future conversations.

Remember durable facts the user states about themselves or their life:
- profile: name, where they live, job/company, role, background
- preference: likes, dislikes, how they want Jarvis to behave
- person: people in their life and who they are (family, friends, colleagues)
- project: ongoing work, goals, plans with a lasting horizon
- other: other lasting facts (e.g. owns a dog named Bruno)

Do NOT remember:
- questions, requests or one-off tasks ("help me write an email")
- temporary states ("I'm tired today", "I'm busy right now")
- anything only the assistant said, or that the user merely agreed with
- secrets: passwords, PINs, codes, account numbers (those belong in notes, only if asked)
- guesses or inferences the user didn't state

Write each memory as one short standalone sentence starting with "User" (e.g. "User's sister is named Priya.").

Give each memory a "subject": the person, project or topic it is about, as a short proper name ("Priya", "Exinent sale", "Dividend investing"). Reuse a name from KNOWN SUBJECTS when it's the same one. Use null when the fact is about the user themselves (their name, home, job, likes).

Compare with EXISTING memories (listed with ids):
- if the message changes or corrects one (moved city, new job), use "update" with that id and the full new sentence
- if the user asks to forget something or says it's no longer true, use "delete" with that id
- if it's already known, do nothing
- otherwise "add"

Return JSON only. If there is nothing to remember, return {"operations": []}."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "update", "delete"]},
                    "id": {"type": ["integer", "null"]},
                    "text": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "subject": {"type": ["string", "null"]},
                },
                "required": ["op"],
            },
        }
    },
    "required": ["operations"],
}

_PERSONAL = re.compile(
    r"\b(i|i'm|im|i've|ive|i'd|i'll|my|me|mine|myself|we|we're|our|ours|us|remember|forget)\b",
    re.IGNORECASE,
)
_SECRETISH = re.compile(r"\b(password|passcode|pin|otp|cvv|ssn|account number|api key)\b", re.I)


def worth_checking(text: str) -> bool:
    """Cheap filter: skip the LLM call when the message can't contain a personal fact."""
    return len(text.split()) >= 3 and bool(_PERSONAL.search(text))


@dataclass
class Change:
    op: Literal["add", "update", "delete", "duplicate"]
    id: int
    text: str


class MemoryExtractor:
    def __init__(self, models: ModelManager, service: MemoryService, role: str = "memory"):
        self.models = models
        self.service = service
        self.role = role

    async def extract(
        self,
        user_text: str,
        context: Sequence[Message] = (),
        conversation_id: int | None = None,
        *,
        force: bool = False,
    ) -> list[Change]:
        if not self.service.enabled or (not force and not worth_checking(user_text)):
            return []
        related = await self.service.recall(user_text, limit=8, min_score=0.3)
        known = {s.memory.id: s.memory for s in related}
        for m in self.service.core():
            known.setdefault(m.id, m)
        existing = (
            "\n".join(
                f"#{i}: {m.text}" + (f"  [subject: {m.subject}]" if m.subject else "")
                for i, m in known.items()
            )
            or "(none)"
        )
        subjects = ", ".join(self.service.store.subjects()[:40]) or "(none yet)"
        recent = "\n".join(
            f"{m.role.capitalize()}: {m.content[:400]}"
            for m in list(context)[-2:]
            if m.role in ("user", "assistant") and m.content
        )
        prompt = (
            f"KNOWN SUBJECTS: {subjects}\n\n"
            f"EXISTING memories:\n{existing}\n\n"
            + (f"Recent conversation (context only):\n{recent}\n\n" if recent else "")
            + f"USER's latest message:\n{user_text}"
        )
        raw = await self._complete([Message.system(EXTRACT_SYSTEM), Message.user(prompt)])
        ops = _parse_ops(raw)
        changes: list[Change] = []
        for op in ops:
            kind, text, mid = op.get("op"), (op.get("text") or "").strip(), op.get("id")
            category = op.get("category") or "other"
            subject = op.get("subject") if isinstance(op.get("subject"), str) else None
            if subject and subject.islower():
                subject = subject[:1].upper() + subject[1:]  # "rahul" → "Rahul"
            if category in ("profile", "preference"):
                subject = None  # facts about the user themselves
            if kind == "add" and text:
                if _SECRETISH.search(text):
                    continue
                mem, created = await self.service.save(
                    text,
                    category=category,
                    source="learned",
                    conversation_id=conversation_id,
                    subject=subject,
                )
                changes.append(Change("add" if created else "duplicate", mem.id, mem.text))
            elif kind == "update" and text and mid in known:  # only ids we showed it
                if _SECRETISH.search(text):
                    continue
                mem = await self.service.update(
                    mid, text, category, subject if subject else known[mid].subject
                )
                if mem:
                    changes.append(Change("update", mem.id, mem.text))
            elif kind == "delete" and mid in known:
                if self.service.forget(mid):
                    changes.append(Change("delete", mid, known[mid].text))
        return changes

    async def _complete(self, messages: list[Message]) -> str:
        parts: list[str] = []
        opts = {"format": EXTRACT_SCHEMA, "temperature": 0, "think": False}
        async for chunk in self.models.stream_chat(messages, self.role, options=opts):
            parts.append(chunk.text)
        return "".join(parts)


SUBJECTS_SYSTEM = """You sort a personal assistant's memories about the user into pages.
For each memory, give its subject: the person, project or topic it is about, as a short proper name ("Priya", "Exinent sale", "Dividend investing"). Use the same name for memories about the same thing. Use null when it's about the user themselves (their name, home, job, likes).
Return JSON only: {"subjects": [{"id": 1, "subject": "Priya"}, ...]}"""

SUBJECTS_SCHEMA = {
    "type": "object",
    "properties": {
        "subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "subject": {"type": ["string", "null"]},
                },
                "required": ["id"],
            },
        }
    },
    "required": ["subjects"],
}


async def assign_subjects(
    models: ModelManager, store, role: str = "memory", batch: int = 25
) -> int:
    """Give older memories (from before Phase 4.5) a subject, so they get their own page."""
    todo = [
        m
        for m in store.list(limit=100_000)
        if m.subject is None and m.category in ("person", "project", "other")
    ]
    done = 0
    for i in range(0, len(todo), batch):
        part = todo[i : i + batch]
        known = ", ".join(store.subjects()[:40]) or "(none yet)"
        listing = "\n".join(f"#{m.id} [{m.category}] {m.text}" for m in part)
        msgs = [
            Message.system(SUBJECTS_SYSTEM),
            Message.user(f"KNOWN SUBJECTS: {known}\n\nMEMORIES:\n{listing}"),
        ]
        parts: list[str] = []
        opts = {"format": SUBJECTS_SCHEMA, "temperature": 0, "think": False}
        async for chunk in models.stream_chat(msgs, role, options=opts):
            parts.append(chunk.text)
        raw = "".join(parts)
        try:
            data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        except (json.JSONDecodeError, ValueError):
            continue
        ids = {m.id for m in part}
        for item in data.get("subjects", []) if isinstance(data, dict) else []:
            if isinstance(item, dict) and item.get("id") in ids and item.get("subject"):
                store.update(item["id"], subject=str(item["subject"]))
                done += 1
    return done


def _parse_ops(raw: str) -> list[dict]:
    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        log.info("extractor returned invalid JSON: %r", raw[:200])
        return []
    ops = data.get("operations") if isinstance(data, dict) else None
    return [o for o in ops if isinstance(o, dict)] if isinstance(ops, list) else []


class MemoryWorker:
    """Runs extraction on its own thread + event loop so replies never wait for it.

    Needs its own model clients (network clients belong to one event loop), so it's
    given a factory rather than an extractor.
    """

    def __init__(
        self,
        make_extractor: Callable[[], MemoryExtractor],
        on_changes: Callable[[list[Change]], None] | None = None,
    ):
        self._make = make_extractor
        self.on_changes = on_changes
        self._loop: asyncio.AbstractEventLoop | None = None
        self._extractor: MemoryExtractor | None = None
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: set[concurrent.futures.Future] = set()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._extractor = self._make()
        self._ready.set()
        loop.run_forever()
        try:
            loop.run_until_complete(self._extractor.models.aclose())
        finally:
            loop.close()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="memory", daemon=True)
            self._thread.start()
            self._ready.wait(5)

    def submit(
        self, user_text: str, context: Sequence[Message] = (), conversation_id: int | None = None
    ) -> concurrent.futures.Future | None:
        if not worth_checking(user_text):
            return None
        self.start()
        assert self._loop and self._extractor
        fut = asyncio.run_coroutine_threadsafe(
            self._job(user_text, list(context), conversation_id), self._loop
        )
        self._pending.add(fut)
        fut.add_done_callback(self._pending.discard)
        return fut

    async def _job(self, user_text, context, conversation_id) -> list[Change]:
        try:
            changes = await self._extractor.extract(user_text, context, conversation_id)
        except ProviderError as e:
            log.info("memory extraction skipped: %s", e)
            return []
        except Exception:
            log.exception("memory extraction failed")
            return []
        if changes and self.on_changes:
            try:
                self.on_changes(changes)
            except Exception:
                log.exception("on_changes callback failed")
        return changes

    def flush(self, timeout: float = 20) -> None:
        concurrent.futures.wait(list(self._pending), timeout=timeout)

    def close(self, timeout: float = 20) -> None:
        if self._thread is None or self._loop is None:
            return
        self.flush(timeout)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._thread = None
