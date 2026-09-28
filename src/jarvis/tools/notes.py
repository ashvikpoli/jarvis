"""Notes and reminders — Jarvis's own data, stored in SQLite."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field, model_validator

from jarvis.storage import Database
from jarvis.tools.base import Notifier, Risk, Tool, ToolContext, ToolError

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    """Canonical UTC timestamp, so string comparison in SQL orders correctly."""
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def _fmt_local(dt_utc: datetime) -> str:
    return dt_utc.astimezone().strftime("%a %d %b %Y, %H:%M")


# ------------------------------------------------------------------ notes


class NoteAddArgs(BaseModel):
    text: str = Field(description="The note, in the user's words.")


class NotesAdd(Tool):
    name = "notes_add"
    description = "Save a note for the user to recall later."
    risk = Risk.CHANGE
    Args = NoteAddArgs

    def describe(self, args: NoteAddArgs) -> str:
        return f"save a note: “{args.text[:80]}”"

    async def run(self, args: NoteAddArgs, ctx: ToolContext):
        with ctx.db.lock, ctx.db.conn:
            cur = ctx.db.conn.execute(
                "INSERT INTO notes (text, created_at) VALUES (?, ?)",
                (args.text, _iso(_utcnow())),
            )
        return f"Saved note #{cur.lastrowid}."


class NoteSearchArgs(BaseModel):
    query: str = Field("", description="Words to look for. Empty = most recent notes.")
    limit: int = Field(20, ge=1, le=100)


def _note_rows(ctx: ToolContext, query: str, limit: int) -> list[dict]:
    sql, params = "SELECT id, text, created_at FROM notes", []
    words = [w for w in query.split() if w]
    if words:
        sql += " WHERE " + " AND ".join("text LIKE ?" for _ in words)
        params = [f"%{w}%" for w in words]
    sql += " ORDER BY id DESC LIMIT ?"
    with ctx.db.lock:
        rows = ctx.db.conn.execute(sql, (*params, limit)).fetchall()
    return [
        {
            "id": r["id"],
            "text": r["text"],
            "saved": _fmt_local(datetime.fromisoformat(r["created_at"])),
        }
        for r in rows
    ]


class NotesSearch(Tool):
    name = "notes_search"
    description = "Search or list the user's saved notes."
    risk = Risk.SAFE
    Args = NoteSearchArgs

    async def run(self, args: NoteSearchArgs, ctx: ToolContext):
        rows = _note_rows(ctx, args.query, args.limit)
        return {"notes": rows} if rows else "No matching notes."


class IdArgs(BaseModel):
    id: int = Field(description="The id number.")


class NotesDelete(Tool):
    name = "notes_delete"
    description = "Delete a saved note by id."
    risk = Risk.CHANGE
    Args = IdArgs

    def describe(self, args: IdArgs) -> str:
        return f"delete note #{args.id}"

    async def run(self, args: IdArgs, ctx: ToolContext):
        with ctx.db.lock, ctx.db.conn:
            cur = ctx.db.conn.execute("DELETE FROM notes WHERE id = ?", (args.id,))
        if not cur.rowcount:
            raise ToolError(f"There's no note #{args.id}.")
        return f"Deleted note #{args.id}."


# ------------------------------------------------------------------ reminders


class When(BaseModel):
    at: str | None = Field(
        None, description="Local date/time, ISO format, e.g. '2026-10-01T09:00'."
    )
    in_minutes: float | None = Field(
        None, description="Or: how many minutes from now, e.g. 10. Use one of at / in_minutes."
    )

    def resolve(self, now: datetime) -> datetime | None:
        """Return the due time in UTC, or None if neither field was given."""
        if self.in_minutes is not None:
            if self.in_minutes <= 0:
                raise ToolError("in_minutes must be positive.")
            return (now + timedelta(minutes=self.in_minutes)).astimezone(UTC)
        if self.at:
            try:
                dt = datetime.fromisoformat(self.at.strip().replace("Z", "+00:00"))
            except ValueError as e:
                raise ToolError(
                    f"Couldn't read the time {self.at!r}; use ISO like 2026-10-01T09:00."
                ) from e
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=now.tzinfo)  # the user's local time
            return dt.astimezone(UTC)
        return None


class ReminderSetArgs(When):
    text: str = Field(description="What to remind the user about.")

    @model_validator(mode="after")
    def _one_time(self):
        if (self.at is None) == (self.in_minutes is None):
            raise ValueError("give exactly one of 'at' or 'in_minutes'")
        return self


class RemindersSet(Tool):
    name = "reminders_set"
    description = (
        "Set a reminder; the user gets a Windows notification when it's due. "
        "To change an existing reminder use reminders_update instead."
    )
    risk = Risk.CHANGE
    Args = ReminderSetArgs

    def describe(self, args: ReminderSetArgs) -> str:
        when = f"in {args.in_minutes:g} min" if args.in_minutes is not None else f"at {args.at}"
        return f"set a reminder {when}: “{args.text[:60]}”"

    async def run(self, args: ReminderSetArgs, ctx: ToolContext):
        now = ctx.clock()
        due = args.resolve(now)
        if due < now.astimezone(UTC) - timedelta(minutes=1):
            raise ToolError(f"{_fmt_local(due)} is in the past.")
        with ctx.db.lock, ctx.db.conn:
            cur = ctx.db.conn.execute(
                "INSERT INTO reminders (text, due_at, created_at) VALUES (?, ?, ?)",
                (args.text, _iso(due), _iso(_utcnow())),
            )
        return f"Reminder #{cur.lastrowid} set for {_fmt_local(due)}: {args.text}"


class ReminderListArgs(BaseModel):
    include_past: bool = Field(False, description="Also show fired/cancelled reminders.")


def reminder_rows(db: Database, include_past: bool = False, limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM reminders"
    if not include_past:
        sql += " WHERE status = 'pending'"
    sql += " ORDER BY due_at LIMIT ?"
    with db.lock:
        rows = db.conn.execute(sql, (limit,)).fetchall()
    return [
        {
            "id": r["id"],
            "text": r["text"],
            "due": _fmt_local(datetime.fromisoformat(r["due_at"])),
            "status": r["status"],
        }
        for r in rows
    ]


class RemindersList(Tool):
    name = "reminders_list"
    description = "List the user's reminders (upcoming by default)."
    risk = Risk.SAFE
    Args = ReminderListArgs

    async def run(self, args: ReminderListArgs, ctx: ToolContext):
        rows = reminder_rows(ctx.db, args.include_past)
        return {"reminders": rows} if rows else "No reminders."


class ReminderUpdateArgs(When):
    id: int = Field(description="Reminder id (see reminders_list).")
    text: str | None = Field(None, description="New text, if changing it.")


class RemindersUpdate(Tool):
    name = "reminders_update"
    description = "Change an existing reminder's time and/or text (e.g. when the user corrects it)."
    risk = Risk.CHANGE
    Args = ReminderUpdateArgs

    def describe(self, args: ReminderUpdateArgs) -> str:
        return f"update reminder #{args.id}"

    async def run(self, args: ReminderUpdateArgs, ctx: ToolContext):
        with ctx.db.lock:
            row = ctx.db.conn.execute("SELECT * FROM reminders WHERE id = ?", (args.id,)).fetchone()
        if row is None:
            raise ToolError(f"There's no reminder #{args.id}.")
        due = args.resolve(ctx.clock()) or datetime.fromisoformat(row["due_at"])
        text = args.text or row["text"]
        with ctx.db.lock, ctx.db.conn:
            ctx.db.conn.execute(
                "UPDATE reminders SET text = ?, due_at = ?, status = 'pending', fired_at = NULL "
                "WHERE id = ?",
                (text, _iso(due), args.id),
            )
        return f"Reminder #{args.id} is now {_fmt_local(due)}: {text}"


class RemindersCancel(Tool):
    name = "reminders_cancel"
    description = "Cancel a reminder by id."
    risk = Risk.CHANGE
    Args = IdArgs

    def describe(self, args: IdArgs) -> str:
        return f"cancel reminder #{args.id}"

    async def run(self, args: IdArgs, ctx: ToolContext):
        with ctx.db.lock, ctx.db.conn:
            cur = ctx.db.conn.execute(
                "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
                (args.id,),
            )
        if not cur.rowcount:
            raise ToolError(f"There's no pending reminder #{args.id}.")
        return f"Cancelled reminder #{args.id}."


NOTE_TOOLS = [
    NotesAdd,
    NotesSearch,
    NotesDelete,
    RemindersSet,
    RemindersList,
    RemindersUpdate,
    RemindersCancel,
]


# ------------------------------------------------------------------ scheduler


class ReminderScheduler:
    """Background thread that fires due reminders while Jarvis is running.

    A thread (not an asyncio task) because the CLI's event loop sleeps while it
    waits for you to type.
    """

    def __init__(self, db: Database, notifier: Notifier, interval_s: float = 10.0):
        self.db = db
        self.notifier = notifier
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def check_once(self, now: datetime | None = None) -> list[dict]:
        now = now or _utcnow()
        with self.db.lock, self.db.conn:
            rows = self.db.conn.execute(
                "SELECT * FROM reminders WHERE status = 'pending' AND due_at <= ? ORDER BY due_at",
                (_iso(now),),
            ).fetchall()
            for r in rows:
                self.db.conn.execute(
                    "UPDATE reminders SET status = 'fired', fired_at = ? WHERE id = ?",
                    (_iso(now), r["id"]),
                )
        fired = [dict(r) for r in rows]
        for r in fired:
            try:
                self.notifier.notify("Jarvis reminder", r["text"])
            except Exception:
                log.exception("notification failed for reminder %s", r["id"])
        return fired

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.check_once()
            except Exception:
                log.exception("reminder check failed")

    def start(self) -> None:
        if self._thread is None:
            self.check_once()  # anything that came due while Jarvis was closed
            self._thread = threading.Thread(target=self._loop, name="reminders", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
