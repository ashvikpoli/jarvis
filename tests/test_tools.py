import os
import sys
from datetime import UTC, datetime, timedelta

import pytest

from jarvis.storage import Database
from jarvis.tools import ReminderScheduler, ToolContext, ToolError, reminder_rows, resolve_root
from jarvis.tools.base import NoArgs
from jarvis.tools.files import (
    DeleteArgs,
    FilesDelete,
    FilesList,
    FilesMkdir,
    FilesMove,
    FilesRead,
    FilesSearch,
    FilesWrite,
    ListArgs,
    MkdirArgs,
    MoveArgs,
    ReadArgs,
    SearchArgs,
    WriteArgs,
    safe_path,
)
from jarvis.tools.notes import (
    IdArgs,
    NoteAddArgs,
    NotesAdd,
    NotesDelete,
    NoteSearchArgs,
    NotesSearch,
    ReminderListArgs,
    RemindersCancel,
    ReminderSetArgs,
    RemindersList,
    RemindersSet,
    RemindersUpdate,
    ReminderUpdateArgs,
)
from jarvis.tools.system import (
    AppArgs,
    AppsOpen,
    SystemInfo,
    SystemTime,
    UrlArgs,
    WebOpen,
    WebSearch,
)
from jarvis.tools.system import (
    SearchArgs as WebSearchArgs,
)
from tests.conftest import FakeLauncher, FakeNotifier

NOW = datetime(2026, 9, 28, 10, 0, tzinfo=UTC).astimezone()


@pytest.fixture
def ctx(roots):
    return ToolContext(
        db=Database(":memory:"),
        file_roots=roots,
        launcher=FakeLauncher(),
        notifier=FakeNotifier(),
        clock=lambda: NOW,
    )


# ------------------------------------------------------------------ sandbox


def test_safe_path_forms(ctx, roots):
    docs, desk = roots
    assert safe_path("", ctx) == docs
    assert safe_path("report.txt", ctx) == docs / "report.txt"
    assert safe_path("Desktop/x.txt", ctx) == desk / "x.txt"
    assert safe_path("desktop", ctx) == desk
    assert safe_path(str(docs / "projects"), ctx) == docs / "projects"


WIN_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="Windows path")


@pytest.mark.parametrize(
    "bad",
    [
        "../outside.txt",
        "Documents/../../outside.txt",
        "/etc/passwd" if sys.platform != "win32" else "C:\\Windows\\win.ini",
        pytest.param("C:\\Windows\\System32", marks=WIN_ONLY),
        pytest.param("%WINDIR%\\System32", marks=WIN_ONLY),
    ],
)
def test_safe_path_blocks_escapes(ctx, bad):
    with pytest.raises(ToolError, match="outside"):
        safe_path(bad, ctx)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin on Windows")
def test_safe_path_blocks_symlink_escape(ctx, roots, tmp_path):
    os.symlink(tmp_path / "outside.txt", roots[0] / "link.txt")
    with pytest.raises(ToolError, match="outside"):
        safe_path("link.txt", ctx)


def test_resolve_root_known_and_explicit(tmp_path):
    assert resolve_root("Documents").name == "Documents"
    assert resolve_root(str(tmp_path)) == tmp_path.resolve()


# ------------------------------------------------------------------ files


async def test_list_and_search(ctx):
    out = await FilesList().run(ListArgs(), ctx)
    names = {e["name"] for e in out["entries"]}
    assert names == {"report.txt", "budget-2026.csv", "projects"}
    csv = await FilesList().run(ListArgs(pattern="*.csv"), ctx)
    assert [e["name"] for e in csv["entries"]] == ["budget-2026.csv"]

    hits = await FilesSearch().run(SearchArgs(query="JARVIS"), ctx)
    assert [m["name"] for m in hits["matches"]] == ["jarvis-notes.md"]
    globbed = await FilesSearch().run(SearchArgs(query="*.md"), ctx)
    assert len(globbed["matches"]) == 1


async def test_read_text_and_reject_binary(ctx, roots):
    out = await FilesRead().run(ReadArgs(path="report.txt"), ctx)
    assert "Revenue up 12%" in out["content"] and not out["truncated"]
    (roots[0] / "img.bin").write_bytes(b"\x89PNG\x00\x00\x01")
    with pytest.raises(ToolError, match="binary"):
        await FilesRead().run(ReadArgs(path="img.bin"), ctx)
    with pytest.raises(ToolError, match="doesn't exist"):
        await FilesRead().run(ReadArgs(path="nope.txt"), ctx)


async def test_write_never_overwrites_by_default(ctx, roots):
    w = FilesWrite()
    await w.run(WriteArgs(path="Desktop/list.txt", content="milk"), ctx)
    assert (roots[1] / "list.txt").read_text() == "milk"
    with pytest.raises(ToolError, match="already exists"):
        await w.run(WriteArgs(path="Desktop/list.txt", content="eggs"), ctx)
    await w.run(WriteArgs(path="Desktop/list.txt", content="eggs", overwrite=True), ctx)
    assert (roots[1] / "list.txt").read_text() == "eggs"
    await w.run(WriteArgs(path="Desktop/sub/new.txt", content="x"), ctx)  # creates parents
    assert (roots[1] / "sub" / "new.txt").exists()


async def test_mkdir_move_delete(ctx, roots, monkeypatch):
    docs, desk = roots
    await FilesMkdir().run(MkdirArgs(path="Documents/archive"), ctx)
    assert (docs / "archive").is_dir()
    await FilesMove().run(MoveArgs(source="report.txt", destination="Documents/archive"), ctx)
    assert (docs / "archive" / "report.txt").exists() and not (docs / "report.txt").exists()
    await FilesMove().run(MoveArgs(source="budget-2026.csv", destination="Desktop/b.csv"), ctx)
    assert (desk / "b.csv").exists()
    (desk / "c.csv").write_text("x")
    with pytest.raises(ToolError, match="already exists"):
        await FilesMove().run(MoveArgs(source="Desktop/c.csv", destination="Desktop/b.csv"), ctx)

    trashed = []
    import send2trash

    monkeypatch.setattr(send2trash, "send2trash", lambda p: trashed.append(p))
    await FilesDelete().run(DeleteArgs(path="Desktop/c.csv"), ctx)
    assert trashed == [str(desk / "c.csv")]
    with pytest.raises(ToolError, match="top-level"):
        await FilesDelete().run(DeleteArgs(path="Desktop"), ctx)
    with pytest.raises(ToolError, match="top-level"):
        await FilesMove().run(MoveArgs(source="Desktop", destination="Documents/old-desktop"), ctx)


# ------------------------------------------------------------------ apps / web / system


async def test_apps_and_web_use_launcher(ctx, monkeypatch):
    await AppsOpen().run(AppArgs(name="Notepad"), ctx)
    assert ctx.launcher.opened == ["notepad.exe"]

    from jarvis.tools import system

    monkeypatch.setattr(
        system,
        "start_menu_shortcuts",
        lambda: {
            "spotify": system.Path("C:/x/Spotify.lnk"),
            "visual studio code": system.Path("C:/x/Visual Studio Code.lnk"),
        },
    )
    await AppsOpen().run(AppArgs(name="spotfy"), ctx)  # typo → fuzzy match
    await AppsOpen().run(AppArgs(name="studio"), ctx)  # substring
    assert ctx.launcher.opened[1].endswith("Spotify.lnk")
    assert ctx.launcher.opened[2].endswith("Visual Studio Code.lnk")
    with pytest.raises(ToolError, match="couldn't find"):
        await AppsOpen().run(AppArgs(name="photoshop"), ctx)

    await WebOpen().run(UrlArgs(url="youtube.com"), ctx)
    assert ctx.launcher.opened[-1] == "https://youtube.com"
    with pytest.raises(ToolError):
        await WebOpen().run(UrlArgs(url="file:///C:/Windows/system32/cmd.exe"), ctx)
    await WebSearch().run(WebSearchArgs(query="best pizza near me"), ctx)
    assert ctx.launcher.opened[-1].endswith("q=best+pizza+near+me")


async def test_system_time_and_info(ctx):
    t = await SystemTime().run(NoArgs(), ctx)
    assert "2026" in t["local_time"]
    info = await SystemInfo().run(NoArgs(), ctx)
    assert info["ram_total_gb"] > 0 and 0 <= info["cpu_pct"] <= 100


# ------------------------------------------------------------------ notes


async def test_notes(ctx):
    await NotesAdd().run(NoteAddArgs(text="Guest wifi password is on the fridge"), ctx)
    await NotesAdd().run(NoteAddArgs(text="Buy printer ink"), ctx)
    found = await NotesSearch().run(NoteSearchArgs(query="wifi fridge"), ctx)
    assert [n["text"] for n in found["notes"]] == ["Guest wifi password is on the fridge"]
    assert len((await NotesSearch().run(NoteSearchArgs(), ctx))["notes"]) == 2
    await NotesDelete().run(IdArgs(id=2), ctx)
    with pytest.raises(ToolError):
        await NotesDelete().run(IdArgs(id=2), ctx)
    assert await NotesSearch().run(NoteSearchArgs(query="ink"), ctx) == "No matching notes."


# ------------------------------------------------------------------ reminders


def test_reminder_args_need_exactly_one_time():
    with pytest.raises(ValueError):
        ReminderSetArgs(text="x")
    with pytest.raises(ValueError):
        ReminderSetArgs(text="x", at="2026-10-01T09:00", in_minutes=5)


async def test_reminders_lifecycle_and_scheduler(ctx):
    out = await RemindersSet().run(ReminderSetArgs(text="stretch", in_minutes=10), ctx)
    assert out.startswith("Reminder #1")
    local_at = (NOW + timedelta(days=1)).replace(tzinfo=None).isoformat(timespec="minutes")
    await RemindersSet().run(ReminderSetArgs(text="dentist", at=local_at), ctx)
    with pytest.raises(ToolError, match="past"):
        await RemindersSet().run(ReminderSetArgs(text="x", at="2020-01-01T09:00"), ctx)

    listed = await RemindersList().run(ReminderListArgs(), ctx)
    assert [r["text"] for r in listed["reminders"]] == ["stretch", "dentist"]

    # the Tuesday → Thursday fix: update in place, no duplicate
    await RemindersUpdate().run(ReminderUpdateArgs(id=2, in_minutes=60 * 48), ctx)
    rows = reminder_rows(ctx.db)
    assert len(rows) == 2 and rows[1]["id"] == 2

    sched = ReminderScheduler(ctx.db, ctx.notifier)
    assert sched.check_once(NOW.astimezone(UTC) + timedelta(minutes=5)) == []
    fired = sched.check_once(NOW.astimezone(UTC) + timedelta(minutes=11))
    assert [r["text"] for r in fired] == ["stretch"]
    assert ctx.notifier.sent == [("Jarvis reminder", "stretch")]
    assert sched.check_once(NOW.astimezone(UTC) + timedelta(minutes=12)) == []  # fires once

    await RemindersCancel().run(IdArgs(id=2), ctx)
    assert await RemindersList().run(ReminderListArgs(), ctx) == "No reminders."
    everything = await RemindersList().run(ReminderListArgs(include_past=True), ctx)
    assert {r["status"] for r in everything["reminders"]} == {"fired", "cancelled"}
    with pytest.raises(ToolError):
        await RemindersCancel().run(IdArgs(id=2), ctx)


def test_scheduler_thread_starts_and_stops(ctx):
    sched = ReminderScheduler(ctx.db, ctx.notifier, interval_s=0.01)
    sched.start()
    sched.stop()
    assert not sched._thread.is_alive()
