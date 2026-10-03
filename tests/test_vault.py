"""Phase 4.5 — the Obsidian vault: pages from Jarvis's data, and your edits flowing back."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from jarvis.app import resolve_vault
from jarvis.memory.extractor import MemoryExtractor, assign_subjects
from jarvis.tools.notes import NoteAddArgs, NotesAdd, looks_secret
from jarvis.tools.vault import VaultSearch, VaultSearchArgs
from jarvis.vault.markdown import Frontmatter, build_page, parse_items, parse_page, safe_name
from jarvis.vault.sync import VaultWorker


@pytest.fixture
def vault(tmp_path, make_app):
    """(app, sync, root) with a few memories, notes, a reminder and a conversation."""

    async def build(turns=(), extra_config: str = ""):
        root = tmp_path / "vault"
        root.mkdir(exist_ok=True)
        cfg = f'[vault]\npath = "{root.as_posix()}"\nquiet_s = 0\nuser_name = "Adi"\n'
        app, provider = make_app(
            ["Nice to meet you. Priya sounds lovely.", *turns], config=cfg + extra_config
        )
        m = app.memory
        await m.save("User's name is Adi.", category="profile")
        await m.save("User prefers short answers.", category="preference")
        await m.save("User's sister is named Priya.", category="person", subject="Priya")
        await m.save("Priya is a doctor in Pune.", category="person", subject="priya")
        await m.save("User is selling Exinent.", category="project", subject="Exinent sale")
        with app.db.lock, app.db.conn:
            for text, secret in (("Garage code is 4417", 1), ("Plumber is Ravi, 555-0142", 0)):
                app.db.conn.execute(
                    "INSERT INTO notes (text, created_at, secret) VALUES (?, ?, ?)",
                    (text, "2026-10-03T10:00:00+00:00", secret),
                )
            app.db.conn.execute(
                "INSERT INTO reminders (text, due_at, created_at) VALUES (?, ?, ?)",
                ("Call the bank", "2030-10-05T03:30:00+00:00", "2026-10-03T10:00:00+00:00"),
            )
        conv = app.conversations.new()
        async for _ in app.conversations.send(conv.id, "My sister Priya is visiting next week"):
            pass
        sync = app.make_vault_sync()
        sync.grace_s = 0  # tests: deletions count at once (see test_grace_period)
        await sync.cycle()
        return app, sync, root, provider

    return build


def read(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    # a distinct, settled mtime (not "being edited right now")
    write.n = getattr(write, "n", 0) + 1
    t = time.time() - 30 + write.n * 0.01
    os.utime(p, (t, t))


def facts(app, subject=...) -> list[str]:
    return [m.text for m in app.memory_store.list() if subject is ... or m.subject == subject]


# ---------------------------------------------------------------- writing the vault


async def test_layout(vault):
    app, sync, root, _ = await vault()
    files = {p.relative_to(root).as_posix() for p in root.rglob("*.md")}
    assert {"Home.md", "About me.md", "Reminders.md", "People/Priya.md",
            "Projects/Exinent sale.md"} <= files  # fmt: skip
    assert any(f.startswith("Conversations/") for f in files)
    assert any(f.startswith("Journal/") for f in files)
    priya = read(root, "People/Priya.md")
    assert "- User's sister is named Priya. ^m" in priya and "Priya is a doctor" in priya
    assert "### Mentioned in" in priya and "## My notes" in priya
    about = read(root, "About me.md")
    assert "## Who I am" in about and "User's name is Adi." in about
    everything = "\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.md"))
    assert "4417" not in everything  # secrets never leave the database
    assert "🔒" in everything and "555-0142" in everything
    conv = next(root.glob("Conversations/*/*.md")).read_text(encoding="utf-8")
    assert "**Adi**" in conv and "Mentions: [[Priya]]" in conv
    assert "- [ ] Call the bank 📅 2030-10-0" in read(root, "Reminders.md")


async def test_second_cycle_writes_nothing(vault):
    app, sync, root, _ = await vault()
    report = await sync.cycle()
    assert report.written == [] and report.changes == [] and report.deleted == []


async def test_new_memory_appears_and_your_notes_are_kept(vault):
    app, sync, root, _ = await vault()
    page = read(root, "People/Priya.md").replace("## My notes\n", "## My notes\nLoves mangoes.\n")
    page = page.replace("type: person", "type: person\naliases: [my sister]")
    write(root, "People/Priya.md", page)
    await sync.cycle()
    await app.memory.save("Priya's birthday is in March.", category="person", subject="Priya")
    await sync.cycle()
    text = read(root, "People/Priya.md")
    assert "Priya's birthday is in March." in text
    assert "Loves mangoes." in text and "aliases: [my sister]" in text


# ---------------------------------------------------------------- your edits → Jarvis


async def test_edit_add_delete_facts(vault):
    app, sync, root, _ = await vault()
    page = read(root, "People/Priya.md")
    page = page.replace("Priya is a doctor in Pune.", "Priya is a surgeon in Mumbai.")
    page = page.replace(
        "- User's sister is named Priya.",
        "- Priya plays the veena.\n- User's sister is named Priya.",
    )
    write(root, "People/Priya.md", page)
    report = await sync.cycle()
    ops = sorted(c.op for c in report.changes)
    assert ops == ["add", "update"]
    assert "Priya is a surgeon in Mumbai." in facts(app, "Priya")
    new = next(m for m in app.memory_store.list() if m.text == "Priya plays the veena.")
    assert new.subject == "Priya" and new.category == "person" and new.source == "told"
    assert f"Priya plays the veena. ^m{new.id}" in read(root, "People/Priya.md")  # id added

    page = read(root, "People/Priya.md")
    write(root, "People/Priya.md", "\n".join(ln for ln in page.split("\n") if "veena" not in ln))
    report = await sync.cycle()
    assert [c.op for c in report.changes] == ["delete"]
    assert "Priya plays the veena." not in facts(app)


async def test_about_me_sections_set_category(vault):
    app, sync, root, _ = await vault()
    about = read(root, "About me.md").replace(
        "## Preferences\n", "## Preferences\n- User likes filter coffee.\n"
    )
    write(root, "About me.md", about)
    await sync.cycle()
    m = next(m for m in app.memory_store.list() if "filter coffee" in m.text)
    assert m.category == "preference" and m.subject is None


async def test_move_fact_between_pages(vault):
    app, sync, root, _ = await vault()
    about = read(root, "About me.md")
    line = next(ln for ln in about.split("\n") if "short answers" in ln)
    write(root, "About me.md", about.replace(line + "\n", ""))
    write(root, "People/Priya.md", read(root, "People/Priya.md").replace(
        "- User's sister", line + "\n- User's sister"))  # fmt: skip
    report = await sync.cycle()
    assert not [c for c in report.changes if c.op == "delete"]
    m = next(m for m in app.memory_store.list() if "short answers" in m.text)
    assert m.subject == "Priya" and m.category == "person"


async def test_rename_page_renames_subject(vault):
    app, sync, root, _ = await vault()
    (root / "People/Priya.md").rename(root / "People/Priya Rao.md")
    report = await sync.cycle()
    assert not [c for c in report.changes if c.op == "delete"]
    assert len(facts(app, "Priya Rao")) == 2 and not facts(app, "Priya")
    assert (root / "People/Priya Rao.md").exists() and not (root / "People/Priya.md").exists()


async def test_vault_disappearing_forgets_nothing(vault):
    app, sync, root, _ = await vault()
    now = [time.time()]
    sync.clock = lambda: now[0]
    before = len(facts(app))
    for p in ("People/Priya.md", "Projects/Exinent sale.md", "About me.md"):
        (root / p).unlink()
    for _ in range(3):
        report = await sync.cycle()
        assert not report.changes and len(facts(app)) == before
    assert not (root / "People/Priya.md").exists()  # not rushed back (could be OneDrive)
    now[0] += 61
    await sync.cycle()
    assert (root / "People/Priya.md").exists()  # a minute later: written again
    assert len(facts(app)) == before


async def test_deleting_a_whole_page_keeps_its_facts(vault):
    app, sync, root, _ = await vault()
    (root / "People/Priya.md").unlink()
    for _ in range(3):
        await sync.cycle()
    assert len(facts(app, "Priya")) == 2  # delete the lines to make Jarvis forget


async def test_conflict_copy_is_ignored(vault):
    app, sync, root, _ = await vault()
    original = read(root, "People/Priya.md")
    write(root, "People/Priya-DESKTOP-1.md", original.replace("doctor", "dentist"))
    for _ in range(2):
        await sync.cycle()
    assert (root / "People/Priya.md").exists()
    assert sorted(facts(app, "Priya")) == [
        "Priya is a doctor in Pune.",
        "User's sister is named Priya.",
    ]


async def test_your_lines_in_jarvis_part_are_never_dropped(vault):
    app, sync, root, _ = await vault()
    page = read(root, "People/Priya.md").replace(
        "### Mentioned in\n", "### Mentioned in\nMet her at the airport\n| a | b |\n"
    )
    write(root, "People/Priya.md", page)
    await sync.cycle()
    text = read(root, "People/Priya.md")
    assert "Met her at the airport" in text and "| a | b |" in text


async def test_file_being_edited_is_left_alone(vault):
    app, sync, root, _ = await vault()
    sync.quiet_s = 100
    write(root, "People/Priya.md", read(root, "People/Priya.md").replace("doctor", "doc"))
    report = await sync.cycle()
    assert "People/Priya.md" in report.skipped_busy and not report.changes
    assert "Priya is a doctor in Pune." in facts(app)
    sync.quiet_s = 0
    await sync.cycle()
    assert "Priya is a doc in Pune." in facts(app)


async def test_markers_removed_are_restored(vault):
    app, sync, root, _ = await vault()
    page = read(root, "Projects/Exinent sale.md")
    page = "\n".join(ln for ln in page.split("\n") if "jarvis:" not in ln or "%%" not in ln)
    write(root, "Projects/Exinent sale.md", page + "\nMy own plan.\n")
    await sync.cycle()
    text = read(root, "Projects/Exinent sale.md")
    assert "%% jarvis:start" in text and "My own plan." in text
    assert text.count("User is selling Exinent.") == 1


# ---------------------------------------------------------------- notes & reminders


async def test_notes_edit_delete_and_secret(vault):
    app, sync, root, _ = await vault()
    plumber = next(p for p in (root / "Notes").glob("Plumber*.md"))
    rel = plumber.relative_to(root).as_posix()
    write(root, rel, read(root, rel).replace("555-0142", "555-9999"))
    await sync.cycle()
    with app.db.lock:
        texts = [r[0] for r in app.db.conn.execute("SELECT text FROM notes").fetchall()]
    assert "Plumber is Ravi, 555-9999" in texts
    now = [time.time()]
    sync.clock = lambda: now[0]
    plumber.unlink()
    secret = next((root / "Notes").glob("Secret*.md"))
    secret.unlink()
    await sync.cycle()
    now[0] += 30
    await sync.cycle()
    with app.db.lock:
        assert app.db.conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 2  # a blip
    assert not plumber.exists()  # …and it isn't written back while we wait
    now[0] += 31
    await sync.cycle()
    with app.db.lock:
        texts = [r[0] for r in app.db.conn.execute("SELECT text FROM notes").fetchall()]
    assert texts == ["Garage code is 4417"]  # a secret isn't deleted from a file
    assert secret.exists()  # …its placeholder comes back


async def test_new_secret_notes_are_flagged(make_app):
    app, _ = make_app()
    ctx = app.tool_context()
    await NotesAdd().run(NoteAddArgs(text="wifi password is lemonade42"), ctx)
    await NotesAdd().run(NoteAddArgs(text="Alarm: 9182", secret=True), ctx)
    await NotesAdd().run(NoteAddArgs(text="Ideas for the Diwali party"), ctx)
    with app.db.lock:
        flags = [r[0] for r in app.db.conn.execute("SELECT secret FROM notes ORDER BY id")]
    assert flags == [1, 1, 0]
    assert looks_secret("door code 4417") and not looks_secret("code review on Friday")


async def test_reminders_tick_delete_add(vault):
    app, sync, root, _ = await vault()
    with app.db.lock, app.db.conn:
        app.db.conn.execute(
            "INSERT INTO reminders (text, due_at, created_at) VALUES ('Stretch', "
            "'2030-01-01T09:00:00+00:00', '2026-10-03T10:00:00+00:00')"
        )
    await sync.cycle()
    page = read(root, "Reminders.md")
    page = page.replace("- [ ] Call the bank", "- [x] Call the bank")
    page = "\n".join(ln for ln in page.split("\n") if "Stretch" not in ln)
    page = page.replace("## Upcoming\n", "## Upcoming\n- [ ] Renew passport 📅 2030-11-02 10:30\n")
    write(root, "Reminders.md", page)
    report = await sync.cycle()
    with app.db.lock:
        rows = {
            r["text"]: (r["status"], r["due_at"])
            for r in app.db.conn.execute("SELECT * FROM reminders").fetchall()
        }
    assert rows["Call the bank"][0] == "done"
    assert rows["Stretch"][0] == "cancelled"
    assert rows["Renew passport"][0] == "pending" and rows["Renew passport"][1].startswith(
        "2030-11-0"
    )
    assert "Renew passport 📅 2030-11-02 10:30 ^r" in read(root, "Reminders.md")
    assert len(report.imported) == 3


async def test_deleted_conversation_page_stays_deleted(vault):
    app, sync, root, _ = await vault()
    conv = next(root.glob("Conversations/*/*.md"))
    conv.unlink()
    await sync.cycle()
    await sync.cycle()
    assert not conv.exists()


# ---------------------------------------------------------------- inbox & search


async def test_inbox_learns_and_files(vault):
    ops = {"operations": [{"op": "add", "text": "User plays tennis on Sundays.",
                           "category": "preference", "subject": None}]}  # fmt: skip
    app, sync, root, _ = await vault(turns=[json.dumps(ops)])
    write(root, "Inbox/weekend.md", "I play tennis every Sunday morning.")
    report = await sync.cycle()
    assert report.filed == ["Inbox/weekend.md → Notes/weekend.md"]
    assert "User plays tennis on Sundays." in facts(app)
    assert not (root / "Inbox/weekend.md").exists()
    assert read(root, "Notes/weekend.md") == "I play tennis every Sunday morning."  # untouched


async def test_vault_search_finds_your_notes(vault):
    app, sync, root, _ = await vault()
    write(
        root, "Ideas/Startup.md", "# Startup\n\n## Pricing\nCharge 49 dollars a month per store.\n"
    )
    await sync.cycle()
    hits = await app.vault_index.search("pricing per store", limit=3)
    assert hits and hits[0]["page"] == "Ideas/Startup" and hits[0]["section"] == "Pricing"
    out = await VaultSearch().run(VaultSearchArgs(query="pricing per store"), app.tool_context())
    assert out["results"][0]["page"] == "Ideas/Startup"
    # Jarvis's own facts aren't duplicated into the search index (they're in memory)
    assert not await app.vault_index.search("doctor in Pune")
    (root / "Ideas/Startup.md").unlink()
    await sync.cycle()
    assert not await app.vault_index.search("pricing per store")


async def test_no_vault_means_no_vault_tool(make_app):
    app, _ = make_app()
    assert app.registry.get("vault_search") is None and app.vault_root is None
    assert app.start_vault() is False


async def test_worker_thread_syncs(vault):
    app, sync, root, _ = await vault()
    await app.memory.save("Priya runs marathons.", category="person", subject="Priya")
    worker = VaultWorker(app.make_vault_sync, interval_s=0.05)
    worker.start()
    for _ in range(100):
        if "marathons" in read(root, "People/Priya.md"):
            break
        await asyncio.sleep(0.05)
    assert worker.stop() is True
    assert "marathons" in read(root, "People/Priya.md") and worker.last_error is None


# ---------------------------------------------------------------- subjects


async def test_extractor_gives_subjects(make_app):
    ops = {"operations": [
        {"op": "add", "text": "User's friend Rahul lives in Austin.", "category": "person",
         "subject": "rahul"},
        {"op": "add", "text": "User lives in Hyderabad.", "category": "profile",
         "subject": "Hyderabad"},
    ]}  # fmt: skip
    app, provider = make_app([json.dumps(ops)])
    changes = await MemoryExtractor(app.models, app.memory).extract(
        "My friend Rahul lives in Austin and I live in Hyderabad", force=True
    )
    assert len(changes) == 2
    by_text = {m.text: m for m in app.memory_store.list()}
    assert by_text["User's friend Rahul lives in Austin."].subject == "Rahul"
    assert by_text["User lives in Hyderabad."].subject is None  # about the user
    assert "KNOWN SUBJECTS" in provider.calls[0]["messages"][1].content


async def test_assign_subjects_for_old_memories(make_app):
    app, provider = make_app()
    a, _ = await app.memory.save("User's mom is Lakshmi.", category="person")
    b, _ = await app.memory.save("User is learning Telugu.", category="other")
    provider.turns.append(json.dumps({"subjects": [{"id": a.id, "subject": "Mom"},
                                                   {"id": b.id, "subject": None}]}))  # fmt: skip
    assert await assign_subjects(app.models, app.memory_store) == 1
    assert (
        app.memory_store.get(a.id).subject == "Mom" and app.memory_store.get(b.id).subject is None
    )


# ---------------------------------------------------------------- markdown helpers


def test_frontmatter_and_items():
    fm = Frontmatter.parse(["type: person", "aliases:", "  - Pri", "  - my sister", "x: 'a b'"])
    assert fm.get("aliases") == ["Pri", "my sister"] and fm.get("x") == "a b"
    fm.set("updated", "2026-10-03")
    assert fm.render().endswith("updated: 2026-10-03\n---\n")
    items = parse_items("## Upcoming\n- [x] Pay rent 📅 2026-10-01 ^r4\n- [[link]]\n- plain")
    assert (items[0].kind, items[0].id, items[0].checked, items[0].section) == (
        "r", 4, True, "Upcoming")  # fmt: skip
    assert [i.text for i in items] == ["Pay rent 📅 2026-10-01", "plain"]
    page = build_page(None, title="T", region="- a ^m1", props={"type": "x"})
    parsed = parse_page(page)
    assert parsed.region == "- a ^m1" and parsed.after.strip() == "## My notes"
    assert safe_name('a/b: "c"?') == "a b c" and safe_name("   ") == "Untitled"


def test_resolve_vault(tmp_path):
    assert resolve_vault(str(tmp_path)) == tmp_path.resolve()
    assert resolve_vault("Documents/obsidian_notes").name == "obsidian_notes"


# ---------------------------------------------------------------- safety (review findings)


def write_now(root: Path, rel: str, text: str) -> None:
    """A file you're editing right now (fresh mtime)."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


async def test_editing_one_page_never_overwrites_another(vault):
    app, sync, root, _ = await vault()
    sync.quiet_s = 20  # write() leaves files 30 s old; write_now() is "being typed"
    write(root, "People/Priya.md", read(root, "People/Priya.md").replace(
        "- User's sister", "- Priya plays the veena.\n- User's sister"))  # fmt: skip
    write_now(root, "Projects/Exinent sale.md", read(root, "Projects/Exinent sale.md") + "typing…")
    await sync.cycle()
    assert "Priya plays the veena." in facts(app, "Priya")  # read even though another is busy
    assert "typing…" in read(root, "Projects/Exinent sale.md")  # busy page untouched


async def test_any_line_you_write_is_a_fact(vault):
    app, sync, root, _ = await vault()
    page = read(root, "People/Priya.md").replace(
        "- User's sister",
        "Priya is allergic to cats.\n1. Priya studied at AIIMS.\n- [[Mom]] is Priya's mother.\n"
        "- User's sister",
    )
    write(root, "People/Priya.md", page)
    await sync.cycle()
    got = facts(app, "Priya")
    assert {"Priya is allergic to cats.", "Priya studied at AIIMS.",
            "[[Mom]] is Priya's mother."} <= set(got)  # fmt: skip
    await sync.cycle()
    assert len(facts(app, "Priya")) == len(got)  # no re-import loop


async def test_moving_a_page_into_a_subfolder(vault):
    app, sync, root, _ = await vault()
    (root / "People/Family").mkdir()
    (root / "People/Priya.md").rename(root / "People/Family/Priya.md")
    for _ in range(3):
        report = await sync.cycle()
        assert not [c for c in report.changes if c.op == "delete"]
    assert len(facts(app, "Priya")) == 2
    assert not (root / "People/Priya.md").exists()
    await app.memory.save("Priya is visiting in March.", category="person", subject="Priya")
    await sync.cycle()
    assert "visiting in March" in read(root, "People/Family/Priya.md")  # follows the move


async def test_grace_period_and_unreadable_files(vault, monkeypatch):
    app, sync, root, _ = await vault()
    now = [time.time()]
    sync.clock, sync.grace_s = (lambda: now[0]), 10
    page = read(root, "People/Priya.md")
    write(root, "People/Priya.md", "\n".join(ln for ln in page.split("\n") if "doctor" not in ln))
    await sync.cycle()
    assert "Priya is a doctor in Pune." in facts(app)  # not yet…
    assert "doctor" not in read(root, "People/Priya.md")  # …and not written back meanwhile
    now[0] += 11
    await sync.cycle()
    assert "Priya is a doctor in Pune." not in facts(app)  # …gone after the grace period

    real = Path.read_bytes

    def flaky(self):
        if self.name == "Exinent sale.md":
            raise PermissionError("locked by OneDrive")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", flaky)
    write(root, "Projects/Exinent sale.md", "garbage while locked")
    for _ in range(2):
        now[0] += 20
        report = await sync.cycle()
        assert not [c for c in report.changes if c.op == "delete"]
    assert "User is selling Exinent." in facts(app)


async def test_failed_write_never_duplicates_reminders(vault, monkeypatch):
    app, sync, root, _ = await vault()
    write(root, "Reminders.md", read(root, "Reminders.md").replace(
        "## Upcoming\n", "## Upcoming\n- [ ] Pay rent 📅 2030-11-01 9am\n"))  # fmt: skip
    real = sync._write

    def failing(rel, text):
        if rel == "Reminders.md":
            raise PermissionError("in use")
        real(rel, text)

    monkeypatch.setattr(sync, "_write", failing)
    for _ in range(3):
        await sync.cycle()
    with app.db.lock:
        rows = app.db.conn.execute(
            "SELECT due_at FROM reminders WHERE text = 'Pay rent'"
        ).fetchall()
    assert len(rows) == 1
    from datetime import datetime

    assert datetime.fromisoformat(rows[0][0]).astimezone().hour == 9


async def test_inbox_locked_and_non_utf8(vault, monkeypatch):
    app, sync, root, _ = await vault(turns=['{"operations": []}'] * 3)
    raw = "Café résumé notes".encode("cp1252")
    (root / "Inbox").mkdir()
    (root / "Inbox/old.txt").write_bytes(raw)
    t = time.time() - 30
    os.utime(root / "Inbox/old.txt", (t, t))
    real = os.replace

    def locked(src, dst):
        if "Inbox" in str(src):
            raise PermissionError("open in Notepad")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", locked)
    for _ in range(2):
        await sync.cycle()
    assert not list((root / "Notes").glob("old*"))  # nothing copied while locked
    monkeypatch.setattr(os, "replace", real)
    await sync.cycle()
    assert (root / "Notes/old.txt").read_bytes() == raw  # moved, bytes untouched


async def test_cleanup_keeps_pages_with_your_text(vault):
    app, sync, root, _ = await vault()
    page = read(root, "Projects/Exinent sale.md").replace(
        "type: project", "tags: [important]\ntype: project"
    )
    page = page.replace("# Exinent sale\n", "# Exinent sale\nTalk to the banker first.\n")
    write(root, "Projects/Exinent sale.md", page)
    await sync.cycle()
    mem = next(m for m in app.memory_store.list() if m.subject == "Exinent sale")
    app.memory.forget(mem.id)
    await sync.cycle()
    text = read(root, "Projects/Exinent sale.md")
    assert "Talk to the banker first." in text and "tags: [important]" in text


async def test_your_own_page_is_merged_not_replaced(vault):
    app, sync, root, _ = await vault()
    write(
        root,
        "People/Mom.md",
        "---\ntype: family\nDate created: 2026-01-01\n---\n# Mom\nCall on Sundays.\n",
    )
    await sync.cycle()
    assert "Call on Sundays." in read(root, "People/Mom.md")
    await app.memory.save("User's mom is Lakshmi.", category="person", subject="Mom")
    await sync.cycle()
    text = read(root, "People/Mom.md")
    assert "Call on Sundays." in text and "User's mom is Lakshmi." in text
    assert "type: family" in text and "Date created: 2026-01-01" in text


async def test_note_never_takes_over_someone_elses_file(vault):
    app, sync, root, _ = await vault()
    plumber = next((root / "Notes").glob("Plumber*.md"))
    plumber.rename(root / "Notes/Groceries.md")
    await sync.cycle()
    with app.db.lock, app.db.conn:
        app.db.conn.execute(
            "INSERT INTO notes (text, created_at, secret) VALUES ('Groceries', ?, 0)",
            ("2026-10-03T11:00:00+00:00",),
        )
    await sync.cycle()
    assert "555-0142" in read(root, "Notes/Groceries.md")
    assert (root / "Notes/Groceries (2).md").exists()
    assert (await sync.cycle()).written == []  # settled: no rewriting back and forth


async def test_lowercase_rename_settles(vault):
    app, sync, root, _ = await vault()
    (root / "People/Priya.md").rename(root / "People/priya rao.md")
    await sync.cycle()
    await sync.cycle()
    people = sorted(p.name for p in (root / "People").glob("*.md"))
    assert people == ["priya rao.md"]
    assert (await sync.cycle()).changes == []


async def test_same_fact_on_another_page_is_a_new_fact(vault):
    app, sync, root, _ = await vault()
    write(
        root,
        "Topics/Family.md",
        "# Family\n%% jarvis:start %%\n- Priya is a doctor in Pune.\n%% jarvis:end %%\n",
    )
    await sync.cycle()
    assert "Priya is a doctor in Pune." in facts(app, "Priya")
    assert "Priya is a doctor in Pune." in facts(app, "Family")


async def test_subject_marker_survives(vault):
    app, sync, root, _ = await vault()
    await app.prepare_vault(sync)
    await sync.cycle()
    with app.db.lock:
        assert app.db.conn.execute(
            "SELECT 1 FROM vault_files WHERE key = 'meta:subjects-assigned'"
        ).fetchone()


async def test_moved_about_me_still_allows_deletions(vault):
    app, sync, root, _ = await vault()
    (root / "Me").mkdir()
    (root / "About me.md").rename(root / "Me/About me.md")
    await sync.cycle()
    page = read(root, "People/Priya.md")
    write(root, "People/Priya.md", "\n".join(ln for ln in page.split("\n") if "doctor" not in ln))
    await sync.cycle()
    assert "Priya is a doctor in Pune." not in facts(app)
    assert not (root / "About me.md").exists()  # stays where you moved it


async def test_deleting_several_notes_is_honoured(vault):
    app, sync, root, _ = await vault()
    now = [time.time()]
    sync.clock = lambda: now[0]
    with app.db.lock, app.db.conn:
        for t in ("Alpha idea", "Beta idea", "Gamma idea"):
            app.db.conn.execute(
                "INSERT INTO notes (text, created_at, secret) VALUES (?, ?, 0)",
                (t, "2026-10-03T12:00:00+00:00"),
            )
    await sync.cycle()
    for name in ("Alpha idea", "Beta idea", "Gamma idea"):
        (root / f"Notes/{name}.md").unlink()
    await sync.cycle()
    now[0] += 61
    await sync.cycle()
    with app.db.lock:
        left = [r[0] for r in app.db.conn.execute("SELECT text FROM notes").fetchall()]
    assert not any("idea" in t for t in left)
    assert not (root / "Notes/Alpha idea.md").exists()
