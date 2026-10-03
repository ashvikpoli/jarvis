"""What the vault should contain, computed from Jarvis's database (no file I/O here).

Layout:
    Home.md · About me.md · Reminders.md
    People/<name>.md · Projects/<name>.md · Topics/<name>.md
    Notes/<title>.md · Conversations/YYYY-MM/<date time title>.md · Journal/YYYY/<date>.md
    Inbox/  (you drop files here; Jarvis learns from them and files them in Notes/)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from jarvis.memory.store import Memory
from jarvis.storage import Database
from jarvis.vault.markdown import link, safe_name, sha

FOLDERS = {"person": "People", "project": "Projects", "other": "Topics"}
CATEGORY_FOR_FOLDER = {"People": "person", "Projects": "project", "Topics": "other"}
ABOUT = "About me.md"
REMINDERS = "Reminders.md"
HOME = "Home.md"
# About me sections ↔ memory categories (facts without a subject page)
ABOUT_SECTIONS = [
    ("Who I am", "profile"),
    ("Preferences", "preference"),
    ("People", "person"),
    ("Projects", "project"),
    ("Other", "other"),
]
SECTION_CATEGORY = {title.lower(): cat for title, cat in ABOUT_SECTIONS}
SECRET_TEXT = "🔒 Kept only inside Jarvis (it looks like a password or code). Ask Jarvis for it."


@dataclass
class Page:
    key: str  # stable identity, also stored in the file's frontmatter
    path: str  # relative to the vault, with forward slashes
    title: str
    region: str  # Jarvis's part of the page
    props: dict
    kind: str  # about | person | project | topic | note | reminders | conversation | journal | home
    ids: list[int] = field(default_factory=list)  # memory / reminder ids shown
    version: str | None = None  # skip re-rendering when the source hasn't changed
    user_section: str = "## My notes\n\n"


def _local(iso: str | None) -> datetime | None:
    if not iso:
        return None
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone()


def _today() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def memory_page_path(category: str, subject: str | None) -> str:
    if subject and category in FOLDERS:
        return f"{FOLDERS[category]}/{safe_name(subject)}.md"
    if subject and category in ("profile", "preference"):
        return ABOUT
    return ABOUT


def page_target(path: str) -> tuple[str, str | None]:
    """(category, subject) a fact gets when it's on this page."""
    if "/" in path:
        folder, name = path.split("/", 1)
        if folder in CATEGORY_FOR_FOLDER and name.endswith(".md") and "/" not in name:
            return CATEGORY_FOR_FOLDER[folder], name[:-3]
    return "other", None


def _fact(m: Memory) -> str:
    return f"- {' '.join(m.text.split())} ^m{m.id}"


# ---------------------------------------------------------------- memory pages


def memory_pages(memories: list[Memory], mentions: dict[str, list[tuple[str, str]]]) -> list[Page]:
    """About me + one page per person / project / topic."""
    groups: dict[str, list[Memory]] = defaultdict(list)
    about: dict[str, list[Memory]] = defaultdict(list)
    for m in memories:
        path = memory_page_path(m.category, m.subject)
        if path == ABOUT:
            about[m.category].append(m)
        else:
            groups[path].append(m)

    pages: list[Page] = []
    lines: list[str] = []
    for title, cat in ABOUT_SECTIONS:
        if about.get(cat) or cat in ("profile", "preference"):
            lines.append(f"## {title}")
            lines += [_fact(m) for m in about.get(cat, [])]
            lines.append("")
    pages.append(
        Page(
            key="page:" + ABOUT,
            path=ABOUT,
            title="About me",
            region="\n".join(lines),
            props={"type": "about", "jarvis": "page:" + ABOUT},
            kind="about",
            ids=[m.id for ms in about.values() for m in ms],
        )
    )
    for path, ms in sorted(groups.items()):
        cat = ms[0].category
        subject = path.rsplit("/", 1)[-1][:-3]
        region = [_fact(m) for m in ms]
        seen = mentions.get(subject.lower(), [])
        if seen:
            region += ["", "### Mentioned in"] + [f"- {link(p, t)}" for p, t in seen[:12]]
        kind = {"person": "person", "project": "project"}.get(cat, "topic")
        pages.append(
            Page(
                key="page:" + path,
                path=path,
                title=subject,
                region="\n".join(region),
                props={"type": kind, "jarvis": "page:" + path},
                kind=kind,
                ids=[m.id for m in ms],
            )
        )
    return pages


# ---------------------------------------------------------------- notes


def note_title(text: str) -> str:
    first = text.strip().split("\n", 1)[0]
    words = first.split()
    title = " ".join(words[:8]) + ("…" if len(words) > 8 else "")
    return safe_name(title, 60)


def note_pages(db: Database) -> list[Page]:
    with db.lock:
        rows = db.conn.execute(
            "SELECT id, text, created_at, updated_at, secret FROM notes ORDER BY id"
        ).fetchall()
    pages, used = [], set()
    for r in rows:
        secret = bool(r["secret"])
        base = note_title("Secret note" if secret else r["text"])
        name = base if base.lower() not in used else f"{base} ({r['id']})"
        used.add(name.lower())
        created = _local(r["created_at"])
        pages.append(
            Page(
                key=f"note:{r['id']}",
                path=f"Notes/{name}.md",
                title=name,
                region=SECRET_TEXT if secret else r["text"].strip(),
                props={
                    "type": "note",
                    "jarvis": f"note:{r['id']}",
                    "created": created.strftime("%Y-%m-%d %H:%M") if created else "",
                    **({"secret": True} if secret else {}),
                },
                kind="note",
                ids=[r["id"]],
                version=f"{r['updated_at'] or r['created_at']}|{secret}",
                user_section="",
            )
        )
    return pages


# ---------------------------------------------------------------- reminders

DUE_FMT = "%Y-%m-%d %H:%M"


def reminders_page(db: Database, *, done_days: int = 14) -> Page:
    with db.lock:
        rows = db.conn.execute(
            "SELECT * FROM reminders WHERE status = 'pending' "
            "OR (status IN ('fired', 'done') AND due_at >= datetime('now', ?)) "
            "ORDER BY due_at",
            (f"-{done_days} days",),
        ).fetchall()
    upcoming = [r for r in rows if r["status"] == "pending"]
    done = [r for r in rows if r["status"] != "pending"]

    def line(r) -> str:
        due = _local(r["due_at"])
        box = " " if r["status"] == "pending" else "x"
        return f"- [{box}] {r['text']} 📅 {due.strftime(DUE_FMT)} ^r{r['id']}"

    region = ["## Upcoming", *[line(r) for r in upcoming]]
    region += ["", "## Done (last 2 weeks)", *[line(r) for r in reversed(done)]]
    return Page(
        key="reminders",
        path=REMINDERS,
        title="Reminders",
        region="\n".join(region),
        props={"type": "reminders", "jarvis": "reminders"},
        kind="reminders",
        ids=[r["id"] for r in upcoming],
        user_section=(
            "## How to use\n\nTick a box to mark it done. Delete a line to cancel it. Add a "
            "line like `- [ ] Call the bank 📅 2026-10-05 09:00` under Upcoming to set one.\n"
        ),
    )


# ---------------------------------------------------------------- conversations


def _word_re(name: str) -> re.Pattern:
    return re.compile(rf"(?<![\w]){re.escape(name)}(?![\w])", re.I)


def conversation_pages(
    db: Database, subjects: list[str], *, user_name: str = "You", cache: dict | None = None
) -> tuple[list[Page], dict[str, list[tuple[str, str]]]]:
    """Transcripts, plus subject → [(conversation path, title)] for "Mentioned in".

    `cache` (conversation id → page) skips re-reading conversations that haven't changed.
    """
    cache = {} if cache is None else cache
    subjects_key = "|".join(subjects)
    with db.lock:
        convs = db.conn.execute(
            "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS n "
            "FROM conversations c ORDER BY c.created_at"
        ).fetchall()
    patterns = [(s, _word_re(s)) for s in subjects]
    pages: list[Page] = []
    mentions: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for c in convs:
        if not c["n"]:
            continue
        stamp = (c["updated_at"], c["n"], c["title"], subjects_key)
        hit = cache.get(c["id"])
        if hit and hit[0] == stamp:
            page, found, label = hit[1], hit[2], hit[3]
            for s in found:
                mentions[s.lower()].append((page.path, label))
            pages.append(page)
            continue
        with db.lock:
            msgs = db.conn.execute(
                "SELECT role, content, tool_calls, interrupted, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (c["id"],),
            ).fetchall()
        created = _local(c["created_at"])
        title = c["title"] or "Conversation"
        name = safe_name(f"{created:%Y-%m-%d %H.%M} {title}", 90)
        path = f"Conversations/{created:%Y-%m}/{name}.md"
        text_all = " ".join(m["content"] for m in msgs if m["role"] in ("user", "assistant"))
        found = [s for s, rx in patterns if rx.search(text_all)]
        label = f"{created:%d %b %Y} — {title}"
        for s in found:
            mentions[s.lower()].append((path, label))
        body: list[str] = []
        if found:
            body += ["Mentions: " + " · ".join(f"[[{s}]]" for s in found), ""]
        for m in msgs:
            when = _local(m["created_at"])
            if m["role"] == "user":
                body += [f"**{user_name}** · {when:%H:%M}", m["content"].strip(), ""]
            elif m["role"] == "assistant":
                calls = json.loads(m["tool_calls"]) if m["tool_calls"] else []
                for call in calls:
                    args = ", ".join(f"{k}={v!r}" for k, v in call.get("arguments", {}).items())
                    body.append(f"> ⚙ {call.get('name')}({args[:150]})")
                if m["content"].strip():
                    stop = " ⏹" if m["interrupted"] else ""
                    body += ["**Jarvis**" + stop, m["content"].strip(), ""]
                elif calls:
                    body.append("")
            elif m["role"] == "tool":
                first = " ".join(m["content"].split())[:160]
                body += [f"> ↳ {first}", ""]
        page = Page(
            key=f"conv:{c['id']}",
            path=path,
            title=title,
            region="\n".join(body),
            props={
                "type": "conversation",
                "jarvis": f"conv:{c['id']}",
                "date": f"{created:%Y-%m-%d %H:%M}",
                "mentions": [f"[[{s}]]" for s in found],
            },
            kind="conversation",
            version=f"{c['updated_at']}|{c['n']}|{','.join(found)}",
        )
        cache[c["id"]] = (stamp, page, found, label)
        pages.append(page)
    for v in mentions.values():
        v.reverse()  # newest first
    return pages, mentions


# ---------------------------------------------------------------- journal & home


def journal_pages(db: Database, memories: list[Memory], conv_pages: list[Page]) -> list[Page]:
    per_day: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for p in conv_pages:
        day, hm = p.props["date"].split(" ")
        per_day[day]["Conversations"].append(f"- {hm} {link(p.path, p.title)}")
    for m in memories:
        day = _local(m.created_at).strftime("%Y-%m-%d")
        where = memory_page_path(m.category, m.subject)
        per_day[day]["Learned"].append(f"- {link(where)}: {m.text}")
    with db.lock:
        rows = db.conn.execute(
            "SELECT text, due_at, status FROM reminders WHERE status != 'cancelled'"
        ).fetchall()
    for r in rows:
        due = _local(r["due_at"])
        mark = "✅" if r["status"] in ("fired", "done") else "⏰"
        per_day[due.strftime("%Y-%m-%d")]["Reminders"].append(f"- {mark} {due:%H:%M} {r['text']}")

    pages = []
    for day, sections in sorted(per_day.items()):
        region = []
        for name in ("Conversations", "Learned", "Reminders"):
            if sections.get(name):
                region += [f"## {name}", *sections[name], ""]
        text = "\n".join(region)
        pages.append(
            Page(
                key=f"day:{day}",
                path=f"Journal/{day[:4]}/{day}.md",
                title=datetime.strptime(day, "%Y-%m-%d").strftime("%A, %d %B %Y"),
                region=text,
                props={"type": "journal", "jarvis": f"day:{day}", "date": day},
                kind="journal",
                version=sha(text),
            )
        )
    return pages


def home_page(
    memory_pgs: list[Page], note_pgs: list[Page], conv_pages: list[Page], open_reminders: int
) -> Page:
    today = _today()
    lines = [
        f"**Today:** {link(f'Journal/{today[:4]}/{today}', today)} · "
        f"**Reminders:** {open_reminders} open → [[Reminders]] · **You:** [[About me]]",
        "",
    ]
    for kind, title in (("person", "People"), ("project", "Projects"), ("topic", "Topics")):
        items = [p for p in memory_pgs if p.kind == kind]
        if items:
            lines.append(f"## {title}")
            lines += [f"- {link(p.path)} ({len(p.ids)})" for p in items]
            lines.append("")
    if conv_pages:
        lines.append("## Recent conversations")
        for p in sorted(conv_pages, key=lambda p: p.props["date"], reverse=True)[:8]:
            lines.append(f"- {p.props['date']} {link(p.path, p.title)}")
        lines.append("")
    if note_pgs:
        lines.append("## Notes")
        lines += [f"- {link(p.path)}" for p in note_pgs[-10:]]
        lines.append("")
    lines.append(
        "Drop anything into **Inbox/** and Jarvis will learn from it and file it in Notes."
    )
    text = "\n".join(lines)
    return Page(
        key="home",
        path=HOME,
        title="Jarvis",
        region=text,
        props={"type": "home", "jarvis": "home"},
        kind="home",
        version=sha(text),
    )
