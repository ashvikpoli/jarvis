"""Small, dependency-free Markdown helpers for Obsidian pages.

A Jarvis page looks like:

    ---
    type: person
    jarvis: page:People/Priya.md      ← which Jarvis record this file shows
    aliases: [my sister]              ← yours; kept as-is
    ---
    # Priya
    %% jarvis:start — … %%
    - User's sister is named Priya. ^m12
    %% jarvis:end %%

    ## My notes
    (yours — Jarvis never rewrites anything outside the markers)

`%% … %%` is an Obsidian comment: invisible in reading view.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

START = "%% jarvis:start — Jarvis keeps this part in sync. Edit, add or delete lines freely. %%"
END = "%% jarvis:end %%"
_START_RE = re.compile(r"^\s*%%\s*jarvis:start\b.*%%\s*$")
_END_RE = re.compile(r"^\s*%%\s*jarvis:end\s*%%\s*$")
MY_NOTES = "## My notes\n\n"

# "- text ^m12", "- [x] text 📅 2026-10-04 09:00 ^r5"
_BULLET = re.compile(r"^\s*[-*+]\s+(?:\[(?P<box>[ xX])\]\s+)?(?P<text>.*?)\s*$")
_BLOCK_ID = re.compile(r"\s*\^(?P<kind>[mr])(?P<id>\d+)\s*$")
_HEADING = re.compile(r"^\s*#{2,6}\s+(?P<title>.+?)\s*#*\s*$")


def sha(text: str) -> str:
    return hashlib.sha1(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


_BAD = re.compile(r'[<>:"/\\|?*#^\[\]\x00-\x1f]')
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}  # fmt: skip


def safe_name(name: str, limit: int = 80) -> str:
    """A file name Windows, Obsidian links and OneDrive all accept."""
    s = _BAD.sub(" ", name)
    s = " ".join(s.split()).strip(" .")
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0].strip(" .") or s[:limit]
    if s.split(".")[0].lower() in _RESERVED:
        s += " _"
    return s or "Untitled"


def link(path: str, label: str | None = None) -> str:
    """[[Folder/Page|label]] — full path so it resolves even if names repeat."""
    target = path[:-3] if path.endswith(".md") else path
    shown = label or target.rsplit("/", 1)[-1]
    return f"[[{target}]]" if shown == target else f"[[{target}|{shown}]]"


# ---------------------------------------------------------------- frontmatter


def _scalar(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v)
    if s == "" or re.search(r"[:#\[\]{},&*!|>'\"%@`]", s) or s != s.strip():
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def _dump(key: str, value) -> list[str]:
    if isinstance(value, list | tuple):
        if not value:
            return [f"{key}: []"]
        return [f"{key}:"] + [f"  - {_scalar(v)}" for v in value]
    return [f"{key}: {_scalar(value)}"]


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return s


@dataclass
class Frontmatter:
    """Ordered key → raw lines, so keys we don't manage survive untouched."""

    entries: list[tuple[str, list[str]]] = field(default_factory=list)

    @classmethod
    def parse(cls, lines: list[str]) -> Frontmatter:
        fm = cls()
        for line in lines:
            m = re.match(r"^([^\s:#\-][^:]*?)\s*:(\s.*|)$", line)
            if m and not line.startswith((" ", "\t")):
                fm.entries.append((m.group(1), [line]))
            elif fm.entries:
                fm.entries[-1][1].append(line)
            else:
                fm.entries.append(("", [line]))
        return fm

    def get(self, key: str):
        for k, lines in self.entries:
            if k == key:
                first = lines[0].split(":", 1)[1].strip()
                if first.startswith("[") and first.endswith("]"):
                    inner = first[1:-1].strip()
                    return [_unquote(p) for p in inner.split(",") if p.strip()] if inner else []
                if not first and len(lines) > 1:
                    return [
                        _unquote(ln.strip()[1:].strip())
                        for ln in lines[1:]
                        if ln.strip().startswith("-")
                    ]
                return _unquote(first)
        return None

    def set(self, key: str, value) -> None:
        new = _dump(key, value)
        for i, (k, _) in enumerate(self.entries):
            if k == key:
                self.entries[i] = (key, new)
                return
        self.entries.append((key, new))

    def render(self) -> str:
        body = [ln for _, lines in self.entries for ln in lines]
        return "---\n" + "\n".join(body) + "\n---\n"


# ---------------------------------------------------------------- page structure


@dataclass
class ParsedPage:
    frontmatter: Frontmatter
    has_frontmatter: bool
    before: str  # between frontmatter and the region (usually "# Title\n")
    region: str | None  # Jarvis's part (None: markers missing)
    after: str  # everything after the region — the user's


def parse_page(text: str) -> ParsedPage:
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    fm_lines: list[str] = []
    has_fm = False
    i = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                fm_lines, has_fm, i = lines[1:j], True, j + 1
                break
    rest = lines[i:]
    start = next((k for k, ln in enumerate(rest) if _START_RE.match(ln)), None)
    end = (
        next((k for k in range(start + 1, len(rest)) if _END_RE.match(rest[k])), None)
        if start is not None
        else None
    )
    fm = Frontmatter.parse(fm_lines)
    if start is None or end is None:
        return ParsedPage(fm, has_fm, "\n".join(rest), None, "")
    return ParsedPage(
        fm,
        has_fm,
        "\n".join(rest[:start]) + ("\n" if start else ""),
        "\n".join(rest[start + 1 : end]),
        "\n".join(rest[end + 1 :]),
    )


def build_page(
    existing: str | None,
    *,
    title: str,
    region: str,
    props: dict,
    user_section: str = MY_NOTES,
    append_after: str = "",
) -> str:
    """The page with Jarvis's part replaced; your frontmatter keys and notes kept.

    `append_after`: lines rescued from Jarvis's part (that it couldn't understand),
    put just below it so they're never lost.
    """
    region = region.strip("\n")
    extra = (append_after.strip("\n") + "\n") if append_after.strip() else ""
    if existing is None:
        fm = Frontmatter()
        for k, v in props.items():
            fm.set(k, v)
        tail = ("\n" + user_section) if user_section else ""
        return f"{fm.render()}# {title}\n{START}\n{region}\n{END}\n{extra}{tail}"
    page = parse_page(existing)
    fm = page.frontmatter
    for k, v in props.items():
        if k == "type" and fm.get("type") not in (None, ""):
            continue  # your own type wins
        fm.set(k, v)
    if page.region is None:
        # Markers were removed: put Jarvis's part back under the title, keep the rest.
        body = page.before
        m = re.match(r"^(\s*# [^\n]*\n)", body)
        head, rest = (m.group(1), body[m.end() :]) if m else (f"# {title}\n", body)
        # Drop Jarvis's old lines (they're re-written inside the markers).
        rest = "\n".join(ln for ln in rest.split("\n") if not _BLOCK_ID.search(ln))
        rest = rest.lstrip("\n")
        return f"{fm.render()}{head}{START}\n{region}\n{END}\n{extra}" + (
            "\n" + rest if rest else ("\n" + user_section if user_section else "")
        )
    before = page.before if page.before.strip() else f"# {title}\n"
    return f"{fm.render()}{before}{START}\n{region}\n{END}\n{extra}{page.after}"


JARVIS_KEYS = {"type", "jarvis", "created", "date", "mentions", "secret", "updated"}
_BOILERPLATE = ("## My notes", "## How to use")


def has_user_content(text: str) -> bool:
    """Did you write anything on this page (outside Jarvis's part)?"""
    page = parse_page(text)
    if any(k and k not in JARVIS_KEYS for k, _ in page.frontmatter.entries):
        return True
    before = re.sub(r"^\s*# [^\n]*\n?", "", page.before, count=1).strip()
    if before and page.region is not None:
        return True
    after = page.after if page.region is not None else page.before
    for marker in _BOILERPLATE:
        if marker == "## How to use" and marker in after:
            after = after.split(marker, 1)[0]
        after = after.replace(marker, "")
    return bool(after.strip())


# ---------------------------------------------------------------- reading the region


@dataclass
class Item:
    text: str
    section: str  # nearest ## heading inside the region ("" if none)
    kind: str | None = None  # "m" memory / "r" reminder / None (new line)
    id: int | None = None
    checked: bool | None = None  # task lines only


_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(?P<text>.*?)\s*$")
_ONLY_LINKS = re.compile(r"^(\[\[[^\]]+\]\][\s,·:–-]*)+$")
SKIP_SECTIONS = {"mentioned in"}


def parse_items(region: str, *, loose: bool = False, strays: list | None = None) -> list[Item]:
    """Lines in Jarvis's part of a page, with their ^ids and section.

    loose: plain lines and numbered items count too (memory pages: every line you
    write there is a fact). Otherwise only bullets do, and anything else is put in
    `strays` so the caller can keep it.
    """
    items: list[Item] = []
    section = ""
    for line in region.split("\n"):
        if h := _HEADING.match(line):
            section = h.group("title").strip()
            continue
        if not line.strip():
            continue
        if section.lower() in SKIP_SECTIONS:  # Jarvis's links; anything else is yours
            b = _BULLET.match(line)
            if not (b and _ONLY_LINKS.match(b.group("text").strip())) and strays is not None:
                strays.append(line)
            continue
        m = _BULLET.match(line) or _NUMBERED.match(line)
        if not m:
            if loose and not line.strip().startswith(("%%", "<!--", "|", "---")):
                m = re.match(r"^\s*(?P<text>.*?)\s*$", line)
            else:
                if strays is not None:
                    strays.append(line)
                continue
        text = m.group("text")
        kind = mid = None
        if b := _BLOCK_ID.search(text):
            kind, mid = b.group("kind"), int(b.group("id"))
            text = text[: b.start()].rstrip()
        text = text.strip()
        if not text or _ONLY_LINKS.match(text):  # just links, not a fact
            continue
        box = m.groupdict().get("box")
        items.append(
            Item(
                text=text,
                section=section,
                kind=kind,
                id=mid,
                checked=None if box is None else box.lower() == "x",
            )
        )
    return items
