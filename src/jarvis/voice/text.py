"""Text helpers for voice: what to say out loud, when a sentence is complete, yes/no."""

from __future__ import annotations

import re
from typing import Literal

# ---------------------------------------------------------------- speakable text

_CODE_BLOCK = re.compile(r"```.*?(```|$)", re.S)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://(?:www\.)?([^/\s)]+)[^\s)]*")
_WIN_PATH = re.compile(r"\b[A-Za-z]:\\(?:[^\\\s\"'<>|]+\\)*([^\\\s\"'<>|]+)")
_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC = re.compile(r"(?<![\w*])[*_]([^*_\n]+)[*_](?![\w*])")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
_QUOTE = re.compile(r"^\s*>\s?", re.M)
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}.*$", re.M)
# Emoji & pictographs, dingbats, arrows, box drawing, misc symbols.
_SYMBOLS = re.compile(
    "[\U0001f000-\U0001faff\u2600-\u27bf\u2190-\u21ff\u2500-\u259f\u2b00-\u2bff"
    "\ufe0f\u200d\u2300-\u23ff]"
)
_SPACES = re.compile(r"[ \t]+")
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\b")


def _clock(m: re.Match) -> str:
    h, mm = m.group(1), m.group(2)
    if mm == "00":
        return h
    return f"{h} oh {int(mm)}" if mm.startswith("0") else f"{h} {mm}"


def speech_text(text: str) -> str:
    """Turn a chat reply into something a TTS voice can read naturally."""
    t = _CODE_BLOCK.sub(" (the code is on screen) ", text)
    t = _INLINE_CODE.sub(r"\1", t)
    t = _MD_LINK.sub(r"\1", t)
    t = _URL.sub(r"\1", t)
    t = _CLOCK.sub(_clock, t)  # 3:45 -> 3 45, 9:05 -> 9 oh 5, 3:00 -> 3
    t = _WIN_PATH.sub(r"\1", t)  # C:\Users\me\Desktop\notes.txt -> notes.txt
    t = _BOLD.sub(r"\2", t)
    t = _ITALIC.sub(r"\1", t)
    t = _TABLE_RULE.sub("", t)
    t = _HEADING.sub("", t)
    t = _QUOTE.sub("", t)
    # A bullet list read aloud needs pauses between items.
    t = _BULLET.sub("\n", t)
    t = t.replace("|", ", ").replace("&", " and ").replace("°C", " degrees Celsius")
    t = t.replace("°F", " degrees Fahrenheit").replace("→", " to ").replace("×", " times ")
    t = _SYMBOLS.sub("", t)
    lines = [_SPACES.sub(" ", ln).strip() for ln in t.splitlines()]
    out: list[str] = []
    for ln in lines:
        if not ln:
            continue
        if out and not out[-1].endswith((".", "!", "?", ":", ";", ",")):
            out[-1] += "."
        out.append(ln)
    return " ".join(out).strip()


# ---------------------------------------------------------------- sentence splitting

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "approx",
    "fig", "inc", "ltd", "co", "mt", "a.m", "p.m",
}  # fmt: skip
_END = re.compile(r"[.!?…]+[\"')\]]*(?=\s)|\n+")


class SentenceSplitter:
    """Feed streamed text, get back sentences as soon as they're complete.

    The first chunk may be cut at a comma once it's long enough, so Jarvis starts
    speaking sooner; later chunks are whole sentences (long ones are cut at commas).
    """

    def __init__(self, *, first_min: int = 24, min_chars: int = 3, max_chars: int = 220):
        self.first_min = first_min
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.buf = ""
        self.emitted = 0

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while (s := self._next()) is not None:
            out.append(s)
        return out

    def flush(self) -> list[str]:
        rest = self.buf.strip()
        self.buf = ""
        if rest:
            self.emitted += 1
            return [rest]
        return []

    def _take(self, end: int) -> str | None:
        piece, self.buf = self.buf[:end].strip(), self.buf[end:]
        if not piece:
            return None
        self.emitted += 1
        return piece

    def _next(self) -> str | None:
        # Code blocks are held back until closed, then go out as one piece (which
        # speech_text turns into "the code is on screen") — never read line by line.
        fence = self.buf.find("```")
        if fence == 0:
            close = self.buf.find("```", 3)
            return None if close == -1 else self._take(close + 3)
        if fence > 0:
            return self._take(fence)
        start = 0
        while m := _END.search(self.buf, start):
            end = m.end()
            candidate = self.buf[:end].strip()
            start = end
            if "\n" not in m.group() and self._is_abbreviation(candidate):
                continue
            if len(candidate) < self.min_chars and "\n" not in m.group():
                continue
            return self._take(end)
        # No sentence end yet: cut early at a comma for the first chunk, or when long.
        limit = self.first_min if self.emitted == 0 else self.max_chars
        if len(self.buf) >= limit:
            cut = max(self.buf.rfind(", "), self.buf.rfind("; "), self.buf.rfind(": "))
            if cut >= 10 and (self.emitted == 0 or len(self.buf) >= self.max_chars):
                return self._take(cut + 1)
            if len(self.buf) >= self.max_chars * 1.5:  # no punctuation at all: cut at a space
                sp = self.buf.rfind(" ", 0, self.max_chars)
                if sp > 0:
                    return self._take(sp)
        return None

    @staticmethod
    def _is_abbreviation(candidate: str) -> bool:
        words = candidate.rstrip(".!?…\"')]").rsplit(None, 1)
        if not words or not candidate.rstrip("\"')]").endswith("."):
            return False
        last = words[-1].lower()
        # "U.S." / single initials "J." / known abbreviations
        return last in _ABBREV or (len(last) == 1 and last.isalpha()) or "." in last


# ---------------------------------------------------------------- yes / no

Answer = Literal["once", "session", "deny"]

_YES = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok(ay)?|go ahead|do it|please do|proceed|confirm(ed)?|"
    r"approve[d]?|allow(ed)?|affirmative|correct|absolutely|of course|fine|alright|all right)\b"
)
_NO = re.compile(
    r"\b(no|nope|nah|don'?t|do not|stop|cancel|never ?mind|deny|denied|negative|wait|hold on|"
    r"not now|abort)\b"
)
# "not sure", "not okay", "not correct" must never count as a yes.
_NEGATED_YES = re.compile(
    r"\b(not|never|n't)\s+(\w+\s+)?(sure|ok(ay)?|fine|correct|right|allowed|good|yes)\b"
    r"|n't\s+(\w+\s+)?(sure|ok(ay)?|fine|correct|right|allowed|good)\b"
)
_UNSURE = re.compile(r"\b(not sure|unsure|maybe|perhaps|i don'?t know|no idea|hmm+)\b")
_ALWAYS = re.compile(r"\b(always|every time|for this session|from now on|don'?t ask)\b")


def parse_approval(text: str, *, allow_session: bool = True) -> Answer | None:
    """Understand a spoken or typed answer to "shall I …?". None = unclear."""
    t = text.lower().strip().replace("’", "'")
    t = " ".join(re.sub(r"[^\w' ]+", " ", t).split())
    if t in ("y", "a"):
        return "session" if t == "a" and allow_session else "once"
    if t == "n":
        return "deny"
    if allow_session and _ALWAYS.search(t) and not re.search(r"\b(no|nope|don'?t allow)\b", t):
        return "session"
    if _UNSURE.search(t):
        return None
    if _NEGATED_YES.search(t):
        return "deny"
    no, yes = _NO.search(t), _YES.search(t)
    if no and not yes:
        return "deny"
    if yes and not no:
        return "once"
    if yes and no:  # "no, wait — yes do it" → whichever came last
        return "once" if yes.start() > no.start() else "deny"
    return None
