"""Eval cases and deterministic checks."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from jarvis.core.config import REPO_ROOT
from jarvis.core.types import Message, ToolCall

DEFAULT_CASES = REPO_ROOT / "evals" / "cases.toml"


class Checks(BaseModel):
    contains_all: list[str] = Field(default_factory=list)  # every phrase, case-insensitive
    contains_any: list[str] = Field(default_factory=list)  # at least one phrase
    not_contains: list[str] = Field(default_factory=list)  # none of these phrases
    regex: str | None = None  # must match (case-insensitive)
    max_words: int | None = None
    min_words: int | None = None
    number: float | None = None  # this number must appear (commas/decimals tolerated)
    json_keys: list[str] | None = None  # reply must contain a JSON object with these keys
    no_markdown: bool = False  # no **bold**, # headings, bullet lists, code fences
    # tool use (evals run with tools in a sandbox)
    tools_called: list[str] = Field(default_factory=list)  # every one of these was called
    tools_called_any: list[str] = Field(default_factory=list)  # at least one was called
    tools_not_called: list[str] = Field(default_factory=list)
    no_tools: bool = False  # answered without calling any tool
    # {"reminders_set": {"text": "stretch"}}: some call's arg contains the substring
    tool_args: dict[str, dict[str, str]] = Field(default_factory=dict)


class EvalCase(BaseModel):
    id: str
    category: str
    prompt: str
    history: list[Message] = Field(default_factory=list)
    checks: Checks = Field(default_factory=Checks)
    note: str = ""


_THINK_TAGS = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_MARKDOWN = re.compile(
    r"(\*\*[^*]+\*\*|^\s{0,3}#{1,6}\s|^\s*[-*+]\s+\S|^\s*\d+\.\s+\S|```)", re.MULTILINE
)


def clean_reply(text: str) -> str:
    """Strip inline reasoning some models emit even with thinking off."""
    return _THINK_TAGS.sub("", text).strip()


def _numbers_in(text: str) -> list[float]:
    out = []
    for raw in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            pass
    return out


def _find_json_object(text: str) -> dict[str, Any] | None:
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def _tool_failures(calls: list[ToolCall], checks: Checks) -> list[str]:
    fails = []
    names = [c.name for c in calls]
    if missing := [t for t in checks.tools_called if t not in names]:
        fails.append(f"didn't call {missing} (called {names or 'nothing'})")
    if checks.tools_called_any and not set(checks.tools_called_any) & set(names):
        fails.append(f"called none of {checks.tools_called_any} (called {names or 'nothing'})")
    if wrong := [t for t in checks.tools_not_called if t in names]:
        fails.append(f"shouldn't have called {wrong}")
    if checks.no_tools and names:
        fails.append(f"used tools unnecessarily: {names}")
    for tool, expected in checks.tool_args.items():
        candidates = [c for c in calls if c.name == tool]
        ok = any(
            all(
                sub.lower() in str(c.arguments.get(key, "")).lower()
                for key, sub in expected.items()
            )
            for c in candidates
        )
        if candidates and not ok:
            fails.append(f"{tool} args {[c.arguments for c in candidates]} don't match {expected}")
    return fails


def run_checks(reply: str, checks: Checks, tool_calls: list[ToolCall] | None = None) -> list[str]:
    """Return a list of failure reasons (empty = pass)."""
    text = clean_reply(reply)
    low = text.lower()
    words = len(text.split())
    fails: list[str] = _tool_failures(tool_calls or [], checks)

    if not text:
        return [*fails, "empty reply"]
    missing = [p for p in checks.contains_all if p.lower() not in low]
    if missing:
        fails.append(f"missing {missing}")
    if checks.contains_any and not any(p.lower() in low for p in checks.contains_any):
        fails.append(f"none of {checks.contains_any}")
    present = [p for p in checks.not_contains if p.lower() in low]
    if present:
        fails.append(f"should not contain {present}")
    if checks.regex and not re.search(checks.regex, text, re.IGNORECASE | re.MULTILINE):
        fails.append(f"no match for /{checks.regex}/")
    if checks.max_words is not None and words > checks.max_words:
        fails.append(f"{words} words > {checks.max_words}")
    if checks.min_words is not None and words < checks.min_words:
        fails.append(f"{words} words < {checks.min_words}")
    if checks.number is not None and not any(
        abs(n - checks.number) < 1e-6 for n in _numbers_in(text)
    ):
        fails.append(f"number {checks.number:g} not found")
    if checks.json_keys is not None:
        obj = _find_json_object(text)
        if obj is None:
            fails.append("no valid JSON object")
        elif missing_keys := [k for k in checks.json_keys if k not in obj]:
            fails.append(f"JSON missing keys {missing_keys}")
    if checks.no_markdown and _MARKDOWN.search(text):
        fails.append("contains markdown")
    return fails


def load_cases(path: Path | str | None = None) -> list[EvalCase]:
    path = Path(path) if path else DEFAULT_CASES
    with path.open("rb") as f:
        data = tomllib.load(f)
    cases = [EvalCase(**c) for c in data.get("case", [])]
    ids = [c.id for c in cases]
    if dupes := {i for i in ids if ids.count(i) > 1}:
        raise ValueError(f"Duplicate eval case ids: {sorted(dupes)}")
    return cases
