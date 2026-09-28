"""Tiny vector helpers — no numpy needed at personal-memory scale (hundreds/thousands)."""

from __future__ import annotations

import math
import re
import zlib
from array import array
from collections.abc import Sequence

STOPWORDS = set(
    """a an the and or but is are was were be been am i me my mine you your we our us it its
    to of in on at for with from by as that this these those do does did have has had what
    who where when which how why can could would should will about user user's s so if not
    no yes please tell know any there""".split()
)


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOPWORDS]


def normalize(vec: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; vectors from `normalize` make this a plain dot product."""
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))


def pack(vec: Sequence[float]) -> bytes:
    return array("f", vec).tobytes()


def unpack(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    a = array("f")
    a.frombytes(blob)
    return a.tolist()


def keyword_score(query: str, text: str) -> float:
    """Share of the query's content words found in text (0..1). Fallback ranking."""
    q = set(words(query))
    if not q:
        return 0.0
    t = set(words(text))
    # match simple plurals/possessives too: "sisters" ~ "sister"
    stem = lambda w: w.rstrip("s").rstrip("'")  # noqa: E731
    ts = {stem(w) for w in t}
    hits = sum(1 for w in q if w in t or stem(w) in ts)
    return hits / len(q)


def toy_embed(text: str, dim: int = 64) -> list[float]:
    """Deterministic bag-of-words vector for tests and the offline echo provider."""
    vec = [0.0] * dim
    for w in words(text):
        vec[zlib.crc32(w.rstrip("s").encode()) % dim] += 1.0
    return normalize(vec) if any(vec) else vec
