"""Ask a vision model about a screenshot, or where something is on it."""

from __future__ import annotations

import json
import logging
import re

from jarvis.core.types import Message
from jarvis.providers import ModelManager

log = logging.getLogger(__name__)

LOOK_SYSTEM = (
    "You are looking at a screenshot of the user's Windows PC. Answer the question about "
    "it accurately and concisely, reading any text carefully. Say what app or page it is "
    "when that helps. If you can't make something out, say so — never guess at text."
)

LOCATE_SYSTEM = (
    "You find things on screenshots of a Windows PC. Given a description, find the single "
    "best matching on-screen element (button, link, field, icon, menu item, text) and give "
    "the CENTER of it as x and y in thousandths of the image width and height (0-1000, "
    "origin top-left). If nothing on the screenshot matches, set found to false."
)

LOCATE_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "what": {"type": "string"},
    },
    "required": ["found"],
}


class Vision:
    """Wraps the `vision` model role (e.g. ollama:qwen3-vl:8b-instruct)."""

    def __init__(self, models: ModelManager, role: str = "vision"):
        self.models = models
        self.role = role

    @property
    def model_ref(self) -> str | None:
        ref = self.models.roles.get(self.role)
        return str(ref) if ref else None

    async def _ask(self, messages: list[Message], options: dict) -> str:
        parts: list[str] = []
        async for chunk in self.models.stream_chat(messages, self.role, options=options):
            parts.append(chunk.text)
        return "".join(parts).strip()

    async def look(self, image_b64: str, question: str) -> str:
        msgs = [
            Message.system(LOOK_SYSTEM),
            Message(role="user", content=question or "What's on the screen?", images=[image_b64]),
        ]
        text = await self._ask(msgs, {"think": False, "temperature": 0.2})
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        if "</think>" in text:  # a thinking-only model leaked its reasoning
            text = text.rsplit("</think>", 1)[1]
        return text.strip()

    async def locate(self, image_b64: str, description: str) -> tuple[float, float, str] | None:
        """(x, y) as fractions of the image (0..1) and what it found — or None."""
        msgs = [
            Message.system(LOCATE_SYSTEM),
            Message(role="user", content=f"Find: {description}", images=[image_b64]),
        ]
        raw = await self._ask(msgs, {"think": False, "temperature": 0, "format": LOCATE_SCHEMA})
        try:
            data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        except (json.JSONDecodeError, ValueError):
            log.info("vision locate: unreadable answer %r", raw[:200])
            return None
        if not data.get("found") or "x" not in data or "y" not in data:
            return None
        x, y = float(data["x"]), float(data["y"])
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            return None
        return x / 1000, y / 1000, str(data.get("what") or description)
