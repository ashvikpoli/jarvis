"""PromptBuilder: system prompt + context + as much recent history as fits."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from jarvis.core.types import Message

PER_MESSAGE_OVERHEAD = 4  # role markers / separators the chat template adds


def estimate_tokens(text: str) -> int:
    """Cheap, deliberately conservative estimate (~3.5 chars/token for English).

    Real tokenizers vary; over-estimating slightly just means we trim a bit early,
    which is much better than overflowing the context window.
    """
    return int(len(text) / 3.5) + 1


@dataclass
class BuiltPrompt:
    messages: list[Message]
    dropped: int  # history messages that didn't fit
    est_tokens: int


class PromptBuilder:
    def __init__(
        self,
        system_prompt: str,
        *,
        context_tokens: int = 8192,
        reserve_tokens: int = 1024,
        include_datetime: bool = True,
        estimator: Callable[[str], int] = estimate_tokens,
        clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ):
        self.system_prompt = system_prompt
        self.context_tokens = context_tokens
        self.reserve_tokens = reserve_tokens
        self.include_datetime = include_datetime
        self.estimator = estimator
        self.clock = clock

    def _cost(self, m: Message) -> int:
        cost = self.estimator(m.content) + PER_MESSAGE_OVERHEAD
        for call in m.tool_calls:
            cost += self.estimator(call.name + json.dumps(call.arguments)) + PER_MESSAGE_OVERHEAD
        return cost

    def tokens_for_tools(self, specs: list[dict] | None) -> int:
        return self.estimator(json.dumps(specs)) if specs else 0

    def system_message(self, dropped: int = 0) -> Message:
        parts = [self.system_prompt.strip()]
        if self.include_datetime:
            # Date only: the prompt prefix then stays identical all day, so Ollama can
            # reuse its cache instead of re-reading the prompt + tool definitions every
            # minute. Exact time comes from the system_time tool.
            today = self.clock().strftime("%A, %d %B %Y")
            parts.append(f"Today is {today}. For the exact current time, use system_time.")
        if dropped:
            parts.append(
                f"({dropped} earlier messages in this conversation are not shown due to length.)"
            )
        return Message.system("\n\n".join(parts))

    def build(self, history: Sequence[Message], *, extra_tokens: int = 0) -> BuiltPrompt:
        """Fit history into the budget, newest first. The last message is always kept.

        A leading system message in `history` replaces the configured system prompt
        (used by the stateless API when callers bring their own). `extra_tokens` is
        space taken by things outside the messages, e.g. tool definitions.
        """
        history = list(history)
        system_override = None
        if history and history[0].role == "system":
            system_override = history.pop(0)

        # Reserve a little for the "N messages omitted" note in case we need it.
        budget = self.context_tokens - self.reserve_tokens - extra_tokens - 30
        system = system_override or self.system_message()
        budget -= self._cost(system)

        kept: list[Message] = []
        for i, m in enumerate(reversed(history)):
            cost = self._cost(m)
            if i > 0 and cost > budget:
                break
            kept.append(m)
            budget -= cost
        kept.reverse()
        # Never start with tool results whose tool call was trimmed away.
        while len(kept) > 1 and kept[0].role == "tool":
            kept.pop(0)
        dropped = len(history) - len(kept)

        if dropped and system_override is None:
            system = self.system_message(dropped)
        messages = [system, *kept]
        return BuiltPrompt(
            messages=messages,
            dropped=dropped,
            est_tokens=sum(self._cost(m) for m in messages) + extra_tokens,
        )
