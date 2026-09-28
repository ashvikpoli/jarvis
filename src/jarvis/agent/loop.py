"""AgentLoop: model ⇄ tools until the model answers without calling a tool."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from pydantic import ValidationError

from jarvis.core.types import Message, StreamChunk, ToolCall, ToolEvent, Usage
from jarvis.providers import ModelManager
from jarvis.tools.base import ToolContext, ToolError
from jarvis.tools.registry import (
    ApprovalRequest,
    Approver,
    Decision,
    PermissionPolicy,
    ToolAuditLog,
    ToolRegistry,
)

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 6000  # what the model sees of a tool result
TOOL_TIMEOUT_S = 30

# Called for every message the loop produces, so callers can persist them.
OnMessage = Callable[..., None]


def _noop(*_a, **_k) -> None:
    pass


def format_result(output: Any) -> str:
    text = output if isinstance(output, str) else json.dumps(output, default=str, indent=1)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + f"\n… (truncated, {len(text)} chars total)"
    return text


class AgentLoop:
    def __init__(
        self,
        models: ModelManager,
        registry: ToolRegistry,
        policy: PermissionPolicy,
        audit: ToolAuditLog | None = None,
        *,
        max_steps: int = 6,
    ):
        self.models = models
        self.registry = registry
        self.policy = policy
        self.audit = audit
        self.max_steps = max_steps

    async def run(
        self,
        messages: Sequence[Message],
        ctx: ToolContext,
        approver: Approver | None = None,
        *,
        role: str = "chat",
        options: dict[str, Any] | None = None,
        on_message: OnMessage = _noop,
        use_tools: bool = True,
    ) -> AsyncIterator[StreamChunk]:
        """Yield text/thinking chunks and tool events, then exactly one done chunk.

        `on_message(message, thinking=..., usage=..., interrupted=...)` receives each
        assistant/tool message as it's finalised. If the consumer stops early, the
        partial assistant text is reported with interrupted=True.
        """
        msgs = list(messages)
        specs = self.registry.specs(ctx) if use_tools and len(self.registry) else None
        total = Usage(prompt_tokens=0, completion_tokens=0, total_ms=0.0)
        started = time.perf_counter()
        first_usage: Usage | None = None
        completed = False

        for _step in range(self.max_steps):
            text: list[str] = []
            thinking: list[str] = []
            calls: list[ToolCall] = []
            usage: Usage | None = None
            step_done = False
            try:
                async for chunk in self.models.stream_chat(
                    msgs, role, options=options, tools=specs
                ):
                    if chunk.done:
                        usage = chunk.usage
                        continue
                    if chunk.tool_calls:
                        calls.extend(chunk.tool_calls)
                        continue
                    text.append(chunk.text)
                    thinking.append(chunk.thinking)
                    yield chunk
                step_done = True
            finally:
                if not step_done and "".join(text):
                    on_message(
                        Message.assistant("".join(text)),
                        thinking="".join(thinking),
                        interrupted=True,
                    )

            if usage:
                first_usage = first_usage or usage
                total.prompt_tokens += usage.prompt_tokens or 0
                total.completion_tokens += usage.completion_tokens or 0
                total.total_ms += usage.total_ms or 0  # model time only, not tool time

            assistant = Message.assistant("".join(text), calls)
            on_message(assistant, thinking="".join(thinking), usage=usage)
            msgs.append(assistant)
            if not calls:
                completed = True
                break

            for call in calls:
                yield StreamChunk(event=ToolEvent(kind="call", call=call))
                result, event = await self._execute(call, ctx, approver)
                yield StreamChunk(event=event)
                on_message(result)
                msgs.append(result)

        if not completed:
            note = f"\n(I stopped after {self.max_steps} tool steps.)"
            yield StreamChunk(text=note)
            on_message(Message.assistant(note.strip()))

        if first_usage:
            total.ttft_ms = first_usage.ttft_ms
            total.answer_ttft_ms = first_usage.answer_ttft_ms
        if not total.total_ms:
            total.total_ms = (time.perf_counter() - started) * 1000
        yield StreamChunk(done=True, usage=total)

    async def _execute(
        self, call: ToolCall, ctx: ToolContext, approver: Approver | None
    ) -> tuple[Message, ToolEvent]:
        def result(kind: str, text: str) -> tuple[Message, ToolEvent]:
            return (
                Message.tool(text, call.name, call.id),
                ToolEvent(kind=kind, call=call, output=text),
            )

        tool = self.registry.get(call.name)
        if tool is None:
            names = ", ".join(t.name for t in self.registry.tools)
            return result("error", f"Unknown tool {call.name!r}. Available: {names}")
        try:
            args = tool.Args.model_validate(call.arguments)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'args'}: {err['msg']}" for err in e.errors()
            )
            return result("error", f"Invalid arguments for {tool.name}: {problems}")

        decision, risk = self.policy.decide(tool, args)
        how = "allowed"
        if decision == Decision.ASK:
            if approver is None:
                decision, how = Decision.DENY, "denied"
                reason = "This action needs the user's approval, which isn't available here."
            else:
                answer = await approver(
                    ApprovalRequest(
                        tool=tool, args=args, risk=risk, description=tool.describe(args)
                    )
                )
                if answer == "session":
                    self.policy.allow_for_session(tool.name)
                    how = "approved_session"
                elif answer == "once":
                    how = "approved"
                else:
                    decision, how = Decision.DENY, "denied"
                    reason = (
                        "The user declined this action. Don't retry it; "
                        "ask what they'd like instead."
                    )
        elif decision == Decision.DENY:
            how = "denied"
            reason = "This action is disabled in Jarvis's settings."

        if decision == Decision.DENY:
            self._audit(call, ctx, risk, how, False, reason)
            return result("denied", reason)

        t0 = time.perf_counter()
        try:
            output = await asyncio.wait_for(tool.run(args, ctx), TOOL_TIMEOUT_S)
            ok, kind, text = True, "result", format_result(output)
        except ToolError as e:
            ok, kind, text = False, "error", str(e)
        except TimeoutError:
            ok, kind, text = False, "error", f"{tool.name} timed out after {TOOL_TIMEOUT_S}s"
        except Exception as e:  # a bug in a tool must not kill the conversation
            log.exception("tool %s crashed", tool.name)
            ok, kind, text = False, "error", f"{tool.name} failed unexpectedly: {e}"
        self._audit(call, ctx, risk, how, ok, text, (time.perf_counter() - t0) * 1000)
        return result(kind, text)

    def _audit(self, call, ctx, risk, decision, ok, output, duration_ms=None) -> None:
        if self.audit:
            self.audit.record(
                tool=call.name,
                arguments=call.arguments,
                risk=risk,
                decision=decision,
                ok=ok,
                output=output,
                duration_ms=duration_ms,
                conversation_id=ctx.conversation_id,
                call_id=call.id,
            )
