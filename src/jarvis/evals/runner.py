"""Run eval cases against one or more models and report."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from contextlib import aclosing
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

from jarvis.core.types import Message, ToolCall
from jarvis.evals.cases import EvalCase, clean_reply, run_checks, run_memory_checks
from jarvis.evals.sandbox import SANDBOX_MEMORIES, make_sandbox
from jarvis.memory.extractor import MemoryExtractor
from jarvis.memory.service import MemoryService
from jarvis.memory.store import MemoryStore
from jarvis.providers import ProviderError

if TYPE_CHECKING:
    from jarvis.app import JarvisApp

EVAL_ROLE = "__eval__"


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    failures: list[str]
    reply: str
    answer_ttft_ms: float | None = None
    tokens_per_s: float | None = None
    total_ms: float | None = None
    error: str | None = None
    tool_calls: list[str] = field(default_factory=list)  # "name {args}" for the report


@dataclass
class ModelResult:
    model: str
    options: dict[str, Any]
    started_at: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(c.passed for c in self.cases)

    @property
    def pass_rate(self) -> float:
        return self.passed / len(self.cases) if self.cases else 0.0

    def by_category(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for c in self.cases:
            out[c.category][0] += c.passed
            out[c.category][1] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    def _median(self, attr: str) -> float | None:
        vals = [getattr(c, attr) for c in self.cases if getattr(c, attr)]
        return statistics.median(vals) if vals else None

    @property
    def median_answer_ttft_ms(self) -> float | None:
        return self._median("answer_ttft_ms")

    @property
    def median_tokens_per_s(self) -> float | None:
        return self._median("tokens_per_s")


class EvalRunner:
    def __init__(self, app: JarvisApp, console: Console | None = None):
        self.app = app
        self.console = console or Console(quiet=True)

    async def _sandbox_memory(self, box, case: EvalCase) -> MemoryService:
        """Memory for the sandbox: its own store, the app's embedding model."""
        m = self.app.memory
        svc = MemoryService(
            MemoryStore(box.ctx.db),
            m.embedder,
            top_k=m.top_k,
            min_score=m.min_score,
            core_limit=m.core_limit,
            dedupe_threshold=m.dedupe_threshold,
        )
        seeds = (
            SANDBOX_MEMORIES
            if case.memories == "default"
            else [s.model_dump() for s in case.memories]
        )
        for seed in seeds:
            await svc.save(seed["text"], category=seed.get("category", "other"))
        box.ctx.memory = svc
        return svc

    async def run_extract_case(self, case: EvalCase) -> CaseResult:
        """Test only the learner: what would Jarvis remember from this message?"""
        box = make_sandbox()
        try:
            svc = await self._sandbox_memory(box, case)
            extractor = MemoryExtractor(self.app.models, svc, role=EVAL_ROLE)
            changes = await extractor.extract(case.prompt, case.history)
        except ProviderError as e:
            return CaseResult(case.id, case.category, False, ["error"], "", error=str(e))
        finally:
            box.close()
        failures = run_memory_checks(changes, case.checks)
        summary = "; ".join(f"{c.op}: {c.text}" for c in changes) or "(learned nothing)"
        return CaseResult(
            case.id,
            case.category,
            not failures,
            failures,
            summary,
            tool_calls=[f"{c.op} #{c.id}" for c in changes],
        )

    async def run_case(self, case: EvalCase, options: dict[str, Any] | None) -> CaseResult:
        """Run one case with tools available, inside a throwaway sandbox."""
        if case.kind == "extract":
            return await self.run_extract_case(case)
        messages = [*case.history, Message.user(case.prompt)]
        parts: list[str] = []
        calls: list[ToolCall] = []
        usage = None
        box = make_sandbox()
        try:
            await self._sandbox_memory(box, case)
            stream = self.app.agent_stream(
                messages, box.ctx, box.approver, role=EVAL_ROLE, options=options
            )
            async with aclosing(stream) as s:
                async for chunk in s:
                    parts.append(chunk.text)
                    if chunk.event and chunk.event.kind == "call":
                        calls.append(chunk.event.call)
                    usage = chunk.usage or usage
        except ProviderError as e:
            return CaseResult(case.id, case.category, False, ["error"], "", error=str(e))
        finally:
            opened = list(getattr(box.ctx.launcher, "opened", []))
            box.close()
        reply = clean_reply("".join(parts))
        failures = run_checks(reply, case.checks, calls)
        want = case.checks.opened_any
        if want and not any(w.lower() in o.lower() for o in opened for w in want):
            failures.append(f"opened {opened or 'nothing'}, wanted one of {want}")
        return CaseResult(
            case.id,
            case.category,
            not failures,
            failures,
            reply,
            answer_ttft_ms=usage.answer_ttft_ms if usage else None,
            tokens_per_s=usage.tokens_per_s if usage else None,
            total_ms=usage.total_ms if usage else None,
            tool_calls=[f"{c.name} {json.dumps(c.arguments, default=str)}" for c in calls],
        )

    async def run_model(
        self, ref: str, cases: list[EvalCase], *, options: dict[str, Any] | None = None
    ) -> ModelResult:
        models = self.app.models
        models.set_role(EVAL_ROLE, ref)
        # Same generation settings as real chat, plus any overrides for this run.
        models.role_options[EVAL_ROLE] = {**models.role_options.get("chat", {}), **(options or {})}
        result = ModelResult(
            model=ref,
            options=models.options_for(EVAL_ROLE),
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        with self.console.status(f"Loading {ref}..."):
            try:
                await models.warmup(EVAL_ROLE)
            except ProviderError as e:
                result.cases = [
                    CaseResult(c.id, c.category, False, ["error"], "", error=str(e)) for c in cases
                ]
                self.console.print(f"[red]✗ {ref}: {e}[/]")
                return result

        for i, case in enumerate(cases, 1):
            with self.console.status(f"{ref}  [{i}/{len(cases)}] {case.id}"):
                r = await self.run_case(case, None)
            result.cases.append(r)
            mark = "[green]✓[/]" if r.passed else "[red]✗[/]"
            self.console.print(f"  {mark} {case.id}", highlight=False)

        # Free VRAM before the next model so runs don't skew each other — unless this
        # is the model you chat with, which you'll want to stay loaded.
        if ref != str(models.roles.get("chat")):
            try:
                await models.unload(EVAL_ROLE)
            except ProviderError:
                pass
        return result


def print_report(results: list[ModelResult], console: Console, *, verbose: bool = False) -> None:
    # Summary: one row per model. Kept narrow so it fits a normal terminal.
    summary = Table(title="Eval results", header_style="bold")
    summary.add_column("Model", no_wrap=True)
    summary.add_column("Pass", justify="right", no_wrap=True)
    summary.add_column("Answer ms", justify="right")
    summary.add_column("tok/s", justify="right")
    for r in results:
        ttft, tps = r.median_answer_ttft_ms, r.median_tokens_per_s
        summary.add_row(
            r.model,
            f"{r.passed}/{len(r.cases)} ({r.pass_rate:.0%})",
            f"{ttft:.0f}" if ttft else "-",
            f"{tps:.0f}" if tps else "-",
        )
    console.print(summary)

    # Breakdown: categories as rows, models as columns (scales with more categories).
    categories = sorted({c for r in results for c in r.by_category()})
    breakdown = Table(title="By category", header_style="bold")
    breakdown.add_column("Category", no_wrap=True)
    for r in results:
        breakdown.add_column(r.model.split(":", 1)[-1], justify="right", no_wrap=True)
    for cat in categories:
        cells = []
        for r in results:
            p, t = r.by_category().get(cat, (0, 0))
            style = "green" if t and p == t else ("red" if t and p == 0 else "yellow")
            cells.append(f"[{style}]{p}/{t}[/]" if t else "-")
        breakdown.add_row(cat, *cells)
    console.print(breakdown)

    for r in results:
        failed = [c for c in r.cases if not c.passed]
        if not failed:
            continue
        console.print(f"\n[bold]{r.model}[/] — {len(failed)} failed")
        for c in failed if verbose else failed[:10]:
            reason = c.error or "; ".join(c.failures)
            console.print(f"  [red]✗[/] {c.case_id}: {reason}", highlight=False)
            if verbose:
                for t in c.tool_calls:
                    console.print(f"    [dim]⚙ {t[:160]}[/]", highlight=False)
                if c.reply:
                    snippet = " ".join(c.reply.split())[:200]
                    console.print(f"    [dim]{snippet}[/]", highlight=False)
        if not verbose and len(failed) > 10:
            console.print(f"  [dim]... {len(failed) - 10} more (use -v)[/]")


def save_results(results: list[ModelResult], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    payload = [
        {
            **{k: v for k, v in asdict(r).items() if k != "cases"},
            "pass_rate": r.pass_rate,
            "by_category": r.by_category(),
            "median_answer_ttft_ms": r.median_answer_ttft_ms,
            "median_tokens_per_s": r.median_tokens_per_s,
            "cases": [asdict(c) for c in r.cases],
        }
        for r in results
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
