"""Command line interface. Run `jarvis --help` for the list of commands."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from contextlib import aclosing
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from jarvis import __version__
from jarvis.app import JarvisApp
from jarvis.conversation import ConversationNotFound
from jarvis.core.config import Settings, load_settings
from jarvis.core.logging import setup_logging
from jarvis.core.types import Message, Usage
from jarvis.providers import ProviderError
from jarvis.storage import Conversation, StoredMessage
from jarvis.tools import ApprovalAnswer, ApprovalRequest, Risk, reminder_rows
from jarvis.tools.notify import CallbackNotifier

console = Console()

HELP = """[bold]Commands[/]
  /new             start a new conversation
  /list            recent conversations
  /resume ID       continue an earlier conversation
  /title TEXT      rename this conversation
  /think on|off    let the model reason before answering (slower, sometimes smarter)
  /memory          what Jarvis remembers about you   (/memory off|on for this session)
  /remember TEXT   tell Jarvis something to remember
  /forget ID       delete a memory
  /tools           what Jarvis can do, and what needs your OK
  /reminders       upcoming reminders
  /model [REF]     show or switch the chat model, e.g. /model ollama:qwen3:8b
  /help            show this
  /exit            quit
[dim]Ctrl+C while Jarvis is talking interrupts it. Ctrl+C at the prompt quits.[/]"""


# ---------------------------------------------------------------- helpers


def _stats(usage: Usage | None) -> str:
    if not usage:
        return ""
    bits = []
    if usage.answer_ttft_ms is not None:
        bits.append(f"answer in {usage.answer_ttft_ms:.0f} ms")
        if usage.ttft_ms is not None and usage.answer_ttft_ms - usage.ttft_ms > 50:
            bits.append(f"thought for {(usage.answer_ttft_ms - usage.ttft_ms) / 1000:.1f} s")
    elif usage.ttft_ms is not None:
        bits.append(f"first token {usage.ttft_ms:.0f} ms")
    if usage.tokens_per_s:
        bits.append(f"{usage.tokens_per_s:.0f} tok/s")
    return " · ".join(bits)


def _conversation_table(convs: list[Conversation]) -> Table:
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("ID", justify="right")
    table.add_column("Title")
    table.add_column("Msgs", justify="right")
    table.add_column("Updated")
    for c in convs:
        table.add_row(
            str(c.id),
            c.title or "[dim](untitled)[/]",
            str(c.message_count),
            c.updated_at[:16].replace("T", " "),
        )
    return table


def _print_message(m: StoredMessage) -> None:
    if m.role == "user":
        console.print(f"[bold cyan]you ›[/] {m.content}")
    elif m.role == "assistant":
        for c in m.tool_calls:
            console.print(f"[dim]  ⚙ {_call_str(c)}[/]", highlight=False)
        if m.content:
            console.print("[bold green]jarvis ›[/] ", end="")
            console.out(m.content, highlight=False)
        if m.interrupted:
            console.print("[dim]⏹ interrupted[/]")
    elif m.role == "tool":
        console.print(f"[dim]    ↳ {_first_line(m.content)}[/]", highlight=False)


def _first_line(text: str, limit: int = 110) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _call_str(call) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
    return _first_line(f"{call.name}({args})")


async def cli_approver(req: ApprovalRequest) -> ApprovalAnswer:
    """Ask at the terminal. Blocking input on purpose: nothing else should run meanwhile."""
    color = "red" if req.risk == Risk.DESTRUCTIVE else "yellow"
    options = r"\[y]es / \[n]o" + (
        r" / \[a]lways this session" if req.can_allow_for_session else ""
    )
    console.print(f"\n[{color}]⚠ Jarvis wants to {req.description}[/]", highlight=False)
    while True:
        answer = console.input(f"  Allow? {options} › ").strip().lower()
        if answer in ("y", "yes"):
            return "once"
        if answer in ("n", "no", ""):
            return "deny"
        if answer in ("a", "always") and req.can_allow_for_session:
            return "session"


# ---------------------------------------------------------------- chat


async def _stream_turn(
    app: JarvisApp, conversation_id: int, text: str, show_thinking: bool
) -> Usage | None:
    usage = None
    state = {"mid_line": False, "need_prefix": True}

    def newline() -> None:
        if state["mid_line"]:
            console.print()
            state["mid_line"] = False

    # aclosing(): if we're interrupted, the partial reply is saved right away.
    stream = app.conversations.send(conversation_id, text, approver=cli_approver)
    async with aclosing(stream) as chunks:
        async for chunk in chunks:
            if chunk.thinking and show_thinking:
                if not state["mid_line"]:
                    console.print("[dim]💭 [/]", end="")
                console.print(Text(chunk.thinking, style="dim italic"), end="")
                state["mid_line"] = True
            if chunk.event:
                ev = chunk.event
                newline()
                if ev.kind == "call":
                    console.print(f"[dim]  ⚙ {_call_str(ev.call)}[/]", highlight=False)
                elif ev.kind == "result":
                    console.print(f"[dim]    ✓ {_first_line(ev.output)}[/]", highlight=False)
                elif ev.kind == "denied":
                    console.print(f"[yellow]    ✗ {_first_line(ev.output)}[/]", highlight=False)
                else:
                    console.print(f"[red]    ✗ {_first_line(ev.output)}[/]", highlight=False)
                state["need_prefix"] = True
            if chunk.text:
                if state["need_prefix"]:
                    console.print("[bold green]jarvis ›[/] ", end="")
                    state["need_prefix"] = False
                console.out(chunk.text, end="", highlight=False)
                state["mid_line"] = True
            if chunk.done:
                usage = chunk.usage
    newline()
    return usage


def cmd_chat(settings: Settings, args: argparse.Namespace) -> int:
    app = JarvisApp.create(settings)
    if args.echo:
        app.models.set_role("chat", "echo:echo")
    elif args.model:
        app.models.set_role("chat", args.model)

    show_thinking = args.show_thinking
    if show_thinking:
        app.models.set_option("chat", "think", True)

    conv: Conversation | None = None
    if args.resume:
        try:
            conv = (
                app.store.latest()
                if args.resume == "last"
                else app.conversations.get(int(args.resume))
            )
        except (ConversationNotFound, ValueError) as e:
            console.print(f"[red]✗ {e}[/]")
            return 1

    console.print(
        f"[bold]{settings.app.name}[/] v{__version__} · model [cyan]{app.models.roles['chat']}[/]"
        "  [dim](/help for commands)[/]"
    )
    if conv:
        console.print(f"[dim]Resuming #{conv.id}: {conv.title or '(untitled)'}[/]")
        for m in app.store.messages(conv.id)[-6:]:
            _print_message(m)

    # asyncio.Runner keeps one event loop alive across turns, and turns Ctrl+C into
    # cancellation of the running turn only — so interrupting doesn't exit the app.
    with asyncio.Runner() as runner:
        try:
            with console.status("Loading model into VRAM..."):
                runner.run(app.models.warmup("chat"))
        except ProviderError as e:
            console.print(f"[red]✗ {e}[/]\n[dim]Run `jarvis doctor` to diagnose.[/]")
            runner.run(app.aclose())
            return 1

        app.notifier.add(
            CallbackNotifier(
                lambda title, msg: console.print(f"\n[bold magenta]⏰ Reminder: {msg}[/]")
            )
        )
        app.scheduler.start()
        app.apps.prefetch()  # so the first "open X" doesn't wait for the app scan
        try:
            runner.run(app.memory.backfill())  # also loads the embedding model
        except ProviderError:
            pass
        app.memory_listeners.append(_print_memory_changes)

        while True:
            try:
                text = console.input("\n[bold cyan]you ›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue

            if text.startswith("/"):
                cmd, _, arg = text.partition(" ")
                arg = arg.strip()
                if cmd in ("/exit", "/quit"):
                    break
                elif cmd in ("/new", "/reset"):
                    conv = None
                    console.print("[dim]New conversation.[/]")
                elif cmd == "/list":
                    console.print(_conversation_table(app.store.list(15)))
                elif cmd == "/resume":
                    try:
                        conv = app.conversations.get(int(arg))
                        console.print(f"[dim]Resuming #{conv.id}: {conv.title or ''}[/]")
                        for m in app.store.messages(conv.id)[-6:]:
                            _print_message(m)
                    except (ConversationNotFound, ValueError):
                        console.print("[red]Usage: /resume ID  (see /list)[/]")
                elif cmd == "/title":
                    if conv and arg:
                        app.store.rename(conv.id, arg)
                        console.print("[dim]Renamed.[/]")
                    else:
                        console.print("[dim]Say something first, then /title TEXT[/]")
                elif cmd == "/memory":
                    if arg in ("off", "on"):
                        app.memory.enabled = arg == "on"
                        console.print(f"[dim]Memory {arg} for this session.[/]")
                    else:
                        _print_memories(app)
                elif cmd == "/remember":
                    if not arg:
                        console.print("[dim]Usage: /remember TEXT[/]")
                    else:
                        with console.status("Remembering..."):
                            changes = runner.run(app.remember(arg))
                        _print_memory_changes(changes)
                elif cmd == "/forget":
                    try:
                        mid = int(arg.lstrip("#"))
                        gone = app.memory_store.get(mid)
                        if gone and app.memory.forget(mid):
                            console.print(
                                f"[magenta]🗑 Forgot #{mid}: {gone.text}[/]", highlight=False
                            )
                        else:
                            console.print(f"[red]No memory #{mid}.[/]")
                    except ValueError:
                        console.print("[dim]Usage: /forget ID  (see /memory)[/]")
                elif cmd == "/tools":
                    console.print(_tools_table(app))
                elif cmd == "/reminders":
                    _print_reminders(app)
                elif cmd == "/think":
                    show_thinking = arg.lower() in ("on", "true", "1", "yes")
                    app.models.set_option("chat", "think", show_thinking)
                    console.print(f"[dim]Thinking {'on' if show_thinking else 'off'}.[/]")
                elif cmd == "/model":
                    if arg:
                        try:
                            app.models.set_role("chat", arg)
                            with console.status("Loading model..."):
                                runner.run(app.models.warmup("chat"))
                            console.print(f"[dim]Switched to {app.models.roles['chat']}[/]")
                        except (ValueError, ProviderError) as e:
                            console.print(f"[red]✗ {e}[/]")
                    else:
                        console.print(f"[dim]{app.models.roles['chat']}[/]")
                else:
                    console.print(HELP)
                continue

            if conv is None:
                conv = app.conversations.new()
            usage = None
            try:
                usage = runner.run(_stream_turn(app, conv.id, text, show_thinking))
            except KeyboardInterrupt:
                console.print("\n[dim]⏹ interrupted[/]")
            except ProviderError as e:
                console.print(f"\n[red]✗ {e}[/]")
            if stats := _stats(usage):
                console.print(f"[dim]{stats}[/]")

        runner.run(app.aclose())
    if conv:
        console.print(f"[dim]Saved as #{conv.id}. Continue with: jarvis chat --resume {conv.id}[/]")
    return 0


# ---------------------------------------------------------------- history


def cmd_history(settings: Settings, args: argparse.Namespace) -> int:
    app = JarvisApp.create(settings)
    try:
        if args.id is None:
            convs = app.store.list(args.limit)
            if not convs:
                console.print("[dim]No conversations yet.[/]")
            else:
                console.print(_conversation_table(convs))
            return 0
        try:
            conv = app.conversations.get(args.id)
        except ConversationNotFound as e:
            console.print(f"[red]✗ {e}[/]")
            return 1
        console.print(f"[bold]#{conv.id} {conv.title or ''}[/]  [dim]{conv.created_at[:16]}[/]\n")
        for m in app.store.messages(conv.id):
            _print_message(m)
        return 0
    finally:
        asyncio.run(app.aclose())


# ---------------------------------------------------------------- eval


def cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    from jarvis.evals import EvalRunner, load_cases, print_report, save_results

    cases = load_cases(args.cases)
    if args.category:
        cases = [c for c in cases if c.category in args.category.split(",")]
    if args.only:
        cases = [c for c in cases if c.id in args.only.split(",")]
    if not cases:
        console.print("[red]No eval cases matched.[/]")
        return 1

    app = JarvisApp.create(settings)
    refs = args.models.split(",") if args.models else [str(app.models.roles["chat"])]
    options = {"think": args.think} if args.think is not None else None

    async def run():
        try:
            runner = EvalRunner(app, console=console)
            return [await runner.run_model(ref, cases, options=options) for ref in refs]
        finally:
            await app.aclose()

    results = asyncio.run(run())
    print_report(results, console, verbose=args.verbose)
    path = save_results(results, settings.app.data_dir / "evals")
    console.print(f"[dim]Full results: {path}[/]")
    return 0


# ---------------------------------------------------------------- memory


def _print_memory_changes(changes) -> None:
    icons = {"add": "💾 Remembered", "update": "✏️  Updated memory", "delete": "🗑  Forgot"}
    for c in changes:
        if c.op in icons:
            console.print(f"\n[magenta]{icons[c.op]} #{c.id}: {c.text}[/]", highlight=False)


def _print_memories(app: JarvisApp) -> None:
    items = app.memory_store.list()
    if not items:
        console.print("[dim]No memories yet. Tell me about yourself, or use /remember TEXT.[/]")
        return
    table = Table(header_style="bold", box=None, padding=(0, 2))
    for col in ("ID", "Category", "Memory", "Source", "Used"):
        table.add_column(col, overflow="fold")
    for m in items:
        table.add_row(str(m.id), m.category, m.text, m.source, str(m.use_count))
    console.print(table)
    state = "on" if app.memory.enabled else "[yellow]off[/]"
    console.print(f"[dim]{len(items)} memories · memory {state} · /forget ID to delete[/]")


def cmd_memory(settings: Settings, args: argparse.Namespace) -> int:
    app = JarvisApp.create(settings)
    code = 0
    try:
        action = args.action or "list"
        if action == "list":
            _print_memories(app)
        elif action == "search":
            hits = asyncio.run(app.memory.recall(" ".join(args.rest), limit=20, min_score=0.3))
            for h in hits:
                console.print(
                    f"  #{h.memory.id}  [dim]{h.score:.2f} {h.how}[/]  {h.memory.text}",
                    highlight=False,
                )
            if not hits:
                console.print("[dim]Nothing found.[/]")
        elif action == "forget":
            for raw in args.rest:
                mid = int(raw.lstrip("#"))
                console.print(
                    f"#{mid}: " + ("forgotten" if app.memory.forget(mid) else "not found")
                )
        elif action == "clear":
            n = app.memory_store.count()
            if console.input(f"Delete all {n} memories? Type 'yes' to confirm › ").strip() == "yes":
                app.memory_store.clear()
                console.print(f"Deleted {n} memories.")
        elif action == "export":
            path = Path(args.rest[0]) if args.rest else Path("jarvis-memories.json")
            data = [m.model_dump() for m in app.memory_store.list()]
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            console.print(f"Exported {len(data)} memories to {path.resolve()}")
        else:
            console.print(
                "Usage: jarvis memory [list | search TEXT | forget ID... | clear | export FILE]"
            )
            code = 1
    finally:
        asyncio.run(app.aclose())
    return code


# ---------------------------------------------------------------- tools / audit / reminders


def _tools_table(app: JarvisApp) -> Table:
    table = Table(header_style="bold", box=None, padding=(0, 2))
    table.add_column("Tool", no_wrap=True)
    table.add_column("Risk")
    table.add_column("Permission")
    table.add_column("What it does")
    colors = {"allow": "green", "ask": "yellow", "deny": "red"}
    for t in app.registry.tools:
        perm = app.policy.overrides.get(t.name) or app.policy.risk_defaults[t.risk]
        table.add_row(t.name, str(t.risk), f"[{colors[perm]}]{perm}[/]", t.description)
    return table


def _print_reminders(app: JarvisApp) -> None:
    rows = reminder_rows(app.db)
    if not rows:
        console.print("[dim]No upcoming reminders.[/]")
    for r in rows:
        console.print(f"  #{r['id']}  {r['due']}  {r['text']}", highlight=False)


def cmd_tools(settings: Settings, args: argparse.Namespace) -> int:
    app = JarvisApp.create(settings)
    console.print(_tools_table(app))
    pol = app.file_policy
    console.print("\n[bold]File access[/]")
    for r in app.file_roots:
        console.print(
            f"  [green]your folder[/]  {r}  [dim](read freely; changes ask)[/]", highlight=False
        )
    if not app.file_roots:
        console.print("  [red]none of the configured folders exist[/]")
    rest = "read-only" if pol.read_anywhere else "no access"
    writes = "every change asks (no 'always')" if pol.outside_writes == "ask" else "no changes"
    console.print(f"  [yellow]everywhere else[/]  {rest}; {writes}")
    console.print(
        f"  [red]never touched[/]  {len(pol.no_access)} private locations "
        "(browser data, keys, credentials)"
    )
    console.print("  [red]never changed[/]  Windows, Program Files, Jarvis config & data")
    asyncio.run(app.aclose())
    return 0


def cmd_audit(settings: Settings, args: argparse.Namespace) -> int:
    app = JarvisApp.create(settings)
    entries = app.audit.recent(args.limit)
    if not entries:
        console.print("[dim]No tool calls yet.[/]")
    table = Table(header_style="bold", box=None, padding=(0, 2))
    for col in ("When", "Tool", "Decision", "OK", "Arguments", "Result"):
        table.add_column(col, overflow="fold")
    for e in reversed(entries):
        table.add_row(
            e.created_at[:19].replace("T", " "),
            e.tool,
            e.decision,
            "[green]✓[/]" if e.ok else "[red]✗[/]",
            _first_line(e.arguments, 50),
            _first_line(e.output or "", 60),
        )
    console.print(table)
    asyncio.run(app.aclose())
    return 0


def cmd_reminders(settings: Settings, args: argparse.Namespace) -> int:
    app = JarvisApp.create(settings)
    _print_reminders(app)
    asyncio.run(app.aclose())
    return 0


def cmd_apps(settings: Settings, args: argparse.Namespace) -> int:
    app = JarvisApp.create(settings)
    with console.status("Scanning installed apps..."):
        apps = app.apps.apps()
    if args.query:
        match = app.apps.find(args.query)
        site = app.apps.find_for_site(args.query) if "." in args.query else None
        console.print(
            f"“open {args.query}” → "
            + (
                f"[green]{match.name}[/] [dim]({match.source}: {match.target})[/]"
                if match
                else "[yellow]no installed app[/]"
            ),
            highlight=False,
        )
        if "." in args.query:
            console.print(
                f"website {args.query} → "
                + (
                    f"[green]opens the {site.name} app[/]"
                    if site
                    else "[dim]opens in the browser[/]"
                ),
                highlight=False,
            )
    else:
        for a in sorted(apps, key=lambda a: a.name.lower()):
            console.print(f"  {a.name}  [dim]{a.source}[/]", highlight=False)
        console.print(f"\n[dim]{len(apps)} apps found. Try: jarvis apps discord[/]")
    asyncio.run(app.aclose())
    return 0


def cmd_notify(settings: Settings, args: argparse.Namespace) -> int:
    app = JarvisApp.create(settings)
    app.notifier.notify("Jarvis", args.text)
    console.print("[dim]Sent. You should see a Windows notification in a second or two.[/]")
    asyncio.run(app.aclose())
    return 0


# ---------------------------------------------------------------- voice

VOICE_TEST_PHRASE = "Jarvis, remind me to call Priya at six thirty tomorrow evening."


def _greeting() -> str:
    from datetime import datetime

    hour = datetime.now().hour
    part = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
    return f"Good {part}. Jarvis is online."


def _voice_setup(settings: Settings) -> int:
    from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn

    from jarvis.voice import components as vc
    from jarvis.voice.tts import download_models, missing_files

    v = settings.voice
    kdir = vc.kokoro_dir(settings)
    missing = missing_files(kdir, v.tts_model)
    if missing:
        console.print(f"Downloading the voice ({', '.join(missing)}) …")
        with Progress(
            TextColumn("{task.description}"), BarColumn(), DownloadColumn(), console=console
        ) as progress:
            tasks: dict[str, int] = {}

            def report(name: str, done: int, total: int) -> None:
                if name not in tasks:
                    tasks[name] = progress.add_task(name, total=total or None)
                progress.update(tasks[name], completed=done)

            download_models(kdir, v.tts_model, report)
    console.print(f"[green]✓[/] Voice files in {kdir}")

    if vc.whisper_downloaded(settings):
        console.print(f"[green]✓[/] Speech model {v.stt_model} already downloaded")
    else:
        console.print(f"Downloading the speech model {v.stt_model} (~1.5 GB, once) …")
        vc.download_whisper(settings)
        console.print(f"[green]✓[/] Speech model in {vc.whisper_dir(settings)}")

    stt = vc.make_stt(settings)
    with console.status("Loading the speech model (GPU if possible)…"):
        stt.load()
    console.print(f"[green]✓[/] Speech recognition: {stt.describe()}")
    if stt.fallback_reason:
        console.print(f"[yellow]  GPU didn't work, using the CPU: {stt.fallback_reason}[/]")
    console.print(
        "\nAll set. Next: [bold]uv run jarvis voice test[/], then [bold]uv run jarvis voice[/]"
    )
    return 0


def _voice_say(settings: Settings, text: str, voice: str | None, save: Path | None) -> int:
    from jarvis.voice import components as vc

    tts = vc.make_tts(settings)
    if voice:
        tts.voice = voice
    t0 = time.perf_counter()
    audio = tts.synthesize(text)
    ms = (time.perf_counter() - t0) * 1000
    secs = len(audio) / tts.sample_rate
    console.print(f"[dim]{tts.voice}: {secs:.1f} s of speech made in {ms:.0f} ms[/]")
    if save:
        vc.save_wav(save, audio, tts.sample_rate)
        console.print(f"Saved {save.resolve()}")
        return 0
    speaker = vc.make_speaker(settings)
    speaker.play(audio, tts.sample_rate)
    speaker.wait(secs + 5)
    time.sleep(0.2)
    speaker.close()
    return 0


def _voice_voices(settings: Settings, play: bool) -> int:
    from jarvis.voice import components as vc
    from jarvis.voice.tts import VOICES

    current = settings.voice.tts_voice
    speaker = tts = None
    if play:
        tts = vc.make_tts(settings)
        speaker = vc.make_speaker(settings)
    for name, about in VOICES.items():
        mark = "[green]●[/]" if name == current else " "
        console.print(f" {mark} [bold]{name:<12}[/] {about}", highlight=False)
        if play:
            audio = tts.synthesize(f"Hello, I'm {name.split('_')[1].title()}.", voice=name)
            speaker.play(audio, tts.sample_rate)
            speaker.wait(10)
    if speaker:
        speaker.close()
    console.print(
        '\n[dim]Change it in config/local.toml:  [voice] tts_voice = "bm_lewis"   '
        "· hear them: jarvis voice voices --play · or /voice NAME in voice mode[/]"
    )
    return 0


def _voice_devices() -> int:
    from jarvis.voice.audio import list_devices

    table = Table(header_style="bold", box=None, padding=(0, 2))
    for col in ("#", "Device", "API", "In", "Out", "Rate"):
        table.add_column(col)
    for d in list_devices():
        mark = " [green](default in)[/]" if d["default_input"] else ""
        mark += " [green](default out)[/]" if d["default_output"] else ""
        table.add_row(
            str(d["index"]),
            d["name"] + mark,
            d["hostapi"],
            str(d["inputs"]),
            str(d["outputs"]),
            f"{d['rate']:.0f}",
        )
    console.print(table)
    console.print(
        '[dim]Pick one in config/local.toml:  [voice] input_device = "3"  (or part of its name)[/]'
    )
    return 0


def _voice_test(settings: Settings, mic_test: bool) -> int:
    import threading

    from jarvis.voice import components as vc
    from jarvis.voice.audio import SAMPLE_RATE, resample, rms
    from jarvis.voice.stt import STTError
    from jarvis.voice.tts import TTSError

    ok = True
    try:
        tts = vc.make_tts(settings)
        with console.status("Loading the voice…"):
            tts.load()
            tts.synthesize("warm up")
        t0 = time.perf_counter()
        audio = tts.synthesize(VOICE_TEST_PHRASE)
        synth_ms = (time.perf_counter() - t0) * 1000
        secs = len(audio) / tts.sample_rate
        console.print(
            f"[green]✓[/] Voice {tts.voice}: {secs:.1f} s of speech in {synth_ms:.0f} ms "
            f"({secs * 1000 / synth_ms:.0f}× real time)"
        )
    except TTSError as e:
        console.print(f"[red]✗ {e}[/]")
        return 1

    speaker = vc.make_speaker(settings)
    try:
        console.print("[dim]Playing it now — you should hear Jarvis…[/]")
        speaker.play(audio, tts.sample_rate)
        speaker.wait(secs + 5)
        console.print("[green]✓[/] Speakers")
    except Exception as e:
        console.print(f"[red]✗ speakers: {e}[/]")
        ok = False

    stt = vc.make_stt(settings)
    try:
        with console.status("Loading the speech model…"):
            t0 = time.perf_counter()
            stt.load()
            load_s = time.perf_counter() - t0
        console.print(
            f"[green]✓[/] Speech recognition: {stt.describe()} (loaded in {load_s:.1f} s)"
        )
        if stt.fallback_reason:
            console.print(f"[yellow]  GPU failed, on CPU instead: {stt.fallback_reason}[/]")
        result = stt.transcribe(resample(audio, tts.sample_rate, SAMPLE_RATE))
        acc = vc.word_accuracy(VOICE_TEST_PHRASE, result.text)
        good = acc >= 0.8
        ok = ok and good
        console.print(
            f"{'[green]✓[/]' if good else '[red]✗[/]'} Round trip (voice → speech model): "
            f"{acc:.0%} of words, {result.ms:.0f} ms\n    heard: {escape(result.text)!r}"
        )
    except STTError as e:
        console.print(f"[red]✗ {e}[/]")
        return 1

    if mic_test:
        mic = vc.make_mic(settings)
        try:
            from jarvis.voice.vad import EndpointConfig

            mic.start()
            console.print(
                "\n[bold]Microphone:[/] say something, e.g. [italic]“What's the time?”[/] "
                "(I'll stop when you pause)"
            )
            beep = speaker.earcon("listen")
            cfg = EndpointConfig(
                threshold=settings.voice.vad_threshold,
                silence_ms=settings.voice.silence_ms,
                start_timeout_s=8,
            )
            clip = mic.record(
                threading.Event(), cfg, seed=False, mute_s=beep + speaker.latency + 0.05
            )
            if clip is None:
                console.print(
                    "[red]✗ didn't hear any speech.[/] Check the mic in Windows sound "
                    "settings, or pick one: jarvis voice devices"
                )
                ok = False
            else:
                result = stt.transcribe(clip)
                console.print(
                    f"[green]✓[/] Heard {len(clip) / SAMPLE_RATE:.1f} s (level {rms(clip):.3f}) "
                    f"in {result.ms:.0f} ms: [bold]{escape(result.text) or '(nothing clear)'}[/]"
                )
        except Exception as e:
            console.print(f"[red]✗ microphone: {e}[/]")
            ok = False
        finally:
            mic.close()
    speaker.close()
    console.print(
        "\n"
        + (
            "[green]Voice is ready.[/] Run: uv run jarvis voice"
            if ok
            else "[yellow]Some checks failed.[/]"
        )
    )
    return 0 if ok else 1


def cmd_voice(settings: Settings, args: argparse.Namespace) -> int:
    action = args.action or "start"
    if args.voice:
        settings.voice.tts_voice = args.voice
    try:
        if action == "setup":
            return _voice_setup(settings)
        if action == "say":
            return _voice_say(settings, " ".join(args.rest) or _greeting(), args.voice, args.save)
        if action == "voices":
            return _voice_voices(settings, args.play)
        if action == "devices":
            return _voice_devices()
        if action == "test":
            return _voice_test(settings, not args.no_mic)
    except Exception as e:  # missing models, audio, CUDA … show it plainly
        console.print(f"[red]✗ {e}[/]")
        return 1
    return _voice_start(settings, args)


def _voice_start(settings: Settings, args: argparse.Namespace) -> int:
    from jarvis.voice import components as vc
    from jarvis.voice.hotkey import Hotkey, describe
    from jarvis.voice.session import VoiceSession
    from jarvis.voice.tts import SpeechQueue, missing_files

    v = settings.voice
    if args.no_speak:
        v.speak = False
    if missing_files(vc.kokoro_dir(settings), v.tts_model) or not vc.whisper_downloaded(settings):
        console.print("[yellow]Voice isn't set up yet.[/] Run: [bold]uv run jarvis voice setup[/]")
        return 1

    app = JarvisApp.create(settings)
    if args.model:
        app.models.set_role("chat", args.model)
    conv = None
    if args.resume:
        try:
            conv = (
                app.store.latest()
                if args.resume == "last"
                else app.conversations.get(int(args.resume))
            )
        except (ConversationNotFound, ValueError) as e:
            console.print(f"[red]✗ {e}[/]")
            return 1

    stt, tts = vc.make_stt(settings), vc.make_tts(settings)
    mic, speaker = vc.make_mic(settings), vc.make_speaker(settings)
    speech = None
    code = 0
    with asyncio.Runner() as runner:
        try:
            with console.status("Loading the language model into VRAM…"):
                runner.run(app.models.warmup("chat"))
            with console.status("Loading speech recognition…"):
                stt.load()
            with console.status("Loading the voice…"):
                tts.load()
                tts.synthesize("ready")
            mic.start()
            speaker.start()
        except Exception as e:
            console.print(f"[red]✗ {e}[/]\n[dim]Try: uv run jarvis voice test[/]")
            runner.run(app.aclose())
            mic.close()
            speaker.close()
            return 1

        speech = SpeechQueue(tts, speaker)
        session = VoiceSession(
            app,
            v,
            stt=stt,
            mic=mic,
            speech=speech,
            console=console,
            hotkey_factory=Hotkey,
            conversation=conv,
            voice_names=lambda: list(tts.voices()),
        )

        def on_reminder(title: str, msg: str) -> None:  # scheduler thread
            console.print(f"\n[bold magenta]⏰ Reminder: {msg}[/]")
            session.post("reminder", msg)  # spoken when Jarvis isn't listening/talking

        app.notifier.add(CallbackNotifier(on_reminder))
        app.scheduler.start()
        app.apps.prefetch()
        try:
            runner.run(app.memory.backfill())
        except ProviderError:
            pass
        app.memory_listeners.append(_print_memory_changes)

        key = describe(v.hotkey)
        console.print(
            f"[bold]{settings.app.name}[/] v{__version__} · voice mode · "
            f"model [cyan]{app.models.roles['chat']}[/] · hearing: {stt.describe()} · "
            f"voice: {tts.voice}"
        )
        if conv:
            console.print(f"[dim]Resuming #{conv.id}: {conv.title or '(untitled)'}[/]")
        console.print(
            f"[bold]Tap {key}[/] and speak — I'll stop listening when you pause. "
            f"Tap {key} while I'm talking to cut me off. Typing works too. [dim](/help)[/]"
        )
        if stt.fallback_reason:
            console.print(
                f"[yellow]Speech recognition is on the CPU (GPU failed): {stt.fallback_reason}[/]"
            )

        try:
            runner.run(session.run(greeting=_greeting()))
        except KeyboardInterrupt:
            pass
        except RuntimeError as e:  # e.g. hotkeys unavailable
            console.print(f"[red]✗ {e}[/]")
            code = 1
        finally:
            speech.close()
            mic.close()
            speaker.close()
            runner.run(app.aclose())
    if session.conv:
        console.print(
            f"[dim]Saved as #{session.conv.id}. "
            f"Continue: jarvis voice --resume {session.conv.id}[/]"
        )
    return code


# ---------------------------------------------------------------- serve


def cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn

    from jarvis.interfaces.api import create_api

    host = args.host or settings.server.host
    port = args.port or settings.server.port
    console.print(f"Jarvis API on http://{host}:{port}  (docs: /docs)")
    uvicorn.run(create_api(settings), host=host, port=port, log_level="warning")
    return 0


# ---------------------------------------------------------------- doctor


def _gpu_info() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _voice_doctor_rows(settings: Settings) -> list[tuple[str, bool | None, str]]:
    """Cheap checks only (no model loading) — `jarvis voice test` does the real thing."""
    from jarvis.voice import components as vc
    from jarvis.voice.tts import missing_files

    rows: list[tuple[str, bool | None, str]] = []
    v = settings.voice
    try:
        from jarvis.voice.audio import default_device_names

        mic, out = default_device_names()
        rows.append(
            ("Voice: audio", bool(mic and out), f"mic: {mic or 'none!'} · out: {out or 'none!'}")
        )
    except Exception as e:
        rows.append(("Voice: audio", False, str(e)))
    have_stt = vc.whisper_downloaded(settings)
    have_tts = not missing_files(vc.kokoro_dir(settings), v.tts_model)
    if not (have_stt and have_tts):
        rows.append(("Voice: models", None, "not set up yet — run: jarvis voice setup"))
        return rows
    detail = f"hearing {v.stt_model} · voice {v.tts_voice}"
    if sys.platform == "win32":
        from jarvis.voice.stt import cuda_library_dirs

        n = len(cuda_library_dirs())
        detail += f" · CUDA libs {'found' if n else 'missing (speech runs on CPU)'}"
    rows.append(("Voice: models", True, detail + "  (full check: jarvis voice test)"))
    return rows


async def _doctor(settings: Settings, quick: bool) -> list[tuple[str, bool | None, str]]:
    rows: list[tuple[str, bool | None, str]] = []
    py_ok = sys.version_info >= (3, 11)
    rows.append(("Python", py_ok, sys.version.split()[0]))
    rows.append(("Config", True, str(settings.config_dir)))

    gpu = _gpu_info()
    rows.append(
        ("NVIDIA GPU", gpu is not None, gpu or "nvidia-smi not found — is the driver installed?")
    )

    app = JarvisApp.create(settings)
    roots = ", ".join(str(r) for r in app.file_roots)
    rows.append(
        (
            "File access",
            bool(app.file_roots) or not settings.tools.enabled,
            roots or "none of the configured folders exist — check \\[tools.files] roots",
        )
    )
    rows.append(("Tools", True, f"{len(app.registry)} enabled  (see: jarvis tools)"))
    rows.extend(_voice_doctor_rows(settings))
    try:
        ref = app.models.roles["chat"]
        rows.append(("Chat model", True, str(ref)))
        provider, model = app.models.resolve("chat")
        health = await provider.health()
        rows.append(
            (
                f"{provider.name} server",
                health.ok,
                f"v{health.version} at {health.detail}" if health.ok else health.detail,
            )
        )
        if not health.ok:
            return rows

        available = await provider.list_models()
        wanted = model if ":" in model else f"{model}:latest"
        pulled = wanted in available or model in available
        rows.append(("Model pulled", pulled, "yes" if pulled else f"run:  ollama pull {model}"))
        emb = app.models.roles.get("embed")
        if emb is not None and emb.provider == provider.name:
            emb_model = emb.model if ":" in emb.model else f"{emb.model}:latest"
            have = emb_model in available or emb.model in available
            rows.append(
                (
                    "Memory",
                    have,
                    f"{app.memory_store.count()} memories · embeddings {emb}"
                    if have
                    else f"run:  ollama pull {emb.model}   (until then: keyword recall)",
                )
            )
        if not pulled or quick:
            return rows

        parts: list[str] = []
        usage = None
        await app.models.warmup("chat")
        stream = app.chat_stream([Message.user("Reply with exactly one word: ready")])
        async for chunk in stream:
            parts.append(chunk.text)
            usage = chunk.usage or usage
        rows.append(
            (
                "Test generation",
                bool("".join(parts).strip()),
                f"{''.join(parts).strip()[:40]!r} · {_stats(usage)}",
            )
        )

        for m in await provider.loaded_models():
            if m.size_bytes and m.vram_bytes is not None:
                pct = 100 * m.vram_bytes / m.size_bytes
                ok = pct >= 99
                rows.append(
                    (
                        f"VRAM: {m.name}",
                        ok,
                        f"{m.vram_bytes / 1e9:.1f} GB on GPU ({pct:.0f}%)"
                        + (
                            ""
                            if ok
                            else " — partly on CPU, will be slow; try a smaller model/quant"
                        ),
                    )
                )
    except ProviderError as e:
        rows.append(("Error", False, str(e)))
    finally:
        await app.aclose()
    return rows


def cmd_doctor(settings: Settings, args: argparse.Namespace) -> int:
    rows = asyncio.run(_doctor(settings, args.quick))
    table = Table(title="Jarvis doctor", show_lines=False)
    table.add_column("")
    table.add_column("Check")
    table.add_column("Result")
    for name, ok, detail in rows:
        mark = "[green]✓[/]" if ok else ("[yellow]?[/]" if ok is None else "[red]✗[/]")
        table.add_row(mark, name, detail)
    console.print(table)
    return 0 if all(ok is not False for _, ok, _ in rows) else 1


# ---------------------------------------------------------------- entry


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jarvis", description="Local-first AI assistant")
    p.add_argument("--config-dir", type=Path, help="folder containing default.toml / local.toml")
    p.add_argument("--version", action="version", version=f"jarvis {__version__}")
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("chat", help="chat in the terminal (default)")
    c.add_argument("--model", help="override chat model, e.g. ollama:qwen3:8b")
    c.add_argument("--echo", action="store_true", help="use the offline echo provider")
    c.add_argument("--show-thinking", action="store_true", help="turn thinking on and show it")
    c.add_argument(
        "--resume",
        nargs="?",
        const="last",
        metavar="ID",
        help="continue a conversation (default: the most recent)",
    )

    h = sub.add_parser("history", help="list conversations, or print one")
    h.add_argument("id", nargs="?", type=int)
    h.add_argument("--limit", type=int, default=20)

    e = sub.add_parser("eval", help="score models on the eval cases")
    e.add_argument("--models", help="comma-separated refs, e.g. ollama:qwen3:14b,ollama:qwen3:8b")
    e.add_argument("--cases", type=Path, help="cases file (default: evals/cases.toml)")
    e.add_argument("--category", help="only these categories (comma-separated)")
    e.add_argument("--only", help="only these case ids (comma-separated)")
    e.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force thinking on/off (default: config)",
    )
    e.add_argument("-v", "--verbose", action="store_true", help="show every failure")

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host")
    s.add_argument("--port", type=int)

    mem = sub.add_parser(
        "memory", help="what Jarvis remembers: list | search | forget | clear | export"
    )
    mem.add_argument("action", nargs="?", choices=["list", "search", "forget", "clear", "export"])
    mem.add_argument("rest", nargs="*")

    vp = sub.add_parser(
        "voice", help="talk to Jarvis: voice [setup | test | say TEXT | voices | devices]"
    )
    vp.add_argument(
        "action", nargs="?", choices=["start", "setup", "test", "say", "voices", "devices"]
    )
    vp.add_argument("rest", nargs="*", help="text for `say`")
    vp.add_argument("--voice", help="Kokoro voice, e.g. bm_george (see: jarvis voice voices)")
    vp.add_argument("--model", help="override chat model")
    vp.add_argument("--resume", nargs="?", const="last", metavar="ID")
    vp.add_argument("--no-speak", action="store_true", help="listen, but reply in text only")
    vp.add_argument("--no-mic", action="store_true", help="test: skip the microphone check")
    vp.add_argument("--play", action="store_true", help="voices: hear each one")
    vp.add_argument("--save", type=Path, help="say: write a .wav instead of playing")

    sub.add_parser("tools", help="list tools and their permissions")
    a = sub.add_parser("audit", help="recent tool calls")
    a.add_argument("--limit", type=int, default=30)
    sub.add_parser("reminders", help="upcoming reminders")
    ap = sub.add_parser(
        "apps", help="installed apps Jarvis can open (or test one: jarvis apps discord)"
    )
    ap.add_argument("query", nargs="?")
    n = sub.add_parser("notify", help="send a test notification")
    n.add_argument("text", nargs="?", default="Hello from Jarvis 👋")

    d = sub.add_parser("doctor", help="check your setup")
    d.add_argument("--quick", action="store_true", help="skip the test generation")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        args = build_parser().parse_args([*(argv or sys.argv[1:]), "chat"])
    settings = load_settings(args.config_dir)
    setup_logging(settings)
    handlers = {
        "chat": cmd_chat,
        "history": cmd_history,
        "eval": cmd_eval,
        "serve": cmd_serve,
        "doctor": cmd_doctor,
        "tools": cmd_tools,
        "memory": cmd_memory,
        "audit": cmd_audit,
        "reminders": cmd_reminders,
        "apps": cmd_apps,
        "notify": cmd_notify,
        "voice": cmd_voice,
    }
    return handlers[args.command](settings, args)


if __name__ == "__main__":
    raise SystemExit(main())
