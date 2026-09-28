"""Command line interface. Run `jarvis --help` for the list of commands."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
from contextlib import aclosing
from pathlib import Path

from rich.console import Console
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
    roots = ", ".join(str(r) for r in app.file_roots) or "[red]none found[/]"
    console.print(f"\n[dim]File access limited to: {roots}[/]")
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


def cmd_notify(settings: Settings, args: argparse.Namespace) -> int:
    app = JarvisApp.create(settings)
    app.notifier.notify("Jarvis", args.text)
    console.print("[dim]Sent. You should see a Windows notification in a second or two.[/]")
    asyncio.run(app.aclose())
    return 0


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

    sub.add_parser("tools", help="list tools and their permissions")
    a = sub.add_parser("audit", help="recent tool calls")
    a.add_argument("--limit", type=int, default=30)
    sub.add_parser("reminders", help="upcoming reminders")
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
        "audit": cmd_audit,
        "reminders": cmd_reminders,
        "notify": cmd_notify,
    }
    return handlers[args.command](settings, args)


if __name__ == "__main__":
    raise SystemExit(main())
