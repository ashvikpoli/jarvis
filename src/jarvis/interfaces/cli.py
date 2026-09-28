"""Command line: `jarvis chat`, `jarvis serve`, `jarvis doctor`."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from jarvis import __version__
from jarvis.app import JarvisApp
from jarvis.core.config import Settings, load_settings
from jarvis.core.logging import setup_logging
from jarvis.core.types import Message, Usage
from jarvis.providers import ProviderError

console = Console()

HELP = """[bold]Commands[/]
  /model [ref]   show or switch the chat model, e.g. /model ollama:qwen3:8b
  /reset         clear the conversation
  /help          show this
  /exit          quit
[dim]Ctrl+C while Jarvis is talking interrupts it. Ctrl+C at the prompt quits.[/]"""


# ---------------------------------------------------------------- chat


async def _stream_turn(
    app: JarvisApp, history: list[Message], parts: list[str], show_thinking: bool
) -> Usage | None:
    usage = None
    in_thinking = False
    async for chunk in app.chat_stream(history):
        if chunk.thinking and show_thinking:
            console.print(Text(chunk.thinking, style="dim italic"), end="")
            in_thinking = True
        if chunk.text:
            if in_thinking:
                console.print()
                in_thinking = False
            parts.append(chunk.text)
            console.out(chunk.text, end="", highlight=False)
        if chunk.done:
            usage = chunk.usage
    console.print()
    return usage


def _stats(usage: Usage | None) -> str:
    if not usage:
        return ""
    bits = []
    if usage.ttft_ms is not None:
        bits.append(f"first token {usage.ttft_ms:.0f} ms")
    if usage.tokens_per_s:
        bits.append(f"{usage.tokens_per_s:.0f} tok/s")
    return " · ".join(bits)


def cmd_chat(settings: Settings, args: argparse.Namespace) -> int:
    app = JarvisApp.create(settings)
    if args.echo:
        app.models.set_role("chat", "echo:echo")
    elif args.model:
        app.models.set_role("chat", args.model)

    history: list[Message] = []
    console.print(
        f"[bold]{settings.app.name}[/] v{__version__} · model [cyan]{app.models.roles['chat']}[/]"
        "  [dim](/help for commands)[/]"
    )

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

        while True:
            try:
                text = console.input("\n[bold cyan]you ›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue

            if text.startswith("/"):
                cmd, _, arg = text.partition(" ")
                if cmd in ("/exit", "/quit"):
                    break
                elif cmd == "/reset":
                    history.clear()
                    console.print("[dim]Conversation cleared.[/]")
                elif cmd == "/model":
                    if arg:
                        try:
                            app.models.set_role("chat", arg.strip())
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

            history.append(Message.user(text))
            parts: list[str] = []
            usage = None
            console.print("[bold green]jarvis ›[/] ", end="")
            try:
                usage = runner.run(_stream_turn(app, history, parts, args.show_thinking))
            except KeyboardInterrupt:
                console.print("\n[dim]⏹ interrupted[/]")
            except ProviderError as e:
                console.print(f"\n[red]✗ {e}[/]")

            reply = "".join(parts)
            if reply:
                history.append(Message.assistant(reply))
            else:
                history.pop()  # nothing came back; don't leave a dangling user turn
            if stats := _stats(usage):
                console.print(f"[dim]{stats}[/]")

        runner.run(app.aclose())
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
        async for chunk in app.chat_stream([Message.user("Reply with exactly one word: ready")]):
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
    c.add_argument("--show-thinking", action="store_true", help="print reasoning traces")

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host")
    s.add_argument("--port", type=int)

    d = sub.add_parser("doctor", help="check your setup")
    d.add_argument("--quick", action="store_true", help="skip the test generation")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        args = build_parser().parse_args([*(argv or sys.argv[1:]), "chat"])
    settings = load_settings(args.config_dir)
    setup_logging(settings)
    handlers = {"chat": cmd_chat, "serve": cmd_serve, "doctor": cmd_doctor}
    return handlers[args.command](settings, args)


if __name__ == "__main__":
    raise SystemExit(main())
