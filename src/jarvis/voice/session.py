"""VoiceSession: the talk loop.

    tap hotkey ──► listen (auto-stops when you go quiet) ──► Whisper ──► conversation
        ▲                                                                     │
        └── tap again while Jarvis talks = stop talking & listen ◄── speak ◄──┘

Everything arrives as an event on one asyncio queue — hotkey taps (from pynput's
thread) and typed lines (from a stdin thread) — so typing and talking both work,
and an approval can be answered either way ("yes" out loud, or `y` + Enter).
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import threading
import time
from collections.abc import Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from functools import partial

from rich.console import Console
from rich.markup import escape

from jarvis.core.config import VoiceSettings
from jarvis.core.types import Usage
from jarvis.providers import ProviderError
from jarvis.tools import ApprovalAnswer, ApprovalRequest, Risk
from jarvis.voice.text import SentenceSplitter, parse_approval, speech_text
from jarvis.voice.vad import EndpointConfig

log = logging.getLogger(__name__)

VOICE_HELP = """[bold]Voice mode[/]
  tap {key}       talk (Jarvis stops listening when you pause)
  tap {key} again while Jarvis talks → it stops and listens
  Enter           stop Jarvis talking
  type a message  works too — the reply is spoken
  /new            new conversation      /voice NAME   change voice (/voices lists them)
  /mute /unmute   spoken replies off/on  /exit         quit (or Ctrl+C)"""


@dataclass
class Event:
    kind: str  # "tap" | "text" | "eof"
    text: str = ""


@dataclass
class TurnTiming:
    speech_end: float | None = None  # when you stopped talking (perf_counter)
    stt_ms: float | None = None
    usage: Usage | None = None
    first_audio_ms: float | None = None
    extra: dict = field(default_factory=dict)


def _first_line(text: str, limit: int = 110) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _call_str(call) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
    return _first_line(f"{call.name}({args})")


class VoiceSession:
    def __init__(
        self,
        app,
        cfg: VoiceSettings,
        *,
        stt,
        mic,
        speech,
        console: Console | None = None,
        hotkey_factory: Callable | None = None,
        read_stdin: bool = True,
        conversation=None,
        voice_names: Callable[[], list[str]] | None = None,
    ):
        self.app = app
        self.cfg = cfg
        self.stt = stt
        self.mic = mic
        self.speech = speech  # SpeechQueue
        self.console = console or Console()
        self.hotkey_factory = hotkey_factory
        self.read_stdin = read_stdin
        self.conv = conversation
        self.voice_names = voice_names
        self.speak = cfg.speak
        self.endpoint = EndpointConfig(
            threshold=cfg.vad_threshold,
            silence_ms=cfg.silence_ms,
            start_timeout_s=cfg.start_timeout_s,
            max_s=cfg.max_listen_s,
        )
        self.events: asyncio.Queue[Event] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._listen_stop: threading.Event | None = None
        self._hearing = False  # recording or transcribing
        self._ignore_taps_until = 0.0
        self._approval: asyncio.Future | None = None
        self._deferred: list[Event] = []
        self._hotkey = None
        self.turns = 0
        self.timing = TurnTiming()

    # ------------------------------------------------------------ plumbing

    def post(self, kind: str, text: str = "") -> None:
        """Thread-safe: hand an event to the session loop."""
        if self._loop and self.events is not None:
            self._loop.call_soon_threadsafe(self._on_event, Event(kind, text))

    def _on_event(self, ev: Event) -> None:
        # A tap while listening means "I'm done talking" — handled right here so it
        # works even while the loop is busy waiting on the recorder.
        if ev.kind == "tap":
            if self._listen_stop is not None:
                self._listen_stop.set()
                return
            # A second tap meant as "I'm done" can land just after the pause already
            # ended the recording — it must not interrupt the reply that follows.
            if self._hearing or time.perf_counter() < self._ignore_taps_until:
                return
        self.events.put_nowait(ev)

    def tap(self) -> None:
        self.post("tap")

    def _stdin_reader(self) -> None:
        while True:
            try:
                line = sys.stdin.readline()
            except (OSError, ValueError):
                line = ""
            if not line:
                self.post("eof")
                return
            self.post("text", line.rstrip("\r\n"))

    def earcon(self, name: str) -> float:
        """Play a cue; returns how long it lasts (0 if cues are off)."""
        if self.cfg.earcons:
            try:
                return float(self.speech.speaker.earcon(name) or 0.0)
            except Exception:
                log.debug("earcon failed", exc_info=True)
        return 0.0

    def say(self, text: str) -> None:
        if self.speak:
            self.speech.say(text)

    async def wait_speech(self) -> None:
        while self.speech.busy:  # noqa: ASYNC110 — state lives on audio threads
            await asyncio.sleep(0.03)

    # ------------------------------------------------------------ main loop

    async def run(self, greeting: str | None = None) -> None:
        self._loop = asyncio.get_running_loop()
        self.events = asyncio.Queue()
        if self.hotkey_factory:
            self._hotkey = self.hotkey_factory(self.cfg.hotkey, self.tap)
            self._hotkey.start()
        if self.read_stdin:
            threading.Thread(target=self._stdin_reader, name="jarvis-stdin", daemon=True).start()
        if greeting:
            self.console.print(f"[bold green]jarvis ›[/] {escape(greeting)}")
            self.say(greeting)
        try:
            while True:
                ev = self._deferred.pop(0) if self._deferred else await self.events.get()
                if ev.kind == "eof":
                    break
                if ev.kind == "reminder":
                    self.say(f"Reminder: {ev.text}")
                    continue
                if ev.kind == "tap":
                    text = await self.hear()
                    if text:
                        await self.respond(text)
                elif ev.kind == "text":
                    line = ev.text.strip()
                    if not line:
                        self.speech.stop()
                        continue
                    if line.startswith("/"):
                        if self.command(line) == "exit":
                            break
                        continue
                    await self.respond(line, spoken=False)
        finally:
            if self._hotkey:
                self._hotkey.stop()

    def command(self, line: str) -> str | None:
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd in ("/exit", "/quit"):
            return "exit"
        if cmd in ("/new", "/reset"):
            self.conv = None
            self.console.print("[dim]New conversation.[/]")
        elif cmd == "/mute":
            self.speak = False
            self.speech.stop()
            self.console.print("[dim]Spoken replies off (/unmute to turn back on).[/]")
        elif cmd == "/unmute":
            self.speak = True
            self.console.print("[dim]Spoken replies on.[/]")
        elif cmd == "/voices":
            names = self.voice_names() if self.voice_names else []
            self.console.print("[dim]" + ", ".join(names) + "[/]")
        elif cmd == "/voice":
            try:
                self.speech.tts.set_voice(arg)
                self.console.print(f"[dim]Voice: {arg}[/]")
                self.say("How do I sound?")
            except Exception as e:
                self.console.print(f"[red]✗ {e}[/]")
        else:
            from jarvis.voice.hotkey import describe

            self.console.print(VOICE_HELP.format(key=describe(self.cfg.hotkey)))
        return None

    # ------------------------------------------------------------ listening

    async def hear(
        self, *, prompt: str = "listening…", start_timeout: float | None = None
    ) -> str | None:
        """Record until you pause, transcribe. None if nothing (intelligible) was said."""
        speaker = self.speech.speaker
        just_spoke = speaker.recently_played()  # its tail may still be in the room
        self.speech.stop()  # half-duplex: never listen to ourselves
        beep = self.earcon("listen")
        # Don't let our own beep (or the end of Jarvis's sentence) count as you talking.
        mute_s = (beep + speaker.latency + 0.05) if (beep or just_spoke) else 0.0
        self.console.print(f"[cyan]🎙  {prompt}[/]")
        stop = threading.Event()
        self._listen_stop = stop
        self._hearing = True
        try:
            return await self._hear(stop, start_timeout, seed=not just_spoke, mute_s=mute_s)
        finally:
            self._hearing = False
            self._listen_stop = None

    async def _hear(
        self, stop: threading.Event, start_timeout: float | None, *, seed: bool, mute_s: float
    ) -> str | None:
        cfg = self.endpoint
        if start_timeout is not None:
            cfg = EndpointConfig(**{**cfg.__dict__, "start_timeout_s": start_timeout})
        self.timing = TurnTiming()
        try:
            audio = await asyncio.to_thread(
                partial(self.mic.record, stop, cfg, seed=seed, mute_s=mute_s)
            )
        except asyncio.CancelledError:
            stop.set()
            raise
        except Exception as e:
            self.console.print(f"[red]✗ microphone: {e}[/]")
            return None
        finally:
            self._listen_stop = None
            self._ignore_taps_until = time.perf_counter() + 0.4
        self.timing.speech_end = time.perf_counter()
        if audio is None:
            self.earcon("cancel")
            self.console.print("[dim]   (didn't hear anything)[/]")
            return None
        self.earcon("done")
        try:
            result = await asyncio.to_thread(self.stt.transcribe, audio)
        except Exception as e:
            self.console.print(f"[red]✗ speech recognition: {e}[/]")
            return None
        self.timing.stt_ms = result.ms
        if not result.text:
            self.console.print("[dim]   (couldn't make that out)[/]")
            return None
        self.console.print(f"[bold cyan]you ›[/] {escape(result.text)}")
        return result.text

    # ------------------------------------------------------------ replying

    async def respond(self, text: str, *, spoken: bool = True) -> None:
        """Run a turn; meanwhile a tap interrupts it and listens again."""
        if not spoken:
            self.timing = TurnTiming()
        task = asyncio.create_task(self._reply(text))
        getter: asyncio.Task | None = None
        try:
            while True:
                getter = asyncio.create_task(self.events.get())
                done, _ = await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
                if getter not in done:
                    break
                ev = getter.result()
                getter = None
                asking = self._approval is not None and not self._approval.done()
                if asking and ev.kind == "tap":
                    self.speech.stop()  # skip the spoken question; listening starts next
                    continue
                if asking and ev.kind == "text":
                    answer = parse_approval(ev.text)
                    if answer is not None:
                        self._approval.set_result(answer)
                    else:
                        self.console.print(r"[dim]  (type y or n)[/]")
                    continue
                if ev.kind == "tap" or (ev.kind == "text" and not ev.text.strip()):
                    # Interrupt: stop talking and thinking; a tap then listens.
                    await self._interrupt(task)
                    if ev.kind != "tap":
                        return
                    again = await self.hear()
                    if not again:
                        return
                    task = asyncio.create_task(self._reply(again))
                    continue
                self._deferred.append(ev)  # typed while busy: handle after this turn
        finally:
            # Also on Ctrl+C: never leave a reply (or a queue read) running while
            # the app shuts down around it.
            if getter is not None and not getter.done():
                getter.cancel()
                await asyncio.wait({getter})
            if not task.done():
                task.cancel()
                await asyncio.wait({task})
        if task.cancelled():
            return
        if (err := task.exception()) is not None:  # one bad turn mustn't end the session
            log.error("voice turn failed", exc_info=err)
            self.speech.stop()
            self.console.print(f"[red]✗ {escape(str(err))}[/]")

    async def _interrupt(self, task: asyncio.Task) -> None:
        self.speech.stop()
        if not task.done():
            task.cancel()
            await asyncio.wait({task})  # (unlike `await task`, keeps our own cancellation)
            self.console.print("[dim]⏹ interrupted[/]")

    async def _reply(self, text: str) -> None:
        app = self.app
        if self.conv is None:
            self.conv = app.conversations.new()
        splitter = SentenceSplitter()
        self.speech.reset_stats()
        state = {"mid_line": False, "need_prefix": True}
        printed: list[str] = []

        def newline() -> None:
            if state["mid_line"]:
                self.console.print()
                state["mid_line"] = False

        stream = app.conversations.send(
            self.conv.id, text, approver=self.approve, system_note=self.cfg.style or None
        )
        try:
            async with aclosing(stream) as chunks:
                async for chunk in chunks:
                    if chunk.event:
                        newline()
                        self._print_event(chunk.event)
                        state["need_prefix"] = True
                    if chunk.text:
                        if state["need_prefix"]:
                            self.console.print("[bold green]jarvis ›[/] ", end="")
                            state["need_prefix"] = False
                        self.console.out(chunk.text, end="", highlight=False)
                        state["mid_line"] = True
                        printed.append(chunk.text)
                        for sentence in splitter.feed(chunk.text):
                            self.say(sentence)
                    if chunk.done:
                        self.timing.usage = chunk.usage  # (a spoken approval replaces timing)
        except ProviderError as e:
            newline()
            self.console.print(f"[red]✗ {e}[/]")
            self.say("Sorry, I couldn't reach the language model.")
            return
        finally:
            newline()
        for sentence in splitter.flush():
            self.say(sentence)
        self.turns += 1
        await self.wait_speech()
        if self.speech.errors:
            self.console.print(f"[red]✗ speech: {self.speech.errors[-1]}[/]")
            self.speech.errors.clear()
        self._print_stats(self.timing)

    def _print_event(self, ev) -> None:
        if ev.kind == "call":
            self.console.print(f"[dim]  ⚙ {escape(_call_str(ev.call))}[/]", highlight=False)
        elif ev.kind == "result":
            self.console.print(f"[dim]    ✓ {escape(_first_line(ev.output))}[/]", highlight=False)
        else:
            color = "yellow" if ev.kind == "denied" else "red"
            self.console.print(
                f"[{color}]    ✗ {escape(_first_line(ev.output))}[/]", highlight=False
            )

    def _print_stats(self, t: TurnTiming) -> None:
        bits = []
        if t.stt_ms is not None:
            bits.append(f"heard in {t.stt_ms:.0f} ms")
        if t.usage and t.usage.answer_ttft_ms is not None:
            bits.append(f"answer in {t.usage.answer_ttft_ms:.0f} ms")
        first = self.speech.first_audio_at
        if first and t.speech_end:
            t.first_audio_ms = (first - t.speech_end) * 1000
            bits.append(f"[bold]spoke {t.first_audio_ms:.0f} ms after you stopped[/]")
        if t.usage and t.usage.tokens_per_s:
            bits.append(f"{t.usage.tokens_per_s:.0f} tok/s")
        if bits:
            self.console.print("[dim]" + " · ".join(bits) + "[/]")
        log.info(
            "voice turn",
            extra={
                "stt_ms": t.stt_ms,
                "answer_ttft_ms": t.usage.answer_ttft_ms if t.usage else None,
                "first_audio_ms": t.first_audio_ms,
            },
        )

    # ------------------------------------------------------------ approvals

    async def approve(self, req: ApprovalRequest) -> ApprovalAnswer:
        """Ask out loud; answer by voice ("yes", "no", "always") or by typing y / n / a."""
        await self.wait_speech()  # let Jarvis finish the sentence it was saying
        color = "red" if req.risk == Risk.DESTRUCTIVE else "yellow"
        options = r"\[y]es / \[n]o" + (r" / \[a]lways" if req.can_allow_for_session else "")
        self.console.print(
            f"\n[{color}]⚠ Jarvis wants to {escape(req.description)}[/]  "
            f"[dim]say yes/no or type {options}[/]",
            highlight=False,
        )
        self._approval = asyncio.get_running_loop().create_future()
        try:
            if self.speak:
                short = re.sub(r"\s*\([^)]*\)", "", req.description)  # drop "(2 chars: …)"
                self.speech.say(f"May I {speech_text(short)}?")
                await self.wait_speech()
            for attempt in range(2):
                if self._approval.done():
                    break
                listen = asyncio.create_task(
                    self.hear(prompt="yes or no?", start_timeout=self.cfg.start_timeout_s)
                )
                done, _ = await asyncio.wait(
                    {listen, self._approval}, return_when=asyncio.FIRST_COMPLETED
                )
                if self._approval in done:
                    if not listen.done():
                        if self._listen_stop:
                            self._listen_stop.set()
                        listen.cancel()
                        await asyncio.wait({listen})
                    break
                heard = listen.result()
                answer = (
                    parse_approval(heard, allow_session=req.can_allow_for_session)
                    if heard
                    else None
                )
                if answer is not None:
                    self._approval.set_result(answer)
                    break
                if attempt == 0 and heard:
                    self.console.print("[dim]  (didn't catch a yes or no)[/]")
                    self.say("Sorry — yes or no?")
                    await self.wait_speech()
            if not self._approval.done():
                self.console.print(r"[dim]  Type y or n and press Enter.[/]")
                try:
                    await asyncio.wait_for(asyncio.shield(self._approval), 60)
                except TimeoutError:
                    self._approval.set_result("deny")
            answer = self._approval.result()
        finally:
            self._approval = None
        if answer == "session" and not req.can_allow_for_session:
            answer = "once"
        self.speech.speaker.mark_first_audio()  # latency now counts from your answer
        label = {"once": "approved", "session": "approved for this session", "deny": "declined"}
        self.console.print(f"[dim]  → {label[answer]}[/]")
        return answer
