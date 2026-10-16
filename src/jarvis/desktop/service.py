"""Desktop: what the tools call. Safety checks, element ids, coordinates, threading.

All backend calls run on one dedicated worker thread: Windows UI Automation objects
belong to the (COM) thread that created them, and nothing here may block the event
loop. If an app hangs that thread, it's abandoned and a fresh one takes over.

Safety, checked right before every action (not just once per task):
- Esc Esc, or you moving the mouse, stops everything (Guard).
- Protected apps (password managers, Windows Security) are never looked at or
  controlled — and are blacked out of whole-screen screenshots.
- Clicks land only where checked: the window under the point must be the target.
- Never typing into password fields, terminals, the Run dialog or Start search.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
from collections.abc import Callable
from concurrent.futures import Future

from jarvis.desktop.guard import DesktopStopped, Guard
from jarvis.desktop.model import DesktopBackend, Element, Window
from jarvis.desktop.vision import Vision
from jarvis.tools.base import ToolError

log = logging.getLogger(__name__)

# Never looked at or controlled (passwords, Windows security prompts).
DEFAULT_BLOCKED = [
    "KeePass.exe", "KeePassXC.exe", "1Password.exe", "Bitwarden.exe", "Dashlane.exe",
    "LastPass.exe", "NordPass.exe", "Enpass.exe", "RoboForm.exe", "SecHealthUI.exe",
    "CredentialUIBroker.exe", "consent.exe", "LogonUI.exe",
]  # fmt: skip
BLOCKED_TITLES = ("windows security",)
# Never typed into: these would run whatever Jarvis types.
DEFAULT_NO_TYPING = [
    "WindowsTerminal.exe", "powershell.exe", "pwsh.exe", "cmd.exe", "conhost.exe",
    "OpenConsole.exe", "wsl.exe", "bash.exe", "mintty.exe", "alacritty.exe", "wezterm-gui.exe",
    "SearchHost.exe", "SearchApp.exe", "StartMenuExperienceHost.exe", "Taskmgr.exe",
]  # fmt: skip

KEY_ALIASES = {
    "control": "ctrl", "ctl": "ctrl", "cmd": "win", "windows": "win", "super": "win",
    "meta": "win", "return": "enter", "esc": "escape", "del": "delete", "ins": "insert",
    "pgup": "page_up", "pageup": "page_up", "pgdn": "page_down", "pagedown": "page_down",
    "option": "alt", "spacebar": "space", "bksp": "backspace", "printscreen": "print_screen",
    "prtsc": "print_screen", "arrowup": "up", "arrowdown": "down", "arrowleft": "left",
    "arrowright": "right",
}  # fmt: skip
NAMED_KEYS = {
    "ctrl", "alt", "shift", "win", "enter", "escape", "tab", "space", "backspace", "delete",
    "insert", "home", "end", "page_up", "page_down", "up", "down", "left", "right",
    "print_screen", "caps_lock", "menu", *(f"f{i}" for i in range(1, 25)),
    "media_play_pause", "media_next", "media_previous", "media_volume_up",
    "media_volume_down", "media_volume_mute",
}  # fmt: skip
# Keys that close, quit or delete → always asked on their own (any order of keys).
DANGEROUS_COMBOS = {
    frozenset(c)
    for c in (
        ("alt", "f4"), ("ctrl", "w"), ("ctrl", "f4"), ("ctrl", "q"), ("ctrl", "shift", "w"),
        ("ctrl", "shift", "q"), ("shift", "delete"), ("ctrl", "shift", "delete"), ("delete",),
        ("ctrl", "d"), ("enter",), ("ctrl", "enter"), ("shift", "enter"), ("alt", "s"),
    )
}  # fmt: skip
# Clicking these always asks on its own too (sending, paying, deleting, closing…).
RISKY_WORDS = re.compile(
    r"\b(close|delete|remove|discard|erase|uninstall|format|reset|send|post|publish|reply|"
    r"pay|buy|purchase|order|checkout|check out|subscribe|transfer|confirm|sign out|log out|"
    r"logout|submit|share|empty (the )?(recycle )?bin)\b",
    re.I,
)
WINDOW_ACTIONS = ("focus", "minimize", "maximize", "restore", "close", "left_half", "right_half")
CALL_TIMEOUT_S = 15.0
TYPE_CHUNK = 16  # characters typed between safety checks


class DesktopError(ToolError):
    pass


def parse_keys(spec: str) -> list[str]:
    """ "Ctrl + Shift + S" → ["ctrl", "shift", "s"]."""
    parts = [p.strip().lower().replace(" ", "_") for p in re.split(r"\s*\+\s*", spec.strip())]
    out = []
    for p in parts:
        if not p:
            raise DesktopError(f"Can't read the keys {spec!r}. Use e.g. 'ctrl+s'.")
        p = KEY_ALIASES.get(p, p)
        if p not in NAMED_KEYS and len(p) != 1:
            raise DesktopError(f"Unknown key {p!r} in {spec!r}.")
        out.append(p)
    return out


def is_dangerous(keys: list[str]) -> bool:
    return frozenset(keys) in DANGEROUS_COMBOS


def risky_label(text: str | None) -> bool:
    return bool(text and RISKY_WORDS.search(text))


class _Worker:
    """One daemon thread running backend calls in order (a hung one can be abandoned
    without blocking Jarvis from exiting)."""

    def __init__(self, initializer: Callable[[], None] | None):
        self._q: queue.Queue = queue.Queue()
        self._init = initializer
        self.thread = threading.Thread(target=self._run, name="desktop", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        keep = None  # the COM initializer lives exactly as long as this thread
        if self._init:
            try:
                keep = self._init()
            except Exception:
                log.exception("desktop worker init failed")
        self._keep = keep
        while True:
            item = self._q.get()
            if item is None:
                return
            fn, args, fut = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn(*args))
            except BaseException as e:
                fut.set_exception(e)

    def submit(self, fn, *args) -> Future:
        fut: Future = Future()
        self._q.put((fn, args, fut))
        return fut

    def stop(self, *, fail_pending: bool = False) -> None:
        if fail_pending:  # abandoned: whatever was queued behind the stuck call fails now
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    item[2].set_exception(DesktopError("cancelled: Windows didn't respond"))
        self._q.put(None)


class Desktop:
    def __init__(
        self,
        backend: DesktopBackend,
        vision: Vision | None,
        guard: Guard | None = None,
        *,
        blocked_apps: list[str] | None = None,
        no_typing_apps: list[str] | None = None,
        max_side: int = 1280,
        element_limit: int = 80,
        settle_s: float = 0.35,
        run_inline: bool = False,
        initializer: Callable[[], None] | None = None,
        own_windows: set[int] | None = None,
    ):
        self.backend = backend
        self.vision = vision
        self.guard = guard or Guard()
        blocked = DEFAULT_BLOCKED if blocked_apps is None else blocked_apps
        self.blocked = {a.lower() for a in blocked}
        no_typing = DEFAULT_NO_TYPING if no_typing_apps is None else no_typing_apps
        self.no_typing = {a.lower() for a in no_typing}
        self.max_side = max_side
        self.element_limit = element_limit
        self.settle_s = settle_s
        self._elements: dict[int, Element] = {}
        self._elements_window: Window | None = None
        # Jarvis's own terminal: "the active window" means the app behind it.
        self.own_windows = set(own_windows or ())
        self._inline = run_inline
        self._initializer = initializer
        self._worker = None if run_inline else _Worker(initializer)
        self._turns = 0  # replies in progress (serve mode can overlap them)
        self._turn_lock = threading.Lock()

    # ------------------------------------------------------------ plumbing

    async def _call(self, fn, *args):
        if self._inline:
            return fn(*args)
        fut = self._worker.submit(fn, *args)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(fut), CALL_TIMEOUT_S)
        except TimeoutError:
            # An app isn't responding and took the worker with it: start a fresh one.
            log.warning("desktop call %s timed out; replacing the worker", fn.__name__)
            self._worker.stop(fail_pending=True)
            self._worker = _Worker(self._initializer)
            self._elements.clear()
            raise DesktopError(
                "Windows didn't respond (an app may be frozen). Try again or ask the user."
            ) from None

    def new_turn(self) -> None:
        """A reply starts: a new task. Clears an Esc-Esc stop — unless another reply is
        still working (then its stop stays in force)."""
        with self._turn_lock:
            if self._turns == 0:
                self.guard.reset()
            self._turns += 1

    def end_turn(self) -> None:
        with self._turn_lock:
            self._turns = max(0, self._turns - 1)

    def close(self) -> None:
        self.guard.stop_listening()
        if self._worker is not None:
            self._worker.stop()

    async def _check(self) -> None:
        try:
            self.guard.check(await self._call(self.backend.cursor))
        except DesktopStopped as e:
            raise DesktopError(
                f"Stopped: {e}. Don't continue this task — tell the user you stopped."
            ) from e

    def _is_blocked(self, w: Window | None) -> bool:
        return w is not None and (
            w.process.lower() in self.blocked or w.title.lower().startswith(BLOCKED_TITLES)
        )

    def _refuse_blocked(self, w: Window | None) -> None:
        if self._is_blocked(w):
            raise DesktopError(
                f"{w.process or w.title} is protected (passwords / security). Jarvis doesn't "
                "look at or control it — the user has to do this part themselves."
            )

    def _no_typing(self, w: Window | None) -> bool:
        if w is None:
            return False
        run_dialog = w.cls == "#32770" and w.title.strip().lower() == "run"
        return w.process.lower() in self.no_typing or run_dialog

    async def _active(self) -> Window | None:
        """The window you're working in — skipping Jarvis's own terminal."""
        w = await self._call(self.backend.active_window)
        if w is None or w.handle not in self.own_windows:
            return w
        for other in await self.windows():  # top of the z-order first
            if other.handle not in self.own_windows and not other.minimized:
                return other
        return w

    async def _settle(self) -> str:
        await asyncio.sleep(self.settle_s)
        w = await self._call(self.backend.active_window)
        return f"Active window now: “{w.title}”." if w else ""

    # ------------------------------------------------------------ windows

    async def windows(self) -> list[Window]:
        return await self._call(self.backend.windows)

    async def find_window(self, name: str | None) -> Window:
        """None/"active" → the active window; else best title/app match."""
        if not name or name.strip().lower() in ("active", "current", "this", "foreground"):
            w = await self._active()
            if w is None:
                raise DesktopError("There's no active window.")
            return w
        want = name.strip().lower().removesuffix(".exe")
        wins = await self.windows()

        def score(w: Window) -> int:
            title, app = w.title.lower(), w.app.lower()
            if want == app or want == title:
                return 4
            if app.startswith(want) or want in app:
                return 3
            if want in title:
                return 2
            return 0

        ranked = sorted((w for w in wins if score(w)), key=lambda w: (-score(w), w.minimized))
        if not ranked:
            names = ", ".join(sorted({w.app or w.title for w in wins})[:15])
            raise DesktopError(f"No open window matches {name!r}. Open windows: {names}")
        return ranked[0]

    async def window_do(self, name: str, action: str) -> str:
        if action not in WINDOW_ACTIONS:
            raise DesktopError(f"Unknown window action {action!r}; use one of {WINDOW_ACTIONS}")
        w = await self.find_window(name)
        self._refuse_blocked(w)
        await self._check()
        await self._call(self.backend.window_action, w, action)
        return f"{action.replace('_', ' ')}: {w.label()}. {await self._settle()}".strip()

    @staticmethod
    def _belongs(x: Window | None, w: Window) -> bool:
        """x is w, or one of w's own dialogs / menus / popups."""
        return x is not None and (x.handle == w.handle or x.owner == w.handle)

    async def _front(self, w: Window | None) -> None:
        """Input lands on whatever is on top: bring the target window up first — and
        make sure it actually came up (Windows sometimes refuses)."""
        if w is None:
            return
        if not await self._call(self.backend.responding, w):
            raise DesktopError(f"“{w.title}” isn't responding.")
        fg = await self._call(self.backend.active_window)
        if self._belongs(fg, w):
            return
        await self._call(self.backend.window_action, w, "focus")
        await asyncio.sleep(0.25)
        fg = await self._call(self.backend.active_window)
        if not self._belongs(fg, w):
            raise DesktopError(
                f"Windows wouldn't bring “{w.title}” to the front. Ask the user to click on it."
            )

    # ------------------------------------------------------------ looking

    async def _shot(self, where: str, window: str | None = None):
        w = await self.find_window(window)
        self._refuse_blocked(w)
        hide = [  # every visible window of a protected app, popups included
            o
            for o in await self._call(self.backend.visible_windows)
            if self._is_blocked(o) and o.handle != w.handle
        ]
        shot = await self._call(self.backend.screenshot, w, where == "screen", self.max_side, hide)
        return w, shot

    async def look(self, question: str, where: str = "window", window: str | None = None) -> str:
        if self.vision is None:
            raise DesktopError("No vision model is set up ([models] roles.vision).")
        w, shot = await self._shot(where, window)
        answer = await self.vision.look(shot.image_b64, question)
        place = f"the screen (active window “{w.title}”)" if where == "screen" else f"“{w.title}”"
        return f"Looked at {place}: {answer}"

    async def locate(
        self, description: str, window: str | None = None
    ) -> tuple[int, int, str, Window]:
        """Screen coordinates of something described in words (via the vision model)."""
        if self.vision is None:
            raise DesktopError("No vision model is set up ([models] roles.vision).")
        w, shot = await self._shot("window", window)
        found = await self.vision.locate(shot.image_b64, description)
        if found is None:
            raise DesktopError(
                f"Couldn't find {description!r} on screen. Try ui_elements, or describe it "
                "differently."
            )
        fx, fy, what = found
        r = shot.region
        return round(r.left + fx * r.width), round(r.top + fy * r.height), what, w

    # ------------------------------------------------------------ UI elements

    async def elements(self, window: str | None = None) -> list[Element]:
        w = await self.find_window(window)
        self._refuse_blocked(w)
        if not await self._call(self.backend.responding, w):
            raise DesktopError(f"“{w.title}” isn't responding.")
        els = await self._call(self.backend.elements, w, self.element_limit)
        self._elements = {}
        for i, el in enumerate(els, start=1):
            el.id = i
            self._elements[i] = el
        self._elements_window = w
        return els

    def element(self, eid: int) -> Element:
        el = self._elements.get(eid)
        if el is None:
            raise DesktopError(f"No element [{eid}] — call ui_elements first (ids change).")
        return el

    async def click_element(self, eid: int, *, double: bool = False, button: str = "left") -> str:
        el = self.element(eid)
        self._refuse_blocked(self._elements_window)
        await self._check()
        fresh = await self._call(self.backend.refresh, el)  # the screen may have changed
        if fresh is None:
            raise DesktopError(f"{el.label()} has changed or gone — call ui_elements again.")
        if fresh.rect.empty:
            # Off-screen: use the control's own action instead of the mouse, if it has one.
            if not double and button == "left" and await self._call(self.backend.invoke, fresh):
                return f"Clicked {el.label()}. {await self._settle()}".strip()
            raise DesktopError(f"{el.label()} isn't visible on screen right now.")
        x, y = fresh.rect.center
        return await self._click(
            x,
            y,
            button,
            2 if double else 1,
            el.label(),
            self._elements_window,
            risky_ok=risky_label(el.name),  # it was asked about by name
        )

    # ------------------------------------------------------------ mouse

    async def _click(
        self,
        x: int,
        y: int,
        button: str,
        count: int,
        what: str,
        w: Window | None,
        *,
        risky_ok: bool = False,
    ) -> str:
        await self._check()
        if not await self._call(self.backend.on_screen, x, y):
            raise DesktopError(f"({x}, {y}) is off the screen.")
        await self._front(w)
        # The window under the point must be the one we checked (not a popup on top,
        # a protected app or a terminal).
        under = await self._call(self.backend.window_at, x, y)
        if under is None or (w is not None and not self._belongs(under, w)):
            name = f"“{under.title}”" if under else "nothing"
            raise DesktopError(f"Something else is in the way at ({x}, {y}): {name}.")
        self._refuse_blocked(under)
        if self._no_typing(under) and button == "right":
            raise DesktopError("Jarvis doesn't right-click in terminals (it would paste).")
        if not risky_ok:
            # Whatever is really under the pointer: a "Send"/"Delete"/"Pay" button needs
            # its own approval (ui_click on it, or mouse_click naming it).
            hit = await self._call(self.backend.element_at, x, y)
            if hit is not None and risky_label(hit.name):
                raise DesktopError(
                    f"That's the “{hit.name}” {hit.kind.lower()} — click it with ui_click "
                    "(or name it in mouse_click) so the user can approve it."
                )
        await self._check()
        await self._call(self.backend.click, x, y, button, count)
        self.guard.moved_to(x, y)
        verb = {1: "Clicked", 2: "Double-clicked"}.get(count, "Clicked")
        if button == "right":
            verb = "Right-clicked"
        return f"{verb} {what} at ({x}, {y}). {await self._settle()}".strip()

    async def click_at(
        self,
        target: str | None = None,
        x: int | None = None,
        y: int | None = None,
        *,
        button: str = "left",
        double: bool = False,
    ) -> str:
        if target:
            x, y, what, w = await self.locate(target)
        elif x is not None and y is not None:
            what, w = "the screen", await self._active()
        else:
            raise DesktopError("Say what to click (target) or give x and y.")
        self._refuse_blocked(w)
        return await self._click(
            int(x), int(y), button, 2 if double else 1, what, w, risky_ok=risky_label(target)
        )

    async def scroll(self, direction: str, amount: int = 5, target: str | None = None) -> str:
        d = direction.lower()
        dx, dy = {"up": (0, amount), "down": (0, -amount), "left": (-amount, 0),
                  "right": (amount, 0)}.get(d, (None, None))  # fmt: skip
        if dx is None:
            raise DesktopError("direction must be up, down, left or right")
        if target:
            x, y, _, w = await self.locate(target)
        else:
            w = await self.find_window(None)
            x, y = w.rect.center
        self._refuse_blocked(w)
        await self._check()
        await self._front(w)
        under = await self._call(self.backend.window_at, x, y)
        if under is None or not self._belongs(under, w):
            raise DesktopError("Something else is covering that window.")
        await self._call(self.backend.scroll, x, y, dx, dy)
        self.guard.moved_to(x, y)
        return f"Scrolled {d}. {await self._settle()}".strip()

    # ------------------------------------------------------------ keyboard

    async def _typing_ok(self) -> Window | None:
        """Keys go to the real foreground window: never protected apps, terminals, the
        Run dialog, Start search — or a password field."""
        w = await self._call(self.backend.active_window)
        self._refuse_blocked(w)
        if self._no_typing(w):
            raise DesktopError(
                f"Jarvis doesn't type into {w.title or w.process} — that would run commands."
            )
        focused = await self._call(self.backend.focused_element)
        if focused is not None:
            if focused.is_password:
                raise DesktopError(
                    "The cursor is in a password field. Jarvis never types passwords — "
                    "ask the user to."
                )
            if "terminal" in focused.name.lower():  # e.g. VS Code's built-in terminal
                raise DesktopError("Jarvis doesn't type into terminals.")
        return w

    async def type_text(
        self, text: str, eid: int | None = None, *, enter: bool = False, replace: bool = False
    ) -> str:
        if "\t" in text:
            raise DesktopError("Type one field at a time (no Tab characters).")
        await self._check()
        el = None
        if eid is not None:
            el = self.element(eid)
            if el.is_password:
                raise DesktopError(
                    "That's a password field. Jarvis never types passwords — ask the user to."
                )
            self._refuse_blocked(self._elements_window)
            await self._front(self._elements_window)
            await self._call(self.backend.focus_element, el)
            await asyncio.sleep(0.1)
        else:
            await self._front(await self._active())  # not Jarvis's own terminal
        target = await self._typing_ok()
        if replace:
            await self._call(self.backend.keys, ["ctrl", "a"])
        # Type in small pieces, re-checking everything in between: Esc Esc works mid-way,
        # and if focus jumps (a popup, a password box) typing stops.
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for i, line in enumerate(lines):
            for start in range(0, len(line), TYPE_CHUNK):
                await self._check()
                now = await self._typing_ok()
                if target is not None and (now is None or now.handle != target.handle):
                    raise DesktopError("The window changed while typing, so I stopped.")
                await self._call(self.backend.type_text, line[start : start + TYPE_CHUNK])
            if i < len(lines) - 1:
                await self._check()
                await self._typing_ok()
                await self._call(self.backend.keys, ["shift", "enter"])  # a new line
        if enter:
            await self._check()
            await self._call(self.backend.keys, ["enter"])
        where = f" into {el.label()}" if el is not None else ""
        shown = text if len(text) <= 60 else text[:57] + "…"
        tail = " and pressed Enter" if enter else ""
        return f"Typed “{shown}”{where}{tail}. {await self._settle()}".strip()

    async def press(self, spec: str) -> str:
        keys = parse_keys(spec)
        await self._check()
        if keys[0] == "win":  # Windows shortcuts (Win+D, Win+←) work from anywhere
            self._refuse_blocked(await self._call(self.backend.active_window))
            if set(keys) == {"win", "r"}:
                raise DesktopError("Jarvis doesn't open the Run dialog.")
        else:
            await self._front(await self._active())  # the app, not Jarvis's terminal
            await self._typing_ok()
        await self._call(self.backend.keys, keys)
        return f"Pressed {'+'.join(keys)}. {await self._settle()}".strip()
