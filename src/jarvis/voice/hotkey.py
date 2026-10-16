"""The push-to-talk key: works anywhere in Windows, even when the terminal isn't focused.

Two kinds of hotkey:
- a single key TAPPED on its own, e.g. "ctrl_l" (Left Ctrl). Holding it as part
  of a shortcut (Ctrl + C) or while clicking/scrolling (Ctrl + click) does NOT
  trigger — only a clean tap does.
- a combination, e.g. "<ctrl>+<alt>+j", triggered when pressed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

TAP_MAX_S = 0.6  # a longer press isn't a tap (you were probably doing something else)


def describe(spec: str) -> str:
    names = {
        "ctrl_r": "Right Ctrl",
        "ctrl": "Ctrl",
        "ctrl_l": "Left Ctrl",
        "alt_r": "Right Alt",
        "alt_gr": "AltGr",
        "shift_r": "Right Shift",
        "caps_lock": "Caps Lock",
        "scroll_lock": "Scroll Lock",
        "pause": "Pause",
        "insert": "Insert",
        "menu": "Menu",
    }
    if "+" in spec:
        return "+".join(p.strip("<>").capitalize() for p in spec.split("+"))
    return names.get(spec, spec.upper() if len(spec) <= 3 else spec.replace("_", " ").title())


class TapDetector:
    """Pure logic for "tapped on its own": feed key downs/ups, get True on a tap."""

    def __init__(self, target, *, max_s: float = TAP_MAX_S, clock=time.monotonic):
        self.target = target
        self.max_s = max_s
        self.clock = clock
        self._down_at: float | None = None
        self._clean = False

    def down(self, key) -> None:
        if key == self.target:
            if self._down_at is None:  # ignore auto-repeat while held
                self._down_at = self.clock()
                self._clean = True
        elif self._down_at is not None:
            self._clean = False  # another key while held: it's a shortcut, not a tap

    def other(self) -> None:
        """A mouse click/scroll while held (Ctrl + click): not a tap either."""
        if self._down_at is not None:
            self._clean = False

    def up(self, key) -> bool:
        if key != self.target or self._down_at is None:
            return False
        tapped = self._clean and self.clock() - self._down_at <= self.max_s
        self._down_at = None
        return tapped


class Hotkey:
    """Global hotkey via pynput. `callback()` runs on pynput's thread — keep it quick."""

    def __init__(self, spec: str, callback: Callable[[], None]):
        self.spec = spec.strip()
        self.callback = callback
        self._listener = None
        self._lock = threading.Lock()

    def _fire(self) -> None:
        try:
            self.callback()
        except Exception:
            log.exception("hotkey callback failed")

    def start(self) -> None:
        try:
            from pynput import keyboard
        except Exception as e:  # ImportError, or no display on Linux
            raise RuntimeError(f"global hotkeys aren't available here: {e}") from e

        if "+" in self.spec:
            hotkey = keyboard.HotKey(keyboard.HotKey.parse(self.spec), self._fire)
            listener = None

            def on_press(key):
                hotkey.press(listener.canonical(key))

            def on_release(key):
                hotkey.release(listener.canonical(key))

            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        else:
            name = self.spec.lower()
            target = (
                getattr(keyboard.Key, name)
                if hasattr(keyboard.Key, name)
                else keyboard.KeyCode.from_char(name)
            )
            tap = TapDetector(target)

            def on_press(key):
                tap.down(key)

            def on_release(key):
                if tap.up(key):
                    self._fire()

            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            try:  # Ctrl + click / Ctrl + scroll mustn't count as a tap
                from pynput import mouse

                self._mouse = mouse.Listener(
                    on_click=lambda *_a: tap.other(), on_scroll=lambda *_a: tap.other()
                )
                self._mouse.daemon = True
                self._mouse.start()
            except Exception:
                log.debug("mouse listener unavailable", exc_info=True)
        listener.daemon = True
        listener.start()
        self._listener = listener

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if getattr(self, "_mouse", None) is not None:
            self._mouse.stop()
            self._mouse = None
