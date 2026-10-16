"""The off switch: Esc twice — or grabbing the mouse — stops Jarvis mid-task."""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


class DesktopStopped(Exception):
    pass


class Guard:
    def __init__(self, *, panic_window_s: float = 0.8, takeover_px: int = 40, clock=time.monotonic):
        self.panic_window_s = panic_window_s
        self.takeover_px = takeover_px
        self.clock = clock
        self._stopped = threading.Event()
        self.reason = ""
        self._last_esc = -1e9
        self._expected: tuple[int, int] | None = None
        self._listener = None

    # -- the panic key (Esc, Esc)

    def start(self) -> None:
        """Listen for Esc Esc anywhere in Windows (keys Jarvis itself sends are ignored)."""
        if self._listener is not None:
            return
        try:
            from pynput import keyboard
        except Exception as e:  # no display (Linux CI), etc.
            log.info("panic key unavailable: %s", e)
            return

        def on_press(key, injected=False):
            if not injected and key == keyboard.Key.esc:
                self.esc()

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    def stop_listening(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def esc(self) -> None:
        now = self.clock()
        if now - self._last_esc <= self.panic_window_s:
            self.stop("you pressed Esc twice")
        self._last_esc = now

    # -- state

    def stop(self, reason: str) -> None:
        if not self._stopped.is_set():
            self.reason = reason
            self._stopped.set()
            log.info("desktop control stopped: %s", reason)

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def reset(self) -> None:
        """A new task: clear the stop and forget where the cursor was."""
        self._stopped.clear()
        self.reason = ""
        self._expected = None

    def moved_to(self, x: int, y: int) -> None:
        self._expected = (x, y)

    def check(self, cursor: tuple[int, int] | None = None) -> None:
        """Before every action: stopped? did you take the mouse back?"""
        if self._stopped.is_set():
            raise DesktopStopped(self.reason)
        if cursor is not None and self._expected is not None:
            dx, dy = cursor[0] - self._expected[0], cursor[1] - self._expected[1]
            if dx * dx + dy * dy > self.takeover_px**2:
                self.stop("you moved the mouse")
                raise DesktopStopped(self.reason)
