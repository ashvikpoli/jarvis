"""Plain data the desktop layer passes around (no Windows imports here)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.right) // 2, (self.top + self.bottom) // 2

    @property
    def empty(self) -> bool:
        return self.width <= 1 or self.height <= 1

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom


@dataclass
class Window:
    handle: int
    title: str
    process: str  # e.g. "Discord.exe"
    rect: Rect
    active: bool = False
    minimized: bool = False
    cls: str = ""  # window class, e.g. "#32770" for dialogs like Run
    owner: int = 0  # the top-level window that owns this one (dialogs, menus); 0 = none

    @property
    def app(self) -> str:
        return self.process.rsplit(".", 1)[0] if self.process else ""

    def label(self) -> str:
        bits = [f"“{self.title}”"]
        if self.process:
            bits.append(f"({self.process})")
        if self.active:
            bits.append("— active")
        if self.minimized:
            bits.append("— minimized")
        return " ".join(bits)


@dataclass
class Element:
    """A control in a window, from Windows UI Automation."""

    kind: str  # Button, Edit, CheckBox, Hyperlink, MenuItem, ListItem, TabItem, …
    name: str
    rect: Rect
    value: str | None = None  # text in an edit box (never for password fields)
    is_password: bool = False
    enabled: bool = True
    checked: bool | None = None
    handle: Any = field(default=None, repr=False)  # backend object (UIA control)
    id: int = 0  # assigned per listing; what the model refers to

    def label(self) -> str:
        name = self.name.strip() or "(no name)"
        bits = [f"[{self.id}] {self.kind} “{name[:60]}”"]
        if self.is_password:
            bits.append("(password field — Jarvis never types here)")
        elif self.value:
            bits.append(f"= “{self.value[:60]}”")
        if self.checked is not None:
            bits.append("☑" if self.checked else "☐")
        if not self.enabled:
            bits.append("(disabled)")
        return " ".join(bits)


@dataclass
class Shot:
    """A screenshot: JPEG bytes as base64 + where on screen it was taken."""

    image_b64: str
    region: Rect  # screen coordinates of the captured area
    size: tuple[int, int]  # pixel size of the image sent to the model (after scaling)


class DesktopBackend(Protocol):
    """Everything that touches the real PC. WindowsBackend implements it; tests fake it."""

    def windows(self) -> list[Window]: ...
    def active_window(self) -> Window | None: ...
    # The window's pixels (even if other windows cover it), or the whole monitor it's on.
    # `hide`: windows whose pixels must never be captured (blocked apps) — blacked out.
    def screenshot(
        self, window: Window, whole_screen: bool, max_side: int, hide: list[Window]
    ) -> Shot: ...
    def window_at(self, x: int, y: int) -> Window | None: ...  # top-level window under a point
    def element_at(self, x: int, y: int) -> Element | None: ...  # the control under a point
    def visible_windows(self) -> list[Window]: ...  # every visible top-level window
    def on_screen(self, x: int, y: int) -> bool: ...
    def refresh(self, element: Element) -> Element | None: ...  # re-read; None = it's gone
    def responding(self, window: Window) -> bool: ...
    def elements(self, window: Window, limit: int) -> list[Element]: ...
    def focused_element(self) -> Element | None: ...
    def invoke(self, element: Element) -> bool: ...  # True = done without the mouse
    def focus_element(self, element: Element) -> None: ...
    def click(self, x: int, y: int, button: str, count: int) -> None: ...
    def type_text(self, text: str) -> None: ...
    def keys(self, combo: list[str]) -> None: ...
    def scroll(self, x: int, y: int, dx: int, dy: int) -> None: ...
    def cursor(self) -> tuple[int, int]: ...
    def window_action(self, window: Window, action: str) -> None: ...
