"""Phase 5 — see the screen, control mouse & keyboard (Windows).

    tools (screen_look, ui_elements, ui_click, ui_type, mouse_click, keys_press, …)
      → Desktop service (safety checks, element ids, coordinate mapping)
          → DesktopBackend (WindowsBackend: mss, Windows UI Automation, pynput, user32)
          → Vision (a vision model, e.g. qwen3-vl, reads screenshots / finds things)
          → Guard (Esc Esc stops everything; moving the mouse yourself stops it)

Screenshots are only kept in memory long enough to show the vision model.
"""

from jarvis.desktop.model import Element, Rect, Window
from jarvis.desktop.service import Desktop, DesktopError

__all__ = ["Desktop", "DesktopError", "Element", "Rect", "Window"]
