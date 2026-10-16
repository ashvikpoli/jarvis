"""Tools to see the screen and use the mouse & keyboard (Phase 5).

Looking is free (safe). Doing — clicking, typing, keys, scrolling, moving windows —
is one group: the first action of a task asks "control your mouse and keyboard for
this task?", the rest of that task doesn't. Closing windows and keys like Alt+F4
always ask on their own. Esc Esc stops everything.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from jarvis.desktop.service import WINDOW_ACTIONS, is_dangerous, parse_keys, risky_label
from jarvis.tools.base import Risk, Tool, ToolContext, ToolError

GROUP = "desktop"


def _desktop(ctx: ToolContext):
    d = getattr(ctx, "desktop", None)
    if d is None:
        raise ToolError("Screen control isn't available here (Windows only, [desktop] enabled).")
    return d


class LookArgs(BaseModel):
    question: str = Field(
        "Describe what's on the screen.",
        description="What you want to know about the screen, in plain words.",
    )
    where: Literal["window", "screen"] = Field(
        "window", description="window = the app the user is in; screen = the whole monitor"
    )
    window: str | None = Field(None, description="Another window by app/title, e.g. 'Chrome'")


class ScreenLook(Tool):
    name = "screen_look"
    description = (
        "Look at the user's screen and answer a question about it (read an error, summarize "
        "a page, describe what's shown). Use whenever they say 'this', 'my screen', 'what "
        "am I looking at'."
    )
    risk = Risk.SAFE
    Args = LookArgs

    async def run(self, args: LookArgs, ctx: ToolContext):
        return await _desktop(ctx).look(args.question, args.where, args.window)


class ElementsArgs(BaseModel):
    window: str | None = Field(None, description="App or title; omit for the active app")


class UiElements(Tool):
    name = "ui_elements"
    description = (
        "List the buttons, fields, links, tabs and menu items in a window, each with an id "
        "for ui_click / ui_type. Do this before clicking or typing in an app."
    )
    risk = Risk.SAFE
    Args = ElementsArgs

    async def run(self, args: ElementsArgs, ctx: ToolContext):
        d = _desktop(ctx)
        els = await d.elements(args.window)
        w = d._elements_window
        if not els:
            return (
                f"No controls found in “{w.title if w else 'the window'}” (some apps hide "
                "them). Use screen_look, or mouse_click with a target description."
            )
        return f"Window “{w.title}” ({w.process}):\n" + "\n".join(e.label() for e in els)


class UiClickArgs(BaseModel):
    id: int = Field(description="Element id from ui_elements")
    double: bool = False
    button: Literal["left", "right"] = "left"


class UiClick(Tool):
    name = "ui_click"
    description = "Click a button/link/tab/menu item by its id from ui_elements."
    risk = Risk.CHANGE
    group = GROUP
    Args = UiClickArgs

    @staticmethod
    def _el(args: UiClickArgs, ctx: ToolContext | None):
        d = getattr(ctx, "desktop", None)
        return d._elements.get(args.id) if d is not None else None

    def risk_in(self, args: UiClickArgs, ctx: ToolContext | None) -> Risk:
        el = self._el(args, ctx)  # "Send", "Delete", "Pay"… always ask on their own
        return Risk.DESTRUCTIVE if el is not None and risky_label(el.name) else Risk.CHANGE

    def describe_in(self, args: UiClickArgs, ctx: ToolContext | None) -> str:
        el = self._el(args, ctx)
        return f"click {el.label() if el else f'element [{args.id}]'}"

    async def run(self, args: UiClickArgs, ctx: ToolContext):
        return await _desktop(ctx).click_element(args.id, double=args.double, button=args.button)


class UiTypeArgs(BaseModel):
    text: str = Field(description="What to type")
    id: int | None = Field(
        None, description="Field id from ui_elements; omit = where the cursor is"
    )
    enter: bool = Field(False, description="Press Enter afterwards (send / submit)")
    replace: bool = Field(False, description="Replace what's in the field instead of adding")


class UiType(Tool):
    name = "ui_type"
    description = (
        "Type text into a field (by id from ui_elements) or where the cursor is. Never use "
        "for passwords or card numbers — ask the user to type those."
    )
    risk = Risk.CHANGE
    group = GROUP
    Args = UiTypeArgs

    def risk_for(self, args: UiTypeArgs) -> Risk:
        # Enter (or a new line, which some boxes treat as Enter) often sends/submits.
        sends = args.enter or "\n" in args.text or "\r" in args.text
        return Risk.DESTRUCTIVE if sends else Risk.CHANGE

    def describe(self, args: UiTypeArgs) -> str:
        where = f" into [{args.id}]" if args.id is not None else ""
        return f"type “{args.text[:60]}”{where}" + (" and press Enter" if args.enter else "")

    async def run(self, args: UiTypeArgs, ctx: ToolContext):
        return await _desktop(ctx).type_text(
            args.text, args.id, enter=args.enter, replace=args.replace
        )


class MouseClickArgs(BaseModel):
    target: str | None = Field(
        None, description="What to click, described (e.g. 'the blue Join button')"
    )
    x: int | None = Field(None, description="Screen x (only if you know it)")
    y: int | None = None
    button: Literal["left", "right"] = "left"
    double: bool = False


class MouseClick(Tool):
    name = "mouse_click"
    description = (
        "Click something on screen found by looking at it (for apps where ui_elements shows "
        "nothing useful). Prefer ui_click when the element is listed."
    )
    risk = Risk.CHANGE
    group = GROUP
    Args = MouseClickArgs

    def risk_for(self, args: MouseClickArgs) -> Risk:
        return Risk.DESTRUCTIVE if risky_label(args.target) else Risk.CHANGE

    def describe(self, args: MouseClickArgs) -> str:
        what = args.target or f"({args.x}, {args.y})"
        return f"{'double-' if args.double else ''}{args.button}-click {what}"

    async def run(self, args: MouseClickArgs, ctx: ToolContext):
        return await _desktop(ctx).click_at(
            args.target, args.x, args.y, button=args.button, double=args.double
        )


class KeysArgs(BaseModel):
    keys: str = Field(description="A shortcut like 'ctrl+s', 'alt+tab', 'win+d', 'enter'")


class KeysPress(Tool):
    name = "keys_press"
    description = (
        "Press a key or shortcut in the active app (ctrl+s save, ctrl+t new tab, win+d show "
        "desktop, media_play_pause …)."
    )
    risk = Risk.CHANGE
    group = GROUP
    Args = KeysArgs

    def risk_for(self, args: KeysArgs) -> Risk:
        try:
            return Risk.DESTRUCTIVE if is_dangerous(parse_keys(args.keys)) else Risk.CHANGE
        except ToolError:
            return Risk.CHANGE

    def describe(self, args: KeysArgs) -> str:
        return f"press {args.keys}"

    async def run(self, args: KeysArgs, ctx: ToolContext):
        return await _desktop(ctx).press(args.keys)


class ScrollArgs(BaseModel):
    direction: Literal["up", "down", "left", "right"] = "down"
    amount: int = Field(5, ge=1, le=50, description="Notches (~3 lines each)")
    target: str | None = Field(None, description="Where to scroll, described; omit = window")


class Scroll(Tool):
    name = "scroll"
    description = "Scroll the active window (or a described part of it)."
    risk = Risk.CHANGE
    group = GROUP
    Args = ScrollArgs

    def describe(self, args: ScrollArgs) -> str:
        return f"scroll {args.direction}"

    async def run(self, args: ScrollArgs, ctx: ToolContext):
        return await _desktop(ctx).scroll(args.direction, args.amount, args.target)


class WindowsList(Tool):
    name = "windows_list"
    description = "List the open windows (title, app, which one is active)."
    risk = Risk.SAFE

    async def run(self, args, ctx: ToolContext):
        wins = await _desktop(ctx).windows()
        return "\n".join(w.label() for w in wins) or "No windows open."


class WindowArgs(BaseModel):
    window: str = Field(description="App or title, e.g. 'Spotify'")
    action: Literal[WINDOW_ACTIONS] = Field(description="What to do with it")  # type: ignore[valid-type]


class WindowDo(Tool):
    name = "window_do"
    description = (
        "Focus, minimize, maximize, restore, close a window, or snap it to the left/right half."
    )
    risk = Risk.CHANGE
    group = GROUP
    Args = WindowArgs

    def risk_for(self, args: WindowArgs) -> Risk:
        return Risk.DESTRUCTIVE if args.action == "close" else Risk.CHANGE

    def describe(self, args: WindowArgs) -> str:
        return f"{args.action.replace('_', ' ')} the {args.window} window"

    async def run(self, args: WindowArgs, ctx: ToolContext):
        return await _desktop(ctx).window_do(args.window, args.action)


DESKTOP_TOOLS = [
    ScreenLook,
    UiElements,
    UiClick,
    UiType,
    MouseClick,
    KeysPress,
    Scroll,
    WindowsList,
    WindowDo,
]
