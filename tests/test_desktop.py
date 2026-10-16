"""Phase 5 — seeing the screen and using mouse & keyboard, against a fake desktop."""

from __future__ import annotations

import json

import pytest

from jarvis.core.types import Message, ToolCall
from jarvis.desktop import Desktop, DesktopError, Element, Rect, Window
from jarvis.desktop.guard import Guard
from jarvis.desktop.model import Shot
from jarvis.desktop.service import is_dangerous, parse_keys
from jarvis.desktop.vision import Vision
from jarvis.providers.ollama import _to_ollama
from jarvis.tools import ToolError
from jarvis.tools.desktop import KeysArgs, KeysPress, WindowArgs, WindowDo

TERMINAL = Window(1, "Jarvis — PowerShell", "WindowsTerminal.exe", Rect(0, 0, 800, 600), True)
DISCORD = Window(2, "#general - Discord", "Discord.exe", Rect(100, 100, 1100, 900))
SPOTIFY = Window(3, "Spotify Premium", "Spotify.exe", Rect(0, 0, 1920, 1080), minimized=True)
KEEPASS = Window(4, "Database.kdbx - KeePass", "KeePass.exe", Rect(0, 0, 500, 400))
OTHER_SHELL = Window(5, "Windows PowerShell", "powershell.exe", Rect(0, 0, 600, 400))


class FakeBackend:
    def __init__(self, windows=None):
        self._windows = list(windows or [TERMINAL, DISCORD, SPOTIFY, KEEPASS])
        self.fg = self._windows[0].handle
        self.actions: list[tuple] = []
        self.pos = (5, 5)
        self.focused: Element | None = None
        self.cover: Window | None = None  # a window pretending to sit on top of everything
        self.gone: set[str] = set()  # elements that disappeared
        self.refuse_focus = False
        self.under: Element | None = None  # the control under the mouse (element_at)
        self.elements_by_window = {
            DISCORD.handle: [
                Element("Edit", "Message #general", Rect(120, 800, 1000, 840)),
                Element("Button", "Send", Rect(1010, 800, 1090, 840)),
                Element("Edit", "Password", Rect(0, 0, 10, 10), is_password=True),
                Element("Button", "Settings", Rect(0, 0, 0, 0)),  # off-screen
            ]
        }

    def windows(self):
        return [Window(**{**w.__dict__, "active": w.handle == self.fg}) for w in self._windows]

    def active_window(self):
        return next((w for w in self.windows() if w.handle == self.fg), None)

    def screenshot(self, window, whole_screen, max_side, hide):
        region = Rect(0, 0, 1920, 1080) if whole_screen else window.rect
        self.actions.append(("shot", window.process, whole_screen))
        self.hidden = [w.process for w in hide]
        return Shot("aW1n", region, (min(max_side, region.width), 100))

    def window_at(self, x, y):
        # whatever is in front at that point: the foreground window if it covers it
        wins = self.windows()
        fg = next((w for w in wins if w.handle == self.fg), None)
        if self.cover is not None:
            return self.cover
        if fg and fg.rect.contains(x, y) and fg.handle != 1:
            return fg
        return next((w for w in wins if w.rect.contains(x, y) and not w.minimized), None)

    def element_at(self, x, y):
        return self.under

    def visible_windows(self):
        return [w for w in self.windows() if not w.minimized]

    def on_screen(self, x, y):
        return 0 <= x < 1920 and 0 <= y < 1080

    def refresh(self, el):
        return None if el.name in self.gone else el

    def responding(self, window):
        return True

    def elements(self, window, limit):
        return [Element(**{**e.__dict__}) for e in self.elements_by_window.get(window.handle, [])]

    def focused_element(self):
        return self.focused

    def invoke(self, el):
        self.actions.append(("invoke", el.name))
        return True

    def focus_element(self, el):
        self.actions.append(("focus_el", el.name))
        self.focused = el

    def click(self, x, y, button, count):
        self.actions.append(("click", x, y, button, count))
        self.pos = (x, y)

    def type_text(self, text):
        self.actions.append(("type", text, self.fg))

    def keys(self, combo):
        self.actions.append(("keys", "+".join(combo), self.fg))

    def scroll(self, x, y, dx, dy):
        self.actions.append(("scroll", x, y, dx, dy))
        self.pos = (x, y)

    def cursor(self):
        return self.pos

    def window_action(self, window, action):
        self.actions.append(("window", window.process, action))
        if action == "focus" and not self.refuse_focus:
            self.fg = window.handle


class FakeVision:
    def __init__(self, where=(0.5, 0.25)):
        self.where = where
        self.asked: list[tuple] = []

    async def look(self, image_b64, question):
        self.asked.append(("look", image_b64, question))
        return "A Discord chat with a message box."

    async def locate(self, image_b64, description):
        self.asked.append(("locate", description))
        return None if self.where is None else (*self.where, description)


def make(backend=None, vision=..., **kw):
    backend = backend or FakeBackend()
    vision = FakeVision() if vision is ... else vision
    d = Desktop(backend, vision, Guard(), run_inline=True, settle_s=0, own_windows={1}, **kw)
    return d, backend


# ---------------------------------------------------------------- keys


def test_parse_keys():
    assert parse_keys("Ctrl + Shift + S") == ["ctrl", "shift", "s"]
    assert parse_keys("win+d") == ["win", "d"] and parse_keys("Return") == ["enter"]
    assert parse_keys("page down") == ["page_down"] and parse_keys("esc") == ["escape"]
    with pytest.raises(DesktopError):
        parse_keys("ctrl+banana")
    assert is_dangerous(["alt", "f4"]) and is_dangerous(["ctrl", "w"])
    assert not is_dangerous(["ctrl", "s"])
    assert KeysPress().risk_for(KeysArgs(keys="alt+f4")) == "destructive"
    assert KeysPress().risk_for(KeysArgs(keys="ctrl+s")) == "change"
    assert WindowDo().risk_for(WindowArgs(window="x", action="close")) == "destructive"


# ---------------------------------------------------------------- windows & looking


async def test_active_window_skips_jarvis_terminal_and_finds_by_name():
    d, b = make()
    assert (await d.find_window(None)).process == "Discord.exe"  # not Jarvis's terminal
    assert (await d.find_window("spotify")).handle == SPOTIFY.handle
    assert (await d.find_window("Discord.exe")).handle == DISCORD.handle
    with pytest.raises(DesktopError, match="No open window matches"):
        await d.find_window("photoshop")


async def test_look_uses_vision_and_never_protected_apps():
    d, b = make()
    out = await d.look("what's this?")
    assert "Discord chat" in out and "#general - Discord" in out
    assert d.vision.asked[0] == ("look", "aW1n", "what's this?")
    assert ("shot", "Discord.exe", False) in b.actions
    await d.look("summarize", where="screen")
    assert ("shot", "Discord.exe", True) in b.actions
    with pytest.raises(DesktopError, match="protected"):
        await d.look("show me", window="keepass")
    d2, _ = make(vision=None)
    with pytest.raises(DesktopError, match="vision model"):
        await d2.look("hi")


# ---------------------------------------------------------------- elements & clicks


async def test_elements_and_clicks():
    d, b = make()
    els = await d.elements("discord")
    assert [e.id for e in els] == [1, 2, 3, 4]
    assert els[0].label() == "[1] Edit “Message #general”"
    assert "password field" in els[2].label()
    out = await d.click_element(2)
    assert ("window", "Discord.exe", "focus") in b.actions  # brought to the front first
    assert ("click", 1050, 820, "left", 1) in b.actions and "Clicked [2] Button" in out
    await d.click_element(4)  # off-screen → the control's own action
    assert ("invoke", "Settings") in b.actions
    with pytest.raises(DesktopError, match="call ui_elements first"):
        await d.click_element(99)


async def test_click_by_description_maps_vision_coordinates():
    d, b = make(vision=FakeVision(where=(0.5, 0.25)))
    await d.click_at("the blue Join button")
    # Discord's window is 1000×800 at (100, 100): (0.5, 0.25) → (600, 300)
    assert ("click", 600, 300, "left", 1) in b.actions
    d2, _ = make(vision=FakeVision(where=None))
    with pytest.raises(DesktopError, match="Couldn't find"):
        await d2.click_at("a unicorn")


# ---------------------------------------------------------------- typing & keys


async def test_typing_goes_to_the_app_and_never_passwords_or_terminals():
    d, b = make()
    await d.elements("discord")
    out = await d.type_text("hello team", 1, enter=True)
    assert ("focus_el", "Message #general") in b.actions
    assert ("type", "hello team", DISCORD.handle) in b.actions
    assert ("keys", "enter", DISCORD.handle) in b.actions and "pressed Enter" in out
    with pytest.raises(DesktopError, match="password field"):
        await d.type_text("hunter2", 3)
    b.focused = Element("Edit", "pw", Rect(0, 0, 9, 9), is_password=True)
    with pytest.raises(DesktopError, match="password"):
        await d.type_text("hunter2")

    d3, b3 = make(FakeBackend([OTHER_SHELL, DISCORD]))
    d3.own_windows = set()
    with pytest.raises(DesktopError, match="run commands"):
        await d3.type_text("rm -rf /")
    with pytest.raises(DesktopError, match="run commands"):
        await d3.press("ctrl+c")
    await d3.press("win+d")  # Windows shortcuts are fine from anywhere
    assert ("keys", "win+d", OTHER_SHELL.handle) in b3.actions


async def test_press_brings_the_app_forward():
    d, b = make()
    await d.press("ctrl+s")
    assert b.actions[0] == ("window", "Discord.exe", "focus")
    assert ("keys", "ctrl+s", DISCORD.handle) in b.actions


# ---------------------------------------------------------------- the off switch


async def test_esc_esc_and_grabbing_the_mouse_stop_everything():
    now = [0.0]
    d, b = make()
    d.guard.clock = lambda: now[0]
    d.guard.esc()
    now[0] = 0.5
    d.guard.esc()  # twice within 0.8 s
    with pytest.raises(DesktopError, match="Esc twice"):
        await d.press("ctrl+s")
    d.new_turn()  # the next task may act again
    d.end_turn()
    await d.click_at(x=200, y=200)
    b.pos = (700, 700)  # you moved the mouse
    with pytest.raises(DesktopError, match="moved the mouse"):
        await d.click_at(x=220, y=220)
    d.new_turn()
    d.new_turn()  # a second reply running at the same time…
    d.end_turn()
    d.guard.stop("you pressed Esc twice")
    d.new_turn()  # …keeps the first one's stop in force
    assert d.guard.stopped
    d.end_turn()
    d.end_turn()
    d.new_turn()
    await d.click_at(x=220, y=220)
    now[0] = 10
    d.guard.esc()
    now[0] = 12
    d.guard.esc()  # too slow: not a double press
    assert not d.guard.stopped


async def test_window_actions():
    d, b = make()
    out = await d.window_do("spotify", "left_half")
    assert ("window", "Spotify.exe", "left_half") in b.actions and "left half" in out
    with pytest.raises(DesktopError, match="Unknown window action"):
        await d.window_do("spotify", "explode")
    with pytest.raises(DesktopError, match="protected"):
        await d.window_do("keepass", "close")


# ---------------------------------------------------------------- vision model


async def test_vision_sends_the_image_and_reads_coordinates(make_app):
    app, provider = make_app(["It's a login page.", json.dumps({"found": True, "x": 250,
                              "y": 900, "what": "Sign in"}), '{"found": false}',
                              '{"found": true, "x": 4000, "y": 1}'])  # fmt: skip
    v = Vision(app.models, role="chat")
    assert await v.look("abc", "what's this?") == "It's a login page."
    sent = provider.calls[0]["messages"][-1]
    assert sent.images == ["abc"] and _to_ollama(sent)["images"] == ["abc"]
    assert await v.locate("abc", "sign in") == (0.25, 0.9, "Sign in")
    assert await v.locate("abc", "x") is None
    assert await v.locate("abc", "x") is None  # out of range
    assert "images" not in _to_ollama(Message.user("hi"))


# ---------------------------------------------------------------- one approval per task


def _desktop_app(make_app, turns):
    app, provider = make_app(turns, config="[desktop]\nenabled = true\n")
    d, b = make()
    app.desktop = d
    return app, provider, b


async def test_one_approval_per_task(make_app):
    turns = [
        [ToolCall(name="ui_elements", arguments={"window": "discord"})],
        [ToolCall(name="ui_click", arguments={"id": 1})],
        [ToolCall(name="ui_type", arguments={"text": "hi", "id": 1, "enter": True})],
        [ToolCall(name="window_do", arguments={"window": "discord", "action": "close"})],
        "Done.",
        [ToolCall(name="keys_press", arguments={"keys": "ctrl+s"})],
        "Saved.",
    ]
    app, provider, b = _desktop_app(make_app, turns)
    asked: list[str] = []

    async def approver(req):
        asked.append(req.description)
        return "once"

    ctx = app.tool_context()
    async for _ in app.agent_stream([Message.user("say hi in discord")], ctx, approver):
        pass
    # once for the task; Enter (sends) and closing a window ask on their own
    assert len(asked) == 3
    assert asked[0].startswith("control your mouse and keyboard for this task — first: click")
    assert "press Enter" in asked[1] and "close the discord window" in asked[2]
    assert ("type", "hi", DISCORD.handle) in b.actions
    async for _ in app.agent_stream([Message.user("save it")], ctx, approver):
        pass
    assert len(asked) == 4  # a new task asks again


async def test_always_for_session_covers_the_group(make_app):
    turns = [
        [ToolCall(name="keys_press", arguments={"keys": "ctrl+s"})],
        "ok",
        [ToolCall(name="scroll", arguments={"direction": "down"})],
        "ok",
    ]
    app, provider, b = _desktop_app(make_app, turns)
    asked = []

    async def approver(req):
        asked.append(req.description)
        return "session"

    ctx = app.tool_context()
    for text in ("save", "scroll"):
        async for _ in app.agent_stream([Message.user(text)], ctx, approver):
            pass
    assert len(asked) == 1 and any(a[0] == "scroll" for a in b.actions)


async def test_desktop_off_or_unavailable(make_app):
    app, _ = make_app()  # tests turn it off by default
    assert app.registry.get("ui_click") is None and app.registry.get("screen_look") is None
    app2, _ = make_app(config="[desktop]\nenabled = true\n")
    assert app2.registry.get("screen_look") is not None and app2.desktop is None  # not Windows
    from jarvis.tools.desktop import LookArgs, ScreenLook

    with pytest.raises(ToolError, match="Windows only"):
        await ScreenLook().run(LookArgs(), app2.tool_context())


# ---------------------------------------------------------------- review fixes


def test_dangerous_keys_any_order():
    for combo in ("f4+alt", "shift+ctrl+w", "ctrl+f4", "ctrl+shift+q", "delete"):
        assert is_dangerous(parse_keys(combo)), combo


async def test_risky_clicks_ask_on_their_own(make_app):
    from jarvis.tools.desktop import MouseClick, MouseClickArgs, UiClick, UiClickArgs

    app, provider, b = _desktop_app(make_app, [])
    d = app.desktop
    await d.elements("discord")
    ctx = app.tool_context()
    assert UiClick().risk_in(UiClickArgs(id=2), ctx) == "destructive"  # "Send"
    assert UiClick().risk_in(UiClickArgs(id=1), ctx) == "change"
    assert MouseClick().risk_for(MouseClickArgs(target="the Buy now button")) == "destructive"
    assert MouseClick().risk_for(MouseClickArgs(target="the search box")) == "change"


async def test_click_refuses_if_something_else_is_on_top():
    d, b = make()
    await d.elements("discord")
    b.cover = KEEPASS
    with pytest.raises(DesktopError, match="in the way"):
        await d.click_element(1)
    b.cover = None
    b.gone = {"Send"}
    with pytest.raises(DesktopError, match="changed or gone"):
        await d.click_element(2)
    with pytest.raises(DesktopError, match="off the screen"):
        await d.click_at(x=5000, y=10)
    b.refuse_focus, b.fg = True, TERMINAL.handle
    with pytest.raises(DesktopError, match="wouldn't bring"):
        await d.click_element(1)


async def test_whole_screen_hides_protected_apps():
    d, b = make()
    await d.look("what's up", where="screen")
    assert b.hidden == ["KeePass.exe"]


async def test_typing_in_pieces_stops_when_asked_and_on_focus_jumps():
    d, b = make()
    await d.elements("discord")
    with pytest.raises(DesktopError, match="Tab"):
        await d.type_text("alice\thunter2", 1)
    await d.type_text("line one\nline two", 1)
    typed = [a for a in b.actions if a[0] in ("type", "keys")]
    assert ("keys", "shift+enter", DISCORD.handle) in typed  # a new line, not "send"
    assert "".join(a[1] for a in typed if a[0] == "type") == "line oneline two"

    long = "x" * 100
    calls = {"n": 0}
    real = b.type_text

    def typing(text):
        calls["n"] += 1
        if calls["n"] == 2:
            d.guard.stop("you pressed Esc twice")
        real(text)

    b.type_text = typing
    b.actions.clear()
    with pytest.raises(DesktopError, match="Esc twice"):
        await d.type_text(long, 1)
    assert sum(len(a[1]) for a in b.actions if a[0] == "type") < 100  # stopped mid-way


async def test_no_typing_in_run_dialog_or_terminals_in_editors():
    run = Window(9, "Run", "explorer.exe", Rect(0, 0, 400, 200), cls="#32770")
    d, b = make(FakeBackend([run, DISCORD]))
    d.own_windows = set()
    with pytest.raises(DesktopError, match="Run"):
        await d.type_text("cmd /c whoami")
    with pytest.raises(DesktopError, match="Run dialog"):
        await d.press("win+r")
    d2, b2 = make()
    b2.focused = Element("Edit", "Terminal 1, powershell", Rect(0, 0, 9, 9))
    with pytest.raises(DesktopError, match="terminals"):
        await d2.type_text("ls")


async def test_explicit_ask_setting_beats_the_task_grant(make_app):
    turns = [
        [ToolCall(name="keys_press", arguments={"keys": "ctrl+s"})],
        [ToolCall(name="scroll", arguments={"direction": "down"})],
        "ok",
    ]
    app, _ = make_app(turns, config="[desktop]\nenabled = true\n"
                      "[tools.permissions.overrides]\nscroll = \"ask\"\n")  # fmt: skip
    app.desktop, b = make()
    asked = []

    async def approver(req):
        asked.append(req.description)
        return "once"

    async for _ in app.agent_stream([Message.user("save")], app.tool_context(), approver):
        pass
    assert len(asked) == 2  # scroll asks again because you set it to "ask"


# ---------------------------------------------------------------- review round 2


async def test_popups_and_dialogs_of_the_app_count_as_the_app():
    d, b = make()
    await d.elements("discord")
    menu = Window(20, "", "Discord.exe", Rect(100, 100, 400, 400), owner=DISCORD.handle)
    b.cover = menu  # a context menu / popup of Discord on top: fine
    await d.click_element(1)
    # A modal dialog of Discord holds the focus: that's still Discord.
    dialog = Window(21, "Upload", "Discord.exe", Rect(200, 200, 600, 500), owner=DISCORD.handle)
    b._windows.append(dialog)
    b.cover, b.fg, b.refuse_focus = None, dialog.handle, True
    await d._front(DISCORD)


async def test_raw_clicks_on_risky_controls_need_their_own_ok():
    d, b = make()
    b.fg = DISCORD.handle
    b.under = Element("Button", "Send", Rect(1010, 800, 1090, 840))
    with pytest.raises(DesktopError, match="ui_click"):
        await d.click_at(x=1050, y=820)
    assert not [a for a in b.actions if a[0] == "click"]
    await d.click_at("the Send button")  # named → it was approved on its own
    assert [a for a in b.actions if a[0] == "click"]


async def test_masking_covers_every_visible_window_of_protected_apps():
    d, b = make()
    popup = Window(30, "Unlock", "KeePass.exe", Rect(0, 0, 300, 200), owner=KEEPASS.handle)
    b.visible_windows = lambda: [*FakeBackend.visible_windows(b), popup]
    await d.look("what's up", where="screen")
    assert b.hidden == ["KeePass.exe", "KeePass.exe"]


def test_enter_and_new_lines_always_ask():
    from jarvis.tools.desktop import UiType, UiTypeArgs

    for k in ("enter", "ctrl+enter", "shift+enter", "alt+s"):
        assert KeysPress().risk_for(KeysArgs(keys=k)) == "destructive", k
    assert UiType().risk_for(UiTypeArgs(text="hi\nthere")) == "destructive"
    assert UiType().risk_for(UiTypeArgs(text="hi", enter=True)) == "destructive"
    assert UiType().risk_for(UiTypeArgs(text="hi")) == "change"


def test_abandoned_worker_fails_what_was_queued():
    import threading

    from jarvis.desktop.service import _Worker

    gate = threading.Event()
    w = _Worker(None)
    stuck = w.submit(gate.wait, 5)
    waiting = w.submit(lambda: "never")
    import time

    time.sleep(0.05)  # let the worker pick up the stuck call
    w.stop(fail_pending=True)
    with pytest.raises(DesktopError, match="cancelled"):
        waiting.result(timeout=1)
    gate.set()
    assert stuck.result(timeout=1) is True
