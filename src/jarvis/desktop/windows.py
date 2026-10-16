"""The real Windows desktop: windows (user32), screenshots (PrintWindow / mss),
controls (UI Automation), mouse & keyboard (pynput).

Every method runs on the Desktop's single worker thread (see service.py): UI
Automation objects belong to the COM thread that created them.
"""

from __future__ import annotations

import base64
import ctypes
import io
import logging
import time
from ctypes import wintypes as wt

from jarvis.desktop.model import Element, Rect, Shot, Window

log = logging.getLogger(__name__)

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32")
dwmapi = ctypes.WinDLL("dwmapi")

GW_OWNER = 4
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_CLOAKED = 14
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SW_MAXIMIZE, SW_MINIMIZE, SW_RESTORE = 3, 6, 9
WM_CLOSE = 0x0010
PW_RENDERFULLCONTENT = 2
MONITOR_DEFAULTTONEAREST = 2
SWP_NOZORDER, SWP_SHOWWINDOW = 0x0004, 0x0040
DIB_RGB_COLORS, BI_RGB = 0, 0

INTERESTING = {
    "ButtonControl", "SplitButtonControl", "EditControl", "CheckBoxControl",
    "RadioButtonControl", "ComboBoxControl", "ListItemControl", "MenuItemControl",
    "TabItemControl", "HyperlinkControl", "TreeItemControl", "DataItemControl",
    "SliderControl", "SpinnerControl", "DocumentControl",
}  # fmt: skip

EnumWindowsProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wt.DWORD),
        ("rcMonitor", wt.RECT),
        ("rcWork", wt.RECT),
        ("dwFlags", wt.DWORD),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wt.DWORD),
        ("biWidth", wt.LONG),
        ("biHeight", wt.LONG),
        ("biPlanes", wt.WORD),
        ("biBitCount", wt.WORD),
        ("biCompression", wt.DWORD),
        ("biSizeImage", wt.DWORD),
        ("biXPelsPerMeter", wt.LONG),
        ("biYPelsPerMeter", wt.LONG),
        ("biClrUsed", wt.DWORD),
        ("biClrImportant", wt.DWORD),
    ]


def _sig(fn, restype, *argtypes):
    fn.restype = restype
    fn.argtypes = argtypes


# 64-bit handles must not be squeezed through c_int.
_sig(user32.GetForegroundWindow, wt.HWND)
_sig(user32.EnumWindows, wt.BOOL, EnumWindowsProc, wt.LPARAM)
_sig(user32.IsWindowVisible, wt.BOOL, wt.HWND)
_sig(user32.IsIconic, wt.BOOL, wt.HWND)
_sig(user32.IsZoomed, wt.BOOL, wt.HWND)
_sig(user32.IsWindow, wt.BOOL, wt.HWND)
_sig(user32.GetWindow, wt.HWND, wt.HWND, wt.UINT)
_sig(user32.GetWindowLongW, ctypes.c_long, wt.HWND, ctypes.c_int)
_sig(user32.GetWindowTextLengthW, ctypes.c_int, wt.HWND)
_sig(user32.GetWindowTextW, ctypes.c_int, wt.HWND, wt.LPWSTR, ctypes.c_int)
_sig(user32.GetWindowRect, wt.BOOL, wt.HWND, ctypes.POINTER(wt.RECT))
_sig(user32.GetWindowThreadProcessId, wt.DWORD, wt.HWND, ctypes.POINTER(wt.DWORD))
_sig(user32.ShowWindow, wt.BOOL, wt.HWND, ctypes.c_int)
_sig(user32.SetForegroundWindow, wt.BOOL, wt.HWND)
_sig(user32.BringWindowToTop, wt.BOOL, wt.HWND)
_sig(user32.AttachThreadInput, wt.BOOL, wt.DWORD, wt.DWORD, wt.BOOL)
_sig(user32.PostMessageW, wt.BOOL, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)
_sig(user32.SetWindowPos, wt.BOOL, wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int,
     ctypes.c_int, ctypes.c_int, wt.UINT)  # fmt: skip
_sig(user32.MonitorFromWindow, wt.HMONITOR, wt.HWND, wt.DWORD)
_sig(user32.GetMonitorInfoW, wt.BOOL, wt.HMONITOR, ctypes.POINTER(MONITORINFO))
_sig(user32.PrintWindow, wt.BOOL, wt.HWND, wt.HDC, wt.UINT)
_sig(user32.GetDC, wt.HDC, wt.HWND)
_sig(user32.ReleaseDC, ctypes.c_int, wt.HWND, wt.HDC)
_sig(gdi32.CreateCompatibleDC, wt.HDC, wt.HDC)
_sig(gdi32.CreateCompatibleBitmap, wt.HBITMAP, wt.HDC, ctypes.c_int, ctypes.c_int)
_sig(gdi32.SelectObject, wt.HGDIOBJ, wt.HDC, wt.HGDIOBJ)
_sig(gdi32.DeleteObject, wt.BOOL, wt.HGDIOBJ)
_sig(gdi32.DeleteDC, wt.BOOL, wt.HDC)
_sig(gdi32.GetDIBits, ctypes.c_int, wt.HDC, wt.HBITMAP, wt.UINT, wt.UINT, ctypes.c_void_p,
     ctypes.POINTER(BITMAPINFOHEADER), wt.UINT)  # fmt: skip
_sig(dwmapi.DwmGetWindowAttribute, ctypes.c_long, wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD)
_sig(kernel32.OpenProcess, wt.HANDLE, wt.DWORD, wt.BOOL, wt.DWORD)
_sig(kernel32.QueryFullProcessImageNameW, wt.BOOL, wt.HANDLE, wt.DWORD, wt.LPWSTR,
     ctypes.POINTER(wt.DWORD))  # fmt: skip
_sig(kernel32.CloseHandle, wt.BOOL, wt.HANDLE)
_sig(kernel32.GetCurrentThreadId, wt.DWORD)
_sig(kernel32.GetConsoleWindow, wt.HWND)
_sig(user32.GetAncestor, wt.HWND, wt.HWND, wt.UINT)
_sig(user32.WindowFromPoint, wt.HWND, wt.POINT)
_sig(user32.GetClassNameW, ctypes.c_int, wt.HWND, wt.LPWSTR, ctypes.c_int)
_sig(user32.FindWindowExW, wt.HWND, wt.HWND, wt.HWND, wt.LPCWSTR, wt.LPCWSTR)
_sig(user32.IsWindowEnabled, wt.BOOL, wt.HWND)
_sig(user32.IsHungAppWindow, wt.BOOL, wt.HWND)
_sig(user32.GetSystemMetrics, ctypes.c_int, ctypes.c_int)
_sig(user32.VkKeyScanW, ctypes.c_short, wt.WCHAR)
GA_ROOT, GA_ROOTOWNER = 2, 3
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 76, 77, 78, 79
INPUT_KEYBOARD, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE = 1, 0x0002, 0x0004


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wt.WORD),
        ("wScan", wt.WORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _MOUSEINPUT(ctypes.Structure):  # only here so INPUT has the right size
    _fields_ = [
        ("dx", wt.LONG),
        ("dy", wt.LONG),
        ("mouseData", wt.DWORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", _MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _INPUTUNION)]


_sig(user32.SendInput, wt.UINT, wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int)


def _h(hwnd) -> int:
    return int(hwnd or 0)


def _rect(r: wt.RECT) -> Rect:
    return Rect(r.left, r.top, r.right, r.bottom)


def own_windows() -> set[int]:
    """Jarvis's console window — and, in Windows Terminal, the terminal window that owns
    its hidden pseudo-console."""
    console = kernel32.GetConsoleWindow()
    if not console:
        return set()
    return {h for h in (_h(console), _h(user32.GetAncestor(console, GA_ROOTOWNER))) if h}


def init_thread():
    """Run once in each Desktop worker thread: COM for UI Automation, quiet logging.

    Returns the initializer; the worker keeps it for its own lifetime (a replacement
    worker gets its own — dropping one uninitializes COM on the thread that drops it).
    """
    import uiautomation as auto

    auto.Logger.SetLogFile("")  # it would otherwise write @AutomationLog.txt in the cwd
    keep = auto.UIAutomationInitializerInThread()
    # The UIA client object belongs to the thread that made it: make a fresh one here.
    try:
        auto._AutomationClient._instance = None
    except AttributeError:
        pass
    return keep


class WindowsBackend:
    def __init__(self):
        try:  # physical pixels everywhere, so screenshots, UIA and the mouse agree
            ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
        except (OSError, AttributeError):
            pass
        from pynput import keyboard, mouse

        self._kb = keyboard.Controller()
        self._mouse = mouse.Controller()
        self._keyboard_mod = keyboard
        self._mouse_mod = mouse

    # ------------------------------------------------------------ windows

    def _title(self, hwnd) -> str:
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value

    def _process(self, hwnd) -> str:
        pid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not h:
            return ""
        try:
            size = wt.DWORD(1024)
            buf = ctypes.create_unicode_buffer(size.value)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return buf.value.rsplit("\\", 1)[-1]
            return ""
        finally:
            kernel32.CloseHandle(h)

    def _bounds(self, hwnd) -> Rect:
        r = wt.RECT()
        if (
            dwmapi.DwmGetWindowAttribute(
                hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(r), ctypes.sizeof(r)
            )
            != 0
        ):
            user32.GetWindowRect(hwnd, ctypes.byref(r))
        return _rect(r)

    def _class(self, hwnd) -> str:
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    def _app_process(self, hwnd) -> str:
        """The real app: Store apps live inside ApplicationFrameHost.exe."""
        proc = self._process(hwnd)
        if proc.lower() == "applicationframehost.exe":
            core = user32.FindWindowExW(hwnd, None, "Windows.UI.Core.CoreWindow", None)
            if core:
                proc = self._process(core) or proc
        return proc

    def _window(self, hwnd, fg: int) -> Window:
        return Window(
            handle=_h(hwnd),
            title=self._title(hwnd),
            process=self._app_process(hwnd),
            rect=self._bounds(hwnd),
            active=_h(hwnd) == fg,
            minimized=bool(user32.IsIconic(hwnd)),
            cls=self._class(hwnd),
            owner=_h(user32.GetAncestor(hwnd, GA_ROOTOWNER))
            if user32.GetWindow(hwnd, GW_OWNER)
            else 0,
        )

    def _listed(self, hwnd) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return False
        if user32.GetWindow(hwnd, GW_OWNER) and not user32.IsWindowEnabled(hwnd):
            return False  # owned windows (dialogs like "Save changes?") count when usable
        if user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
            return False
        cloaked = wt.DWORD()
        dwmapi.DwmGetWindowAttribute(
            hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
        )
        if cloaked.value:  # on another virtual desktop / suspended UWP app
            return False
        title = self._title(hwnd)
        return bool(title) and title != "Program Manager"

    def windows(self) -> list[Window]:
        fg = _h(user32.GetForegroundWindow())
        found: list[int] = []

        def cb(hwnd, _lparam):
            if self._listed(hwnd):
                found.append(_h(hwnd))
            return True

        user32.EnumWindows(EnumWindowsProc(cb), 0)  # top of the z-order first
        return [self._window(h, fg) for h in found]

    def visible_windows(self) -> list[Window]:
        """Everything visible on screen (also popups, tool windows, untitled ones)."""
        fg = _h(user32.GetForegroundWindow())
        found: list[int] = []

        def cb(hwnd, _lparam):
            if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
                cloaked = wt.DWORD()
                dwmapi.DwmGetWindowAttribute(
                    hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
                )
                if not cloaked.value:
                    found.append(_h(hwnd))
            return True

        user32.EnumWindows(EnumWindowsProc(cb), 0)
        return [w for w in (self._window(h, fg) for h in found) if not w.rect.empty]

    def active_window(self) -> Window | None:
        fg = user32.GetForegroundWindow()
        return self._window(fg, _h(fg)) if fg else None

    def window_action(self, window: Window, action: str) -> None:
        hwnd = wt.HWND(window.handle)
        if not user32.IsWindow(hwnd):
            raise RuntimeError("that window has closed")
        if action == "minimize":
            user32.ShowWindow(hwnd, SW_MINIMIZE)
        elif action == "maximize":
            user32.ShowWindow(hwnd, SW_MAXIMIZE)
        elif action == "restore":
            user32.ShowWindow(hwnd, SW_RESTORE)
        elif action == "close":
            user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)  # the app may still ask to save
        elif action == "focus":
            self._focus(hwnd)
        elif action in ("left_half", "right_half"):
            self._snap(hwnd, action == "left_half")

    def _focus(self, hwnd) -> None:
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)
        fg = user32.GetForegroundWindow()
        fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        me = kernel32.GetCurrentThreadId()
        attached = bool(fg_tid) and fg_tid != me and user32.AttachThreadInput(me, fg_tid, True)
        try:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(me, fg_tid, False)

    def _work_area(self, hwnd) -> Rect:
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW(
            user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), ctypes.byref(info)
        )
        return _rect(info.rcWork)

    def _monitor(self, hwnd) -> Rect:
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW(
            user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), ctypes.byref(info)
        )
        return _rect(info.rcMonitor)

    def _snap(self, hwnd, left: bool) -> None:
        work = self._work_area(hwnd)  # the monitor it's on now, before restoring moves it
        if user32.IsZoomed(hwnd) or user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)
        # Windows draws an invisible resize border; compensate so the visible edges line up.
        outer = wt.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(outer))
        vis = self._bounds(hwnd)
        bl, bt = vis.left - outer.left, vis.top - outer.top
        br, bb = outer.right - vis.right, outer.bottom - vis.bottom
        half = work.width // 2
        x = work.left if left else work.left + half
        user32.SetWindowPos(
            hwnd, None, x - bl, work.top - bt, half + bl + br, work.height + bt + bb,
            SWP_NOZORDER | SWP_SHOWWINDOW,
        )  # fmt: skip
        self._focus(hwnd)

    # ------------------------------------------------------------ screenshots

    def _print_window(self, hwnd, outer: Rect):
        """The window's own pixels, even if other windows cover it."""
        from PIL import Image

        w, h = outer.width, outer.height
        if w <= 0 or h <= 0:
            return None
        screen_dc = user32.GetDC(None)
        mem_dc = gdi32.CreateCompatibleDC(screen_dc)
        bmp = gdi32.CreateCompatibleBitmap(screen_dc, w, h)
        old = gdi32.SelectObject(mem_dc, bmp)
        try:
            if not user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT):
                return None
            gdi32.SelectObject(mem_dc, old)  # GetDIBits needs the bitmap deselected
            old = None
            header = BITMAPINFOHEADER()
            header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            header.biWidth, header.biHeight = w, -h  # negative = top-down rows
            header.biPlanes, header.biBitCount, header.biCompression = 1, 32, BI_RGB
            buf = ctypes.create_string_buffer(w * h * 4)
            if not gdi32.GetDIBits(mem_dc, bmp, 0, h, buf, ctypes.byref(header), DIB_RGB_COLORS):
                return None
            img = Image.frombuffer("RGB", (w, h), buf.raw, "raw", "BGRX", 0, 1)
            return None if img.getbbox() is None else img  # all black: app refused
        finally:
            if old is not None:
                gdi32.SelectObject(mem_dc, old)
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(None, screen_dc)

    def _grab(self, region: Rect):
        import mss
        from PIL import Image

        with mss.mss() as sct:
            raw = sct.grab(
                {
                    "left": region.left,
                    "top": region.top,
                    "width": region.width,
                    "height": region.height,
                }
            )
            return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    def screenshot(
        self, window: Window, whole_screen: bool, max_side: int, hide: list[Window]
    ) -> Shot:
        from PIL import ImageDraw

        hwnd = wt.HWND(window.handle)
        if not user32.IsHungAppWindow(hwnd) and not whole_screen and user32.IsIconic(hwnd):
            raise RuntimeError(f"“{window.title}” is minimized — restore it first")
        own_pixels = False
        if whole_screen:
            region = self._monitor(hwnd)
            img = self._grab(region)
        else:
            region = self._bounds(hwnd)
            outer = wt.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(outer))
            outer_r = _rect(outer)
            img = None if user32.IsHungAppWindow(hwnd) else self._print_window(hwnd, outer_r)
            if img is not None:  # crop the invisible border away
                own_pixels = True
                img = img.crop(
                    (
                        region.left - outer_r.left,
                        region.top - outer_r.top,
                        region.left - outer_r.left + region.width,
                        region.top - outer_r.top + region.height,
                    )
                )
            else:
                img = self._grab(region)  # what's visible on screen there
        if not own_pixels and hide:
            # Screen pixels may include protected windows: black them out first.
            draw = ImageDraw.Draw(img)
            for w in hide:
                r = w.rect
                box = (
                    max(r.left, region.left) - region.left,
                    max(r.top, region.top) - region.top,
                    min(r.right, region.right) - region.left,
                    min(r.bottom, region.bottom) - region.top,
                )
                if box[2] > box[0] and box[3] > box[1]:
                    draw.rectangle(box, fill=(0, 0, 0))
        img.thumbnail((max_side, max_side))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=82)
        return Shot(base64.b64encode(out.getvalue()).decode("ascii"), region, img.size)

    def window_at(self, x: int, y: int) -> Window | None:
        hwnd = user32.WindowFromPoint(wt.POINT(x, y))
        if not hwnd:
            return None
        root = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
        return self._window(root, _h(user32.GetForegroundWindow()))

    def on_screen(self, x: int, y: int) -> bool:
        left = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        top = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        w = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        h = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
        return left <= x < left + w and top <= y < top + h

    def responding(self, window: Window) -> bool:
        return not user32.IsHungAppWindow(wt.HWND(window.handle))

    # ------------------------------------------------------------ UI Automation

    def _element(self, c) -> Element | None:
        try:
            kind = c.ControlTypeName
            r = c.BoundingRectangle
            rect = Rect(r.left, r.top, r.right, r.bottom)
            pw = bool(c.IsPassword)
            value = None
            if kind == "EditControl" and not pw:
                try:
                    value = c.GetValuePattern().Value or None
                except Exception:
                    value = None
            checked = None
            if kind == "CheckBoxControl":
                try:
                    checked = c.GetTogglePattern().ToggleState == 1
                except Exception:
                    checked = None
            return Element(
                kind=kind.removesuffix("Control"),
                name=(c.Name or "").strip(),
                rect=rect,
                value=value,
                is_password=pw,
                enabled=bool(c.IsEnabled),
                checked=checked,
                handle=c,
            )
        except Exception:
            return None

    def elements(self, window: Window, limit: int) -> list[Element]:
        import uiautomation as auto

        root = auto.ControlFromHandle(window.handle)
        if root is None:
            return []
        out: list[Element] = []
        deadline = time.monotonic() + 3.0
        visited = 0
        for c, _depth in auto.WalkControl(root, maxDepth=25):
            visited += 1
            if len(out) >= limit or time.monotonic() > deadline or visited > 4000:
                break
            try:
                if c.ControlTypeName not in INTERESTING or c.IsOffscreen:
                    continue
            except Exception:
                continue
            el = self._element(c)
            if el is None or el.rect.empty:
                continue
            if el.kind == "Document" and not el.name:
                continue
            if not el.name and el.kind not in ("Edit", "ComboBox", "Document"):
                continue  # an unnamed button is no use to the model
            out.append(el)
        return out

    def refresh(self, element: Element) -> Element | None:
        """Read the control again (it may have moved, changed or disappeared)."""
        fresh = self._element(element.handle) if element.handle is not None else None
        if fresh is None or fresh.name != element.name or fresh.kind != element.kind:
            return None
        try:
            if element.handle.IsOffscreen:  # scrolled away (maybe under a sticky header)
                fresh.rect = Rect(0, 0, 0, 0)
        except Exception:
            pass
        fresh.id = element.id
        return fresh

    def element_at(self, x: int, y: int) -> Element | None:
        import uiautomation as auto

        try:
            c = auto.ControlFromPoint(x, y)
        except Exception:
            return None
        return self._element(c) if c is not None else None

    def focused_element(self) -> Element | None:
        import uiautomation as auto

        c = auto.GetFocusedControl()
        return self._element(c) if c is not None else None

    def focus_element(self, element: Element) -> None:
        element.handle.SetFocus()

    def invoke(self, element: Element) -> bool:
        """Use the control's own action (works even off-screen). False = not supported."""
        c = element.handle
        for getter, action in (
            ("GetInvokePattern", "Invoke"),
            ("GetTogglePattern", "Toggle"),
            ("GetSelectionItemPattern", "Select"),
        ):
            fn = getattr(c, getter, None)
            if fn is None:
                continue
            try:
                pattern = fn()
                if pattern is not None:
                    getattr(pattern, action)(waitTime=0.1)
                    return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------ mouse & keyboard

    def cursor(self) -> tuple[int, int]:
        x, y = self._mouse.position
        return int(x), int(y)

    def click(self, x: int, y: int, button: str, count: int) -> None:
        b = getattr(self._mouse_mod.Button, button, self._mouse_mod.Button.left)
        self._mouse.position = (x, y)
        time.sleep(0.05)
        self._mouse.click(b, count)

    def scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self._mouse.position = (x, y)
        time.sleep(0.05)
        self._mouse.scroll(dx, dy)

    def type_text(self, text: str) -> None:
        """Type characters as Unicode — unaffected by Caps Lock or a held Shift."""
        events = []
        for ch in text:
            code = ord(ch)
            units = (
                [code]
                if code <= 0xFFFF
                else [  # outside the BMP: a surrogate pair
                    0xD800 + ((code - 0x10000) >> 10),
                    0xDC00 + ((code - 0x10000) & 0x3FF),
                ]
            )
            for u in units:
                for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
                    ev = INPUT(type=INPUT_KEYBOARD)
                    ev.u.ki = KEYBDINPUT(0, u, flags, 0, 0)
                    events.append(ev)
        if events:
            arr = (INPUT * len(events))(*events)
            user32.SendInput(len(events), arr, ctypes.sizeof(INPUT))

    def _key(self, name: str):
        K = self._keyboard_mod.Key
        special = {"win": K.cmd, "escape": K.esc, "enter": K.enter}
        if name in special:
            return special[name]
        if hasattr(K, name):
            return getattr(K, name)
        # A letter/digit by its virtual key, so Ctrl+Shift+S is S — not a typed "S".
        vk = user32.VkKeyScanW(name) & 0xFF
        if vk != 0xFF:
            return self._keyboard_mod.KeyCode.from_vk(vk)
        return self._keyboard_mod.KeyCode.from_char(name)

    def keys(self, combo: list[str]) -> None:
        keys = [self._key(k) for k in combo]
        pressed = []
        try:
            for k in keys:
                self._kb.press(k)
                pressed.append(k)
                time.sleep(0.02)
        finally:
            for k in reversed(pressed):
                self._kb.release(k)
