"""Apps, web and system-info tools."""

from __future__ import annotations

import difflib
import os
import platform
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import quote_plus, urlparse

from pydantic import BaseModel, Field

from jarvis.tools.base import Risk, Tool, ToolContext, ToolError

# Friendly names → what to launch. Checked before Start Menu shortcuts.
APP_ALIASES = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "file explorer": "explorer.exe",
    "explorer": "explorer.exe",
    "files": "explorer.exe",
    "task manager": "taskmgr.exe",
    "settings": "ms-settings:",
    "windows settings": "ms-settings:",
    "snipping tool": "ms-screenclip:",
    "camera": "microsoft.windows.camera:",
    "clock": "ms-clock:",
    "store": "ms-windows-store:",
}


class SystemLauncher:
    """Real launcher: os.startfile on Windows, browser for URLs elsewhere."""

    def open(self, target: str) -> None:
        if sys.platform == "win32":
            os.startfile(target)  # type: ignore[attr-defined]  # noqa: S606
        elif target.startswith(("http://", "https://")):
            webbrowser.open(target)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, target])  # noqa: S603


def start_menu_shortcuts() -> dict[str, Path]:
    """Installed apps, from Start Menu .lnk files (user + all users)."""
    dirs = []
    if appdata := os.environ.get("APPDATA"):
        dirs.append(Path(appdata) / "Microsoft/Windows/Start Menu/Programs")
    if programdata := os.environ.get("PROGRAMDATA"):
        dirs.append(Path(programdata) / "Microsoft/Windows/Start Menu/Programs")
    found: dict[str, Path] = {}
    for d in dirs:
        if d.is_dir():
            for lnk in d.rglob("*.lnk"):
                name = lnk.stem
                if "uninstall" not in name.lower():
                    found.setdefault(name.lower(), lnk)
    return found


class AppArgs(BaseModel):
    name: str = Field(description="App name, e.g. 'Notepad', 'Spotify', 'Visual Studio Code'.")


class AppsOpen(Tool):
    name = "apps_open"
    description = "Open an installed app by name (Start Menu apps and common Windows apps)."
    risk = Risk.CHANGE
    Args = AppArgs

    def describe(self, args: AppArgs) -> str:
        return f"open the app “{args.name}”"

    async def run(self, args: AppArgs, ctx: ToolContext):
        key = args.name.strip().lower()
        if key in APP_ALIASES:
            ctx.launcher.open(APP_ALIASES[key])
            return f"Opened {args.name}."
        shortcuts = start_menu_shortcuts()
        if not shortcuts and sys.platform != "win32":
            raise ToolError("Opening apps by name only works on Windows.")
        match = shortcuts.get(key)
        if match is None:
            close = difflib.get_close_matches(key, list(shortcuts), n=1, cutoff=0.6)
            contains = [n for n in shortcuts if key in n]
            pick = close[0] if close else (min(contains, key=len) if contains else None)
            if pick is None:
                raise ToolError(f"I couldn't find an app called “{args.name}”.")
            match = shortcuts[pick]
        ctx.launcher.open(str(match))
        return f"Opened {match.stem}."


class AppsListArgs(BaseModel):
    query: str = Field("", description="Optional filter, e.g. 'adobe'.")


class AppsList(Tool):
    name = "apps_list"
    description = "List installed apps (from the Start Menu), optionally filtered."
    risk = Risk.SAFE
    Args = AppsListArgs

    async def run(self, args: AppsListArgs, ctx: ToolContext):
        names = sorted({p.stem for p in start_menu_shortcuts().values()}, key=str.lower)
        if args.query:
            names = [n for n in names if args.query.lower() in n.lower()]
        return {"apps": names[:150], "total": len(names)}


class UrlArgs(BaseModel):
    url: str = Field(description="Web address, e.g. 'youtube.com' or 'https://example.com'.")


class WebOpen(Tool):
    name = "web_open"
    description = "Open a website in the default browser."
    risk = Risk.CHANGE
    Args = UrlArgs

    def describe(self, args: UrlArgs) -> str:
        return f"open {args.url} in your browser"

    async def run(self, args: UrlArgs, ctx: ToolContext):
        url = args.url.strip()
        if "://" not in url:
            url = "https://" + url
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ToolError("I can only open http(s) web addresses.")
        ctx.launcher.open(url)
        return f"Opened {url}"


class SearchArgs(BaseModel):
    query: str = Field(description="What to search the web for.")


class WebSearch(Tool):
    name = "web_search"
    description = (
        "Open a web search in the user's browser. You will NOT see the results — "
        "tell the user the search is open."
    )
    risk = Risk.CHANGE
    Args = SearchArgs

    def describe(self, args: SearchArgs) -> str:
        return f"search the web for “{args.query}”"

    async def run(self, args: SearchArgs, ctx: ToolContext):
        url = f"https://www.google.com/search?q={quote_plus(args.query)}"
        ctx.launcher.open(url)
        return f"Opened a browser search for “{args.query}”. (I can't see the results.)"


class SystemTime(Tool):
    name = "system_time"
    description = "Get the current local date, time and timezone."
    risk = Risk.SAFE

    async def run(self, args, ctx: ToolContext):
        now = ctx.clock()
        return {
            "local_time": now.strftime("%A, %d %B %Y, %H:%M:%S"),
            "iso": now.isoformat(timespec="seconds"),
            "timezone": now.strftime("%Z %z"),
        }


def _gpu() -> dict | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        name, util, used, total, temp = [x.strip() for x in out.split(",")]
        return {
            "name": name,
            "utilization_pct": float(util),
            "vram_used_mb": float(used),
            "vram_total_mb": float(total),
            "temperature_c": float(temp),
        }
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None


class SystemInfo(Tool):
    name = "system_info"
    description = "Get current CPU, memory, disk, battery and GPU usage of this PC."
    risk = Risk.SAFE

    async def run(self, args, ctx: ToolContext):
        import psutil

        vm = psutil.virtual_memory()
        disk_root = Path.home().anchor or "/"
        du = psutil.disk_usage(disk_root)
        info = {
            "os": f"{platform.system()} {platform.release()}",
            "cpu_pct": psutil.cpu_percent(interval=0.3),
            "cpu_cores": psutil.cpu_count(),
            "ram_used_gb": round(vm.used / 1e9, 1),
            "ram_total_gb": round(vm.total / 1e9, 1),
            "ram_pct": vm.percent,
            "disk": {
                "drive": disk_root,
                "free_gb": round(du.free / 1e9, 1),
                "total_gb": round(du.total / 1e9, 1),
            },
            "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
        }
        if (batt := psutil.sensors_battery()) is not None:
            info["battery"] = {"pct": batt.percent, "plugged_in": batt.power_plugged}
        if gpu := _gpu():
            info["gpu"] = gpu
        return info


SYSTEM_TOOLS = [AppsOpen, AppsList, WebOpen, WebSearch, SystemTime, SystemInfo]
