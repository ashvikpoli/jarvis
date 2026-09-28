"""Apps, web and system-info tools."""

from __future__ import annotations

import asyncio
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

from jarvis.tools.apps import AppCatalog
from jarvis.tools.base import Risk, Tool, ToolContext, ToolError


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


_DEFAULT_CATALOG = AppCatalog()


def catalog(ctx: ToolContext) -> AppCatalog:
    return ctx.apps or _DEFAULT_CATALOG


class AppArgs(BaseModel):
    name: str = Field(description="App name, e.g. 'Discord', 'Spotify', 'Visual Studio Code'.")


class AppsOpen(Tool):
    name = "apps_open"
    description = (
        "Open an installed app by name. Use this whenever the user says 'open <something>' "
        "for an app or service (Discord, Spotify, WhatsApp, Word, Steam...) — prefer it "
        "over web_open unless they explicitly ask for the website."
    )
    risk = Risk.CHANGE
    Args = AppArgs

    def describe(self, args: AppArgs) -> str:
        return f"open the app “{args.name}”"

    async def run(self, args: AppArgs, ctx: ToolContext):
        cat = catalog(ctx)
        app = await asyncio.to_thread(cat.find, args.name)
        if app is None:
            if sys.platform != "win32" and ctx.apps is None:
                raise ToolError("Opening apps by name only works on Windows.")
            raise ToolError(
                f"“{args.name}” doesn't seem to be installed. If the user wants, you can "
                "open its website with web_open instead."
            )
        ctx.launcher.open(app.target)
        return f"Opened the {app.name} app."


class AppsListArgs(BaseModel):
    query: str = Field("", description="Optional filter, e.g. 'adobe'.")


class AppsList(Tool):
    name = "apps_list"
    description = "List installed apps, optionally filtered by name."
    risk = Risk.SAFE
    Args = AppsListArgs

    async def run(self, args: AppsListArgs, ctx: ToolContext):
        apps = await asyncio.to_thread(catalog(ctx).apps)
        names = sorted({a.name for a in apps}, key=str.lower)
        if args.query:
            names = [n for n in names if args.query.lower() in n.lower()]
        return {"apps": names[:150], "total": len(names)}


class UrlArgs(BaseModel):
    url: str = Field(description="Web address, e.g. 'youtube.com' or 'https://example.com'.")
    browser: bool = Field(
        False,
        description="Set true ONLY if the user explicitly asked for the website or browser. "
        "Otherwise, if a matching app is installed (e.g. Discord), the app opens instead.",
    )


class WebOpen(Tool):
    name = "web_open"
    description = (
        "Open a website. If the site has an installed app (Discord, Spotify, Slack...) the "
        "app opens instead, unless browser=true."
    )
    risk = Risk.CHANGE
    Args = UrlArgs

    def describe(self, args: UrlArgs) -> str:
        return f"open {args.url}" + (" in your browser" if args.browser else "")

    async def run(self, args: UrlArgs, ctx: ToolContext):
        url = args.url.strip()
        if "://" not in url:
            url = "https://" + url
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ToolError("I can only open http(s) web addresses.")
        if not args.browser:
            app = await asyncio.to_thread(catalog(ctx).find_for_site, url)
            if app is not None:
                ctx.launcher.open(app.target)
                return (
                    f"Opened the installed {app.name} app instead of the website. "
                    "(If the user wanted the website, call web_open again with browser=true.)"
                )
        ctx.launcher.open(url)
        return f"Opened {url} in the browser."


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
