"""Installed-app discovery and matching.

Sources (Windows):
  1. `Get-StartApps` — everything in the Start menu, including Microsoft Store apps
     (WhatsApp, Spotify from the Store…). Launched via shell:AppsFolder\\<AppID>.
  2. Start Menu .lnk files — fallback if PowerShell isn't available.
Plus a few built-in aliases (notepad, calculator, settings…).
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstalledApp:
    name: str
    target: str  # what to pass to the launcher
    source: str  # builtin | store | shortcut


# Friendly names → built-in Windows apps. Checked first.
BUILTIN_APPS = {
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
    "microsoft store": "ms-windows-store:",
}

BROWSERS = {
    "chrome",
    "edge",
    "firefox",
    "brave",
    "opera",
    "vivaldi",
    "safari",
    "arc",
    "internet explorer",
    "tor browser",
}
JUNK = ("uninstall", "readme", "release notes", "documentation", "help", "website", "license")
# Websites whose app has a different name than the domain.
SITE_APP_NAMES = {
    "teams.microsoft.com": "teams",
    "teams.live.com": "teams",
    "outlook.live.com": "outlook",
    "outlook.office.com": "outlook",
    "music.youtube.com": "youtube music",
    "web.telegram.org": "telegram",
}
# Domains that belong to a company, not one app ("google.com" ≠ "Google Chrome").
PLATFORM_DOMAINS = {
    "google",
    "microsoft",
    "amazon",
    "apple",
    "adobe",
    "live",
    "office",
    "windows",
    "bing",
    "yahoo",
    "github",
}
_TWO_PART_TLDS = {"co.uk", "co.in", "com.au", "co.jp", "com.br", "co.nz", "org.uk"}


def normalize(name: str) -> str:
    name = name.lower().replace("&", " and ")
    name = re.sub(r"[^a-z0-9+ ]+", " ", name)
    words = [w for w in name.split() if w not in ("the", "app", "application", "inc")]
    return " ".join(words)


def _is_browser(app: InstalledApp) -> bool:
    n = normalize(app.name)
    return any(b in n for b in BROWSERS)


def site_key(url: str) -> str | None:
    """'https://www.discord.com/app' → 'discord'. None for platform domains."""
    if "://" not in url:
        url = "https://" + url
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return None
    if host in SITE_APP_NAMES:
        return SITE_APP_NAMES[host]
    parts = host.split(".")
    tld_len = 2 if ".".join(parts[-2:]) in _TWO_PART_TLDS else 1
    if len(parts) <= tld_len:
        return None
    sld = parts[-tld_len - 1]
    return None if sld in PLATFORM_DOMAINS else sld


class AppCatalog:
    """Discovers installed apps (cached) and matches names to them."""

    def __init__(self, ttl_s: float = 600):
        self.ttl_s = ttl_s
        self._apps: list[InstalledApp] | None = None
        self._loaded_at = 0.0
        self._lock = threading.Lock()

    # -------------------------------------------------------------- discovery

    def _scan(self) -> list[InstalledApp]:
        if sys.platform != "win32":
            return []
        apps = self._scan_start_apps() or self._scan_shortcuts()
        return apps

    @staticmethod
    def _scan_start_apps() -> list[InstalledApp]:
        try:
            out = subprocess.run(  # noqa: S603 - fixed command
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Get-StartApps | ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
            data = json.loads(out or "[]")
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            log.info("Get-StartApps failed; falling back to Start Menu shortcuts")
            return []
        if isinstance(data, dict):
            data = [data]
        return [
            InstalledApp(d["Name"], f"shell:AppsFolder\\{d['AppID']}", "store")
            for d in data
            if d.get("Name") and d.get("AppID")
        ]

    @staticmethod
    def _scan_shortcuts() -> list[InstalledApp]:
        dirs = []
        if appdata := os.environ.get("APPDATA"):
            dirs.append(Path(appdata) / "Microsoft/Windows/Start Menu/Programs")
        if programdata := os.environ.get("PROGRAMDATA"):
            dirs.append(Path(programdata) / "Microsoft/Windows/Start Menu/Programs")
        found = {}
        for d in dirs:
            if d.is_dir():
                for lnk in d.rglob("*.lnk"):
                    found.setdefault(lnk.stem.lower(), InstalledApp(lnk.stem, str(lnk), "shortcut"))
        return list(found.values())

    def apps(self) -> list[InstalledApp]:
        with self._lock:
            if self._apps is None or time.monotonic() - self._loaded_at > self.ttl_s:
                scanned = self._scan()
                self._apps = [a for a in scanned if not any(j in a.name.lower() for j in JUNK)]
                self._loaded_at = time.monotonic()
            return list(self._apps)

    def prefetch(self) -> None:
        """Scan in the background so the first 'open X' is instant."""
        threading.Thread(target=self.apps, name="app-scan", daemon=True).start()

    # -------------------------------------------------------------- matching

    def find(self, query: str) -> InstalledApp | None:
        return self.lookup(query)[0]

    def lookup(self, query: str) -> tuple[InstalledApp | None, bool]:
        """Return (app, exact). exact=False means a typo-tolerant guess."""
        q = normalize(query)
        if not q:
            return None, False
        if q in BUILTIN_APPS:
            return InstalledApp(query.strip(), BUILTIN_APPS[q], "builtin"), True
        by_norm: dict[str, InstalledApp] = {}
        for a in self.apps():
            by_norm.setdefault(normalize(a.name), a)
        if q in by_norm:
            return by_norm[q], True
        # whole word inside a longer name: "teams" → "Microsoft Teams"
        word_hits = [a for n, a in by_norm.items() if re.search(rf"\b{re.escape(q)}\b", n)]
        if word_hits:
            return min(word_hits, key=lambda a: len(a.name)), True
        # typos: similar spelling AND similar length. Length matters — without it
        # "photoshop" matched "photos" (6 shared letters) and opened the wrong app.
        slack = max(2, len(q) // 5)
        similar = [n for n in by_norm if abs(len(n) - len(q)) <= slack]
        close = difflib.get_close_matches(q, similar, n=1, cutoff=0.8)
        if close:
            return by_norm[close[0]], False
        # the query is part of a longer name: "visual studio" → "Visual Studio Code"
        partial = [a for n, a in by_norm.items() if q in n]
        if partial:
            return min(partial, key=lambda a: len(a.name)), True
        return None, False

    def find_for_site(self, url: str) -> InstalledApp | None:
        """The installed app that corresponds to a website, if any (never a browser)."""
        key = site_key(url)
        if not key:
            return None
        for a in self.apps():
            if _is_browser(a):
                continue
            words = normalize(a.name).split()
            if not words:
                continue
            if " ".join(words) == key or words[0] == key or words[-1] == key:
                return a
            if " " in key and key in " ".join(words):
                return a
        return None


class StaticAppCatalog(AppCatalog):
    """Fixed app list — for tests and evals."""

    def __init__(self, apps: list[InstalledApp]):
        super().__init__()
        self._fixed = list(apps)

    def _scan(self) -> list[InstalledApp]:
        return list(self._fixed)
