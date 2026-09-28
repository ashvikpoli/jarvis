"""Desktop notifications."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable

log = logging.getLogger(__name__)

# Uses the WinRT toast API through Windows PowerShell 5.1 (built into Windows 10/11).
# Title and message come in via environment variables, so no quoting/injection issues.
_TOAST_PS = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
$tpl = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$t = $tpl.GetElementsByTagName('text')
$t.Item(0).AppendChild($tpl.CreateTextNode($env:JARVIS_TOAST_TITLE)) > $null
$t.Item(1).AppendChild($tpl.CreateTextNode($env:JARVIS_TOAST_BODY)) > $null
$toast = [Windows.UI.Notifications.ToastNotification]::new($tpl)
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show($toast)
"""


class WindowsToastNotifier:
    def notify(self, title: str, message: str) -> None:
        env = {**os.environ, "JARVIS_TOAST_TITLE": title, "JARVIS_TOAST_BODY": message}
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(  # noqa: S603 - fixed command, data via env
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _TOAST_PS],
            env=env,
            creationflags=flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class LogNotifier:
    def notify(self, title: str, message: str) -> None:
        log.warning("%s: %s", title, message)


class CallbackNotifier:
    def __init__(self, fn: Callable[[str, str], None]):
        self.fn = fn

    def notify(self, title: str, message: str) -> None:
        self.fn(title, message)


class MultiNotifier:
    def __init__(self, *notifiers):
        self.notifiers = list(notifiers)

    def add(self, notifier) -> None:
        self.notifiers.append(notifier)

    def notify(self, title: str, message: str) -> None:
        for n in self.notifiers:
            try:
                n.notify(title, message)
            except Exception:
                log.exception("notifier %s failed", type(n).__name__)


def default_notifier() -> MultiNotifier:
    return MultiNotifier(WindowsToastNotifier() if sys.platform == "win32" else LogNotifier())
