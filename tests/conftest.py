from pathlib import Path

import pytest

from jarvis.app import JarvisApp
from jarvis.core.config import load_settings
from jarvis.providers.scripted import ScriptedProvider


class FakeLauncher:
    def __init__(self):
        self.opened: list[str] = []

    def open(self, target: str) -> None:
        self.opened.append(target)


class FakeNotifier:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    def notify(self, title: str, message: str) -> None:
        self.sent.append((title, message))


@pytest.fixture
def roots(tmp_path) -> list[Path]:
    docs = tmp_path / "Documents"
    desk = tmp_path / "Desktop"
    docs.mkdir()
    desk.mkdir()
    (docs / "report.txt").write_text("Quarterly report\nRevenue up 12%.", encoding="utf-8")
    (docs / "budget-2026.csv").write_text("item,cost\nrent,1000\n", encoding="utf-8")
    (docs / "projects").mkdir()
    (docs / "projects" / "jarvis-notes.md").write_text("# Jarvis\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    return [docs.resolve(), desk.resolve()]


@pytest.fixture
def make_app(tmp_path, roots):
    """Build a JarvisApp around a ScriptedProvider with fakes for the outside world."""
    apps = []

    def factory(turns=(), config: str = "") -> tuple[JarvisApp, ScriptedProvider]:
        cfg = tmp_path / f"cfg{len(apps)}"
        cfg.mkdir()
        (cfg / "default.toml").write_text(
            '[models]\nroles = { chat = "scripted:main" }\n' + config, encoding="utf-8"
        )
        provider = ScriptedProvider(turns)
        app = JarvisApp.create(
            load_settings(cfg),
            providers={"scripted": provider},
            db_path=":memory:",
            launcher=FakeLauncher(),
            notifier=FakeNotifier(),
            file_roots=roots,
        )
        apps.append(app)
        return app, provider

    return factory
