"""A disposable world for tool evals: temp folders, in-memory DB, fake launcher.

Nothing an eval does can touch your real files, open real apps or send real
notifications.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jarvis.storage import Database
from jarvis.tools import ApprovalRequest, ToolContext
from jarvis.tools.apps import InstalledApp, StaticAppCatalog

FIXTURE_FILES = {
    "Documents/report-q3.txt": "Q3 report\nRevenue up 12% quarter over quarter.\n",
    "Documents/budget-2026.csv": "item,cost\nrent,1500\ngroceries,400\n",
    "Documents/Taxes/receipt-2025.pdf": b"%PDF-1.4 fake\x00\x01",
    "Documents/Projects/jarvis-ideas.md": "# Jarvis ideas\n- voice\n- memory\n",
    "Desktop/todo.txt": "buy milk\ncall mom\nbook dentist\n",
    "Downloads/setup-installer.exe": b"MZ\x00\x00",
    "Downloads/holiday-photo.jpg": b"\xff\xd8\xff\x00",
}


SANDBOX_APPS = [
    InstalledApp("Discord", "shell:AppsFolder\\com.squirrel.Discord.Discord", "store"),
    InstalledApp("Spotify", "shell:AppsFolder\\SpotifyAB.SpotifyMusic", "store"),
    InstalledApp("WhatsApp", "shell:AppsFolder\\5319275A.WhatsAppDesktop", "store"),
    InstalledApp("Google Chrome", "shell:AppsFolder\\Chrome", "store"),
    InstalledApp("Microsoft Teams", "shell:AppsFolder\\MSTeams", "store"),
    InstalledApp("Visual Studio Code", "shell:AppsFolder\\Microsoft.VisualStudioCode", "store"),
    InstalledApp("Photos", "shell:AppsFolder\\Microsoft.Windows.Photos", "store"),
]


SANDBOX_MEMORIES = [
    {"text": "User's name is Adi.", "category": "profile"},
    {"text": "User runs an e-commerce company called Exinent.", "category": "profile"},
    {"text": "User lives in Hyderabad.", "category": "profile"},
    {"text": "User's sister is named Priya.", "category": "person"},
    {"text": "User prefers short, direct answers.", "category": "preference"},
    {"text": "User is building a voice assistant called Jarvis.", "category": "project"},
]


class RecordingLauncher:
    def __init__(self):
        self.opened: list[str] = []

    def open(self, target: str) -> None:
        self.opened.append(target)


class RecordingNotifier:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    def notify(self, title: str, message: str) -> None:
        self.sent.append((title, message))


@dataclass
class Sandbox:
    root: Path
    ctx: ToolContext
    approvals: list[str] = field(default_factory=list)

    async def approver(self, req: ApprovalRequest) -> str:
        self.approvals.append(req.description)
        return "once"

    def close(self) -> None:
        self.ctx.db.close()
        shutil.rmtree(self.root, ignore_errors=True)


def make_sandbox() -> Sandbox:
    # Fixed path (recreated per case): tool descriptions mention the folders, so a
    # random name would change the prompt every case and defeat Ollama's prompt cache,
    # making eval latency look worse than real use. Cases run one at a time.
    root = Path(tempfile.gettempdir()) / "jarvis-eval-sandbox"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    for rel, content in FIXTURE_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")

    db = Database(":memory:")
    now = datetime.now(UTC)
    with db.lock, db.conn:
        db.conn.execute(
            "INSERT INTO notes (text, created_at) VALUES (?, ?)",
            ("Guest wifi password: lemonade42", now.isoformat(timespec="seconds")),
        )
        db.conn.execute(
            "INSERT INTO reminders (text, due_at, created_at) VALUES (?, ?, ?)",
            (
                "dentist appointment",
                (now + timedelta(days=1)).isoformat(timespec="seconds"),
                now.isoformat(timespec="seconds"),
            ),
        )
    ctx = ToolContext(
        db=db,
        file_roots=[(root / d).resolve() for d in ("Documents", "Desktop", "Downloads")],
        launcher=RecordingLauncher(),
        notifier=RecordingNotifier(),
        apps=StaticAppCatalog(SANDBOX_APPS),
    )
    return Sandbox(root=root, ctx=ctx)
