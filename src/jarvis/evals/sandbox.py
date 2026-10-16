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


class EvalVision:
    """Stands in for the vision model: the eval tests the chat model's choices."""

    async def look(self, image_b64: str, question: str) -> str:
        return (
            "A Discord window, channel #general. In front of it an error box titled "
            "'Discord Update' says: 'Update failed — not enough disk space (error 0x70)'."
        )

    async def locate(self, image_b64: str, description: str):
        return 0.5, 0.5, description


def make_eval_desktop():
    """A pretend desktop (Discord, Spotify, Chrome) that records instead of acting."""
    from jarvis.desktop import Desktop, Element, Rect, Window
    from jarvis.desktop.guard import Guard
    from jarvis.desktop.model import Shot

    term = Window(1, "Jarvis", "WindowsTerminal.exe", Rect(0, 0, 900, 600), True)
    discord = Window(2, "#general - Discord", "Discord.exe", Rect(0, 0, 1600, 1000))
    spotify = Window(3, "Spotify Premium", "Spotify.exe", Rect(0, 0, 1600, 1000))
    chrome = Window(4, "New Tab - Google Chrome", "chrome.exe", Rect(0, 0, 1600, 1000))
    controls = {
        2: [
            ("Edit", "Message #general", Rect(300, 900, 1400, 950)),
            ("Button", "Send", Rect(1410, 900, 1480, 950)),
            ("ListItem", "general", Rect(10, 100, 200, 130)),
            ("ListItem", "random", Rect(10, 130, 200, 160)),
            ("Button", "User Settings", Rect(10, 950, 50, 990)),
            ("Button", "Mute", Rect(60, 950, 100, 990)),
        ],
        3: [
            ("Button", "Play", Rect(780, 940, 820, 980)),
            ("Button", "Next", Rect(830, 940, 870, 980)),
            ("Edit", "What do you want to play?", Rect(500, 20, 1100, 60)),
        ],
        4: [
            ("Edit", "Address and search bar", Rect(100, 50, 1500, 90)),
            ("Button", "New Tab", Rect(300, 5, 330, 35)),
        ],
    }

    class Backend:
        def __init__(self):
            self.wins = [term, discord, spotify, chrome]
            self.fg = 1
            self.actions: list = []
            self.pos = (0, 0)

        def windows(self):
            return [Window(**{**w.__dict__, "active": w.handle == self.fg}) for w in self.wins]

        def active_window(self):
            return next(w for w in self.windows() if w.handle == self.fg)

        def screenshot(self, window, whole_screen, max_side, hide):
            return Shot("", window.rect, (1280, 800))

        def window_at(self, x, y):
            return next(w for w in self.windows() if w.handle == self.fg)

        def element_at(self, x, y):
            return None

        def visible_windows(self):
            return [w for w in self.windows() if not w.minimized]

        def on_screen(self, x, y):
            return True

        def refresh(self, el):
            return el

        def responding(self, window):
            return True

        def elements(self, window, limit):
            return [Element(k, n, r) for k, n, r in controls.get(window.handle, [])]

        def focused_element(self):
            return None

        def invoke(self, el):
            return True

        def focus_element(self, el):
            pass

        def click(self, x, y, button, count):
            self.pos = (x, y)
            self.actions.append(("click", x, y))

        def type_text(self, text):
            self.actions.append(("type", text))

        def keys(self, combo):
            self.actions.append(("keys", combo))

        def scroll(self, x, y, dx, dy):
            self.pos = (x, y)

        def cursor(self):
            return self.pos

        def window_action(self, window, action):
            self.actions.append(("window", window.process, action))
            if action == "focus":
                self.fg = window.handle

    return Desktop(Backend(), EvalVision(), Guard(), run_inline=True, settle_s=0, own_windows={1})


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
        desktop=make_eval_desktop(),
    )
    return Sandbox(root=root, ctx=ctx)
