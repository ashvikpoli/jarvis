"""Tools Jarvis can use, plus the registry / permission / audit machinery."""

from jarvis.tools.base import Launcher, Notifier, Risk, Tool, ToolContext, ToolError
from jarvis.tools.calc import CALC_TOOLS
from jarvis.tools.desktop import DESKTOP_TOOLS
from jarvis.tools.files import FILE_TOOLS, resolve_root, safe_path
from jarvis.tools.memory import MEMORY_TOOLS
from jarvis.tools.notes import NOTE_TOOLS, ReminderScheduler, reminder_rows
from jarvis.tools.registry import (
    ApprovalAnswer,
    ApprovalRequest,
    Approver,
    Decision,
    PermissionPolicy,
    ToolAuditLog,
    ToolRegistry,
)
from jarvis.tools.system import SYSTEM_TOOLS, SystemLauncher
from jarvis.tools.vault import VAULT_TOOLS

ALL_TOOLS = [
    *FILE_TOOLS,
    *SYSTEM_TOOLS,
    *NOTE_TOOLS,
    *MEMORY_TOOLS,
    *CALC_TOOLS,
    *VAULT_TOOLS,
    *DESKTOP_TOOLS,
]


def default_registry() -> ToolRegistry:
    return ToolRegistry(cls() for cls in ALL_TOOLS)


__all__ = [
    "ALL_TOOLS",
    "ApprovalAnswer",
    "ApprovalRequest",
    "Approver",
    "Decision",
    "Launcher",
    "Notifier",
    "PermissionPolicy",
    "ReminderScheduler",
    "Risk",
    "SystemLauncher",
    "Tool",
    "ToolAuditLog",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "default_registry",
    "reminder_rows",
    "resolve_root",
    "safe_path",
]
