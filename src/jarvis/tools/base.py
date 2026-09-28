"""Tool contract.

A tool = name + description + pydantic Args (→ JSON schema) + risk + async run().
That's the same shape as an MCP tool, so external MCP servers can be adapted into
this registry later without changing the agent loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from jarvis.storage import Database


class Risk(StrEnum):
    SAFE = "safe"  # read-only
    CHANGE = "change"  # creates/changes something reversible
    DESTRUCTIVE = "destructive"  # overwrites, moves, deletes


class ToolError(Exception):
    """An expected failure. The message is shown to the model (and the user)."""


class Notifier(Protocol):
    def notify(self, title: str, message: str) -> None: ...


class Launcher(Protocol):
    """Opens apps, files and URLs. Swapped for a fake in tests and evals."""

    def open(self, target: str) -> None: ...


@dataclass
class ToolContext:
    """Everything a tool may touch. Built per conversation by JarvisApp."""

    db: Database
    file_roots: list[Path]
    launcher: Launcher
    notifier: Notifier
    conversation_id: int | None = None
    max_read_chars: int = 20_000
    apps: Any = None  # AppCatalog; None = the shared real one
    clock: Any = field(default=lambda: datetime.now().astimezone())


class NoArgs(BaseModel):
    pass


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    risk: ClassVar[Risk] = Risk.SAFE
    Args: ClassVar[type[BaseModel]] = NoArgs

    def risk_for(self, args: BaseModel) -> Risk:
        """Risk can depend on arguments (e.g. overwrite=True). Default: static."""
        return self.risk

    def describe(self, args: BaseModel) -> str:
        """One line for approval prompts and logs."""
        shown = ", ".join(f"{k}={v!r}" for k, v in args.model_dump(exclude_none=True).items())
        return f"{self.name}({shown})"

    def description_for(self, ctx: ToolContext) -> str:
        """Description sent to the model; override to include context (e.g. folders)."""
        return self.description

    def spec(self, ctx: ToolContext) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description_for(ctx),
                "parameters": _clean_schema(self.Args.model_json_schema()),
            },
        }

    @abstractmethod
    async def run(self, args: BaseModel, ctx: ToolContext) -> Any:
        """Do the thing. Return str or JSON-able data. Raise ToolError for known failures."""


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip pydantic noise (titles) so small models see a compact schema."""
    if isinstance(schema, dict):
        return {k: _clean_schema(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema
