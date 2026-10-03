"""Search the user's Obsidian vault (their notes, journal, people & project pages)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.tools.base import Risk, Tool, ToolContext, ToolError


class VaultSearchArgs(BaseModel):
    query: str = Field(description="What to look for, in plain words.")
    limit: int = Field(5, ge=1, le=15)


class VaultSearch(Tool):
    name = "vault_search"
    description = (
        "Search the user's Obsidian notes (their second brain: notes they wrote, journal, "
        "notes on people and projects). Use when they ask what they wrote, noted or planned "
        "about something, or when memory and notes_search don't have the answer."
    )
    risk = Risk.SAFE
    Args = VaultSearchArgs

    async def run(self, args: VaultSearchArgs, ctx: ToolContext):
        index = getattr(ctx, "vault", None)
        if index is None:
            raise ToolError("The Obsidian vault isn't connected ([vault] path in config).")
        hits = await index.search(args.query, limit=args.limit)
        return {"results": hits} if hits else "Nothing about that in your notes."


VAULT_TOOLS = [VaultSearch]
