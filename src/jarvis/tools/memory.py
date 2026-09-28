"""Tools for Jarvis's long-term memory about the user."""

from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.tools.base import Risk, Tool, ToolContext, ToolError

_CATS = "profile (who they are), preference, person (people in their life), project, other"


def _service(ctx: ToolContext):
    mem = getattr(ctx, "memory", None)
    if mem is None or not mem.enabled:
        raise ToolError("Long-term memory is turned off.")
    return mem


class SaveArgs(BaseModel):
    text: str = Field(description='One standalone sentence, e.g. "User\'s sister is Priya."')
    category: str = Field("other", description=f"One of: {_CATS}.")


class MemorySave(Tool):
    name = "memory_save"
    description = (
        "Remember a lasting fact about the user (their life, people, preferences, work) for "
        "future conversations. Use when they say 'remember that…'. For codes, passwords or "
        "reference info, use notes_add instead."
    )
    risk = Risk.CHANGE
    Args = SaveArgs

    def describe(self, args: SaveArgs) -> str:
        return f"remember: “{args.text[:80]}”"

    async def run(self, args: SaveArgs, ctx: ToolContext):
        mem, created = await _service(ctx).save(
            args.text, category=args.category, source="told", conversation_id=ctx.conversation_id
        )
        verb = "Remembered" if created else "Already knew this; updated"
        return f"{verb} memory #{mem.id}: {mem.text}"


class SearchArgs(BaseModel):
    query: str = Field("", description="What to look for. Empty = list everything remembered.")


class MemorySearch(Tool):
    name = "memory_search"
    description = (
        "Search what you remember about the user. With an empty query, lists all memories — "
        "use that when they ask what you know or remember about them."
    )
    risk = Risk.SAFE
    Args = SearchArgs

    async def run(self, args: SearchArgs, ctx: ToolContext):
        svc = _service(ctx)
        if not args.query.strip():
            items = svc.store.list(limit=200)
            if not items:
                return "You don't have any memories about the user yet."
            return {
                "memories": [{"id": m.id, "text": m.text, "category": m.category} for m in items]
            }
        hits = await svc.recall(args.query, limit=10, min_score=0.35)
        if not hits:
            return "Nothing remembered about that."
        return {"memories": [{"id": h.memory.id, "text": h.memory.text} for h in hits]}


class UpdateArgs(BaseModel):
    id: int = Field(description="Memory id (the #number).")
    text: str = Field(description="The corrected full sentence.")


class MemoryUpdate(Tool):
    name = "memory_update"
    description = "Correct a memory when the user says something changed (use its #id)."
    risk = Risk.CHANGE
    Args = UpdateArgs

    def describe(self, args: UpdateArgs) -> str:
        return f"update memory #{args.id} → “{args.text[:70]}”"

    async def run(self, args: UpdateArgs, ctx: ToolContext):
        mem = await _service(ctx).update(args.id, args.text)
        if mem is None:
            raise ToolError(f"There's no memory #{args.id}.")
        return f"Updated memory #{mem.id}: {mem.text}"


class ForgetArgs(BaseModel):
    id: int = Field(description="Memory id (the #number) to delete.")


class MemoryForget(Tool):
    name = "memory_forget"
    description = (
        "Delete a memory when the user asks you to forget something (use its #id; search "
        "first if you don't know it)."
    )
    risk = Risk.CHANGE
    Args = ForgetArgs

    def describe(self, args: ForgetArgs) -> str:
        return f"forget memory #{args.id}"

    async def run(self, args: ForgetArgs, ctx: ToolContext):
        svc = _service(ctx)
        mem = svc.store.get(args.id)
        if mem is None or not svc.forget(args.id):
            raise ToolError(f"There's no memory #{args.id}.")
        return f"Forgot memory #{args.id}: {mem.text}"


MEMORY_TOOLS = [MemorySave, MemorySearch, MemoryUpdate, MemoryForget]
