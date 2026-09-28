"""File tools, sandboxed to allowed root folders."""

from __future__ import annotations

import fnmatch
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from jarvis.tools.base import Risk, Tool, ToolContext, ToolError

# Windows "known folders" — correct even when OneDrive redirects Documents/Desktop.
_KNOWN_FOLDER_GUIDS = {
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
    "videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
}

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "AppData", "$RECYCLE.BIN"}
MAX_SEARCH_VISITS = 25_000


def _windows_known_folder(guid: str) -> Path | None:
    import ctypes
    import uuid
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", wintypes.BYTE * 8),
        ]

    u = uuid.UUID(guid)
    g = GUID(
        u.time_low, u.time_mid, u.time_hi_version, (wintypes.BYTE * 8).from_buffer_copy(u.bytes[8:])
    )
    out = ctypes.c_wchar_p()
    shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
    if shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(out)) != 0:
        return None
    try:
        return Path(out.value)
    finally:
        ctypes.windll.ole32.CoTaskMemFree(out)  # type: ignore[attr-defined]


def resolve_root(spec: str) -> Path:
    """'Documents' / 'Desktop' / 'Downloads' / 'Jarvis' → the real folder; else a path."""
    key = spec.strip().lower()
    if key == "jarvis":
        from jarvis.core.config import REPO_ROOT

        return REPO_ROOT.resolve()
    if key in _KNOWN_FOLDER_GUIDS:
        if sys.platform == "win32":
            found = _windows_known_folder(_KNOWN_FOLDER_GUIDS[key])
            if found:
                return found.resolve()
        return (Path.home() / spec.strip().capitalize()).resolve()
    return Path(os.path.expandvars(spec)).expanduser().resolve()


def _under(p: Path, bases: list[Path]) -> bool:
    return any(p == b or p.is_relative_to(b) for b in bases)


def in_workspace(p: Path, ctx: ToolContext) -> bool:
    return _under(p, ctx.file_roots)


def safe_path(raw: str, ctx: ToolContext, *, must_exist: bool = False, write: bool = False) -> Path:
    """Resolve a model-supplied path and check it against the file zones.

    Accepts absolute paths, '~/...', a root's name ('Documents/notes.txt'), or a path
    relative to the first root. Symlinks are resolved, so they can't sneak around
    the checks. Raises ToolError with a user-facing reason when access isn't allowed.
    """
    pol = ctx.files
    raw = (raw or "").strip().strip('"')
    if not raw and not ctx.file_roots:
        raise ToolError("No folders are enabled for file access.")
    p = Path(os.path.expandvars(raw)).expanduser() if raw else ctx.file_roots[0]
    if not p.is_absolute():
        if not ctx.file_roots:
            raise ToolError("Give me a full path; no default folders are enabled.")
        first, *rest = p.parts or (".",)
        by_name = {r.name.lower(): r for r in ctx.file_roots}
        p = (
            by_name[first.lower()].joinpath(*rest)
            if first.lower() in by_name
            else (ctx.file_roots[0] / p)
        )
    p = p.resolve()

    if _under(p, pol.no_access):
        raise ToolError(
            f"{p} is a private location (passwords, keys or browser data). "
            "I never read or change it."
        )
    inside = in_workspace(p, ctx)
    if write:
        if _under(p, pol.no_write):
            raise ToolError(
                f"{p} is a protected folder (system files or Jarvis's own "
                "settings). I never change it, even with permission."
            )
        if not inside and pol.outside_writes != "ask":
            allowed = ", ".join(str(r) for r in ctx.file_roots)
            raise ToolError(f"{p} is outside the folders I can change ({allowed}).")
    elif not inside and not pol.read_anywhere:
        allowed = ", ".join(str(r) for r in ctx.file_roots)
        raise ToolError(f"{p} is outside the folders I'm allowed to use ({allowed}).")
    if must_exist and not p.exists():
        raise ToolError(f"{p} doesn't exist.")
    return p


def _entry(p: Path) -> dict:
    st = p.stat()
    return {
        "name": p.name,
        "type": "folder" if p.is_dir() else "file",
        "size": None if p.is_dir() else st.st_size,
        "modified": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
    }


class _FileTool(Tool):
    def description_for(self, ctx: ToolContext) -> str:
        roots = "; ".join(str(r) for r in ctx.file_roots) or "none"
        extra = (
            " You can also read files elsewhere on this PC (not private or system data); "
            "changing anything outside these folders needs the user's explicit OK."
            if ctx.files.read_anywhere
            else ""
        )
        return f"{self.description} The user's folders: {roots}.{extra}"


class _WritingFileTool(_FileTool):
    """A tool that changes files. Outside the workspace it's always destructive-level."""

    def _write_paths(self, args) -> list[str]:
        return [args.path]

    def _outside(self, args, ctx: ToolContext | None) -> bool:
        if ctx is None:
            return False
        for raw in self._write_paths(args):
            try:
                if not in_workspace(safe_path(raw, ctx, write=True), ctx):
                    return True
            except ToolError:
                return False  # run() will refuse it with the reason
        return False

    def risk_in(self, args, ctx):
        return Risk.DESTRUCTIVE if self._outside(args, ctx) else self.risk_for(args)

    def describe_in(self, args, ctx):
        text = self.describe(args)
        return text + "  ⚠ OUTSIDE your usual folders" if self._outside(args, ctx) else text


# ------------------------------------------------------------------ read-only


class ListArgs(BaseModel):
    path: str = Field("", description="Folder to list. Empty = the first allowed folder.")
    pattern: str = Field("*", description="Optional glob filter, e.g. '*.pdf'.")
    limit: int = Field(100, ge=1, le=500)


class FilesList(_FileTool):
    name = "files_list"
    description = "List the files and subfolders in a folder, newest first."
    risk = Risk.SAFE
    Args = ListArgs

    async def run(self, args: ListArgs, ctx: ToolContext):
        folder = safe_path(args.path, ctx, must_exist=True)
        if not folder.is_dir():
            raise ToolError(f"{folder} is a file, not a folder.")
        items = [
            p
            for p in folder.iterdir()
            if fnmatch.fnmatch(p.name.lower(), args.pattern.lower()) and not p.name.startswith(".")
        ]
        items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return {
            "folder": str(folder),
            "total": len(items),
            "entries": [_entry(p) for p in items[: args.limit]],
        }


class SearchArgs(BaseModel):
    query: str = Field(description="Part of the file name, or a glob like '*.xlsx'.")
    path: str = Field("", description="Folder to search in. Empty = all allowed folders.")
    limit: int = Field(30, ge=1, le=200)


class FilesSearch(_FileTool):
    name = "files_search"
    description = "Find files by name (case-insensitive, searches subfolders)."
    risk = Risk.SAFE
    Args = SearchArgs

    async def run(self, args: SearchArgs, ctx: ToolContext):
        roots = [safe_path(args.path, ctx, must_exist=True)] if args.path else ctx.file_roots
        q = args.query.lower()
        is_glob = any(ch in q for ch in "*?[")
        hits: list[dict] = []
        visited = 0
        for root in roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in SKIP_DIRS
                    and not d.startswith(".")
                    and not _under(Path(dirpath, d), ctx.files.no_access)
                ]
                for name in dirnames + filenames:
                    visited += 1
                    low = name.lower()
                    if fnmatch.fnmatch(low, q) if is_glob else q in low:
                        p = Path(dirpath) / name
                        hits.append({**_entry(p), "path": str(p)})
                        if len(hits) >= args.limit:
                            return {"matches": hits, "truncated": True}
                if visited > MAX_SEARCH_VISITS:
                    return {"matches": hits, "truncated": True, "note": "search stopped early"}
        return {"matches": hits, "truncated": False}


class ReadArgs(BaseModel):
    path: str = Field(description="File to read.")


class FilesRead(_FileTool):
    name = "files_read"
    description = "Read a text file (txt, md, csv, json, code, logs...)."
    risk = Risk.SAFE
    Args = ReadArgs

    async def run(self, args: ReadArgs, ctx: ToolContext):
        p = safe_path(args.path, ctx, must_exist=True)
        if p.is_dir():
            raise ToolError(f"{p} is a folder; use files_list.")
        raw = p.read_bytes()[: ctx.max_read_chars * 4]
        if b"\x00" in raw[:4096]:
            raise ToolError(f"{p.name} is a binary file; I can only read text files for now.")
        text = raw.decode("utf-8", errors="replace")
        truncated = len(text) > ctx.max_read_chars
        return {"path": str(p), "content": text[: ctx.max_read_chars], "truncated": truncated}


# ------------------------------------------------------------------ changes


class WriteArgs(BaseModel):
    path: str = Field(description="File to create.")
    content: str = Field(description="Full text content of the file.")
    overwrite: bool = Field(False, description="Replace the file if it already exists.")


class FilesWrite(_WritingFileTool):
    name = "files_write"
    description = "Create a text file. Won't replace an existing file unless overwrite=true."
    risk = Risk.CHANGE
    Args = WriteArgs

    def risk_for(self, args: WriteArgs) -> Risk:
        return Risk.DESTRUCTIVE if args.overwrite else Risk.CHANGE

    def describe(self, args: WriteArgs) -> str:
        verb = "OVERWRITE" if args.overwrite else "create"
        preview = args.content[:80].replace("\n", " ⏎ ")
        return f"{verb} file {args.path} ({len(args.content)} chars: “{preview}”)"

    async def run(self, args: WriteArgs, ctx: ToolContext):
        p = safe_path(args.path, ctx, write=True)
        if p.is_dir():
            raise ToolError(f"{p} is a folder.")
        if p.exists() and not args.overwrite:
            raise ToolError(f"{p} already exists. Ask the user before overwriting it.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args.content, encoding="utf-8")
        return f"Wrote {len(args.content)} characters to {p}"


class MkdirArgs(BaseModel):
    path: str = Field(description="Folder to create (parents are created too).")


class FilesMkdir(_WritingFileTool):
    name = "files_mkdir"
    description = "Create a folder."
    risk = Risk.CHANGE
    Args = MkdirArgs

    def describe(self, args: MkdirArgs) -> str:
        return f"create folder {args.path}"

    async def run(self, args: MkdirArgs, ctx: ToolContext):
        p = safe_path(args.path, ctx, write=True)
        if p.exists():
            return f"{p} already exists."
        p.mkdir(parents=True)
        return f"Created folder {p}"


class MoveArgs(BaseModel):
    source: str = Field(description="File or folder to move/rename.")
    destination: str = Field(description="New path. If it's an existing folder, moves inside it.")


class FilesMove(_WritingFileTool):
    name = "files_move"
    description = "Move or rename a file or folder. Never replaces an existing file."
    risk = Risk.CHANGE
    Args = MoveArgs

    def _write_paths(self, args: MoveArgs) -> list[str]:
        return [args.source, args.destination]

    def describe(self, args: MoveArgs) -> str:
        return f"move {args.source} → {args.destination}"

    async def run(self, args: MoveArgs, ctx: ToolContext):
        src = safe_path(args.source, ctx, must_exist=True, write=True)
        if src in ctx.file_roots:
            raise ToolError("I won't move one of the top-level allowed folders.")
        dst = safe_path(args.destination, ctx, write=True)
        if dst.is_dir():
            dst = dst / src.name
            safe_path(str(dst), ctx, write=True)
        if dst.exists():
            raise ToolError(f"{dst} already exists.")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Moved {src} → {dst}"


class DeleteArgs(BaseModel):
    path: str = Field(description="File or folder to delete (goes to the Recycle Bin).")


class FilesDelete(_WritingFileTool):
    name = "files_delete"
    description = "Delete a file or folder by sending it to the Recycle Bin (recoverable)."
    risk = Risk.DESTRUCTIVE
    Args = DeleteArgs

    def describe(self, args: DeleteArgs) -> str:
        return f"move {args.path} to the Recycle Bin"

    async def run(self, args: DeleteArgs, ctx: ToolContext):
        from send2trash import send2trash

        p = safe_path(args.path, ctx, must_exist=True, write=True)
        if p in ctx.file_roots or p == Path(p.anchor):
            raise ToolError("I won't delete one of the top-level folders or a whole drive.")
        send2trash(str(p))
        return f"Moved {p} to the Recycle Bin."


EXECUTABLE_SUFFIXES = {
    ".exe",
    ".bat",
    ".cmd",
    ".com",
    ".ps1",
    ".psm1",
    ".vbs",
    ".vbe",
    ".js",
    ".jse",
    ".wsf",
    ".wsh",
    ".msi",
    ".msp",
    ".scr",
    ".pif",
    ".lnk",
    ".reg",
    ".hta",
    ".cpl",
    ".jar",
    ".appx",
    ".msix",
    ".application",
    ".url",
    ".sh",
    ".py",
    ".pyw",
}


class OpenArgs(BaseModel):
    path: str = Field(description="File or folder to open, e.g. 'Documents/report.pdf'.")


class FilesOpen(_FileTool):
    name = "files_open"
    description = (
        "Open a file in its default app (like double-clicking it) — documents, PDFs, "
        "spreadsheets, images, videos — or open a folder in File Explorer. "
        "Use this when the user says 'open <file>'. Doesn't run programs or scripts."
    )
    risk = Risk.CHANGE
    Args = OpenArgs

    def describe(self, args: OpenArgs) -> str:
        return f"open {args.path}"

    async def run(self, args: OpenArgs, ctx: ToolContext):
        p = safe_path(args.path, ctx, must_exist=True)
        if p.is_file() and p.suffix.lower() in EXECUTABLE_SUFFIXES:
            raise ToolError(
                f"{p.name} is a program or script; I don't run those by opening files. "
                "(To start an installed app, use apps_open.)"
            )
        ctx.launcher.open(str(p))
        return f"Opened {p}" + (" in File Explorer." if p.is_dir() else ".")


FILE_TOOLS = [
    FilesList,
    FilesSearch,
    FilesRead,
    FilesOpen,
    FilesWrite,
    FilesMkdir,
    FilesMove,
    FilesDelete,
]
