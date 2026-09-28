# Phase 2 — Tools & Permissions

## 1. Goal

Jarvis stops only *talking about* things and starts *doing* them — safely.
The model can call tools (search files, open apps, set reminders…) in a loop
until the task is done, and **every action passes through a permission check
and lands in an audit log**.

First tool batch (chosen by Adi): **files & folders, apps & system, reminders & notes.**
Terminal commands come later.

## 2. Architecture

```
 user text
    │
    ▼
 ConversationManager ──────────────► ConversationStore (messages incl. tool calls)
    │
    ▼
 AgentLoop  (max 6 steps)
    │  ┌───────────────────────────────────────────────┐
    ├─►│ ModelManager.stream_chat(messages, tools=specs)│── text → user
    │  └───────────────────────────────────────────────┘
    │        │ tool_calls?
    │        ▼
    │  ToolRegistry.get(name) → validate args (pydantic)
    │        ▼
    │  PermissionPolicy.decide(tool, args) ── allow / ask / deny
    │        │ ask → Approver (CLI prompt; API denies for now)
    │        ▼
    │  tool.run(args, ToolContext) ──► ToolAuditLog
    │        ▼
    └── tool result appended as a `tool` message → next model step
```

## 3. Risk levels & permissions

| Risk | Meaning | Default |
|---|---|---|
| `safe` | read-only (list files, system info, list reminders) | allow |
| `change` | creates/changes something reversible (new file, open app, add note) | ask |
| `destructive` | overwrites, moves, deletes | ask, every time |

Risk can depend on arguments (`files_write` with `overwrite=true` is destructive).
Per-tool overrides in `[tools.permissions.overrides]`; defaults auto-allow Jarvis's
own data (notes, reminders) and opening apps/URLs. At an approval prompt,
"always" allows that tool for the rest of the session (never for destructive).

## 4. Tools

| Tool | Risk | Notes |
|---|---|---|
| `files_list`, `files_search`, `files_read` | safe | only inside allowed roots |
| `files_write`, `files_mkdir`, `files_move` | change (overwrite = destructive) | never overwrites unless asked |
| `files_delete` | destructive | sends to Recycle Bin, never permanent |
| `apps_open`, `apps_list` | change / safe | built-in aliases + Start Menu shortcuts |
| `web_open`, `web_search` | change | http(s) only, opens default browser |
| `system_time`, `system_info` | safe | CPU/RAM/disk/battery/GPU (nvidia-smi) |
| `notes_add/search/list/delete` | change/safe/safe/change | stored in SQLite |
| `reminders_set/list/update/cancel` | change/safe/change/change | Windows toast when due |

**Allowed roots** default to your Windows *known folders* Documents, Desktop,
Downloads (resolved via the Windows API, so OneDrive-redirected folders work).

**Reminders fire only while Jarvis is running** (chat or serve). Phase 6 adds a
background service.

## 5. Data

Migration 2 adds `messages.tool_calls / tool_name / tool_call_id`, and tables
`tool_calls` (audit), `notes`, `reminders`.

## 6. Interfaces

```python
class Tool(ABC):
    name; description; risk; Args: type[BaseModel]
    def risk_for(args) -> Risk
    def describe(args) -> str                       # shown in approval prompts
    async def run(args, ctx: ToolContext) -> Any    # raise ToolError for user-facing errors

class PermissionPolicy:  decide(tool, args) -> Decision; allow_for_session(name)
Approver = async (ApprovalRequest) -> "once" | "session" | "deny"
class AgentLoop:  run(messages, ctx, approver, on_message=...) -> AsyncIterator[StreamChunk]
```

MCP-compatible by design: each tool is name + description + JSON-schema input,
the same shape as MCP tools, so an MCP client can register external servers'
tools into the same registry later.

## 7. CLI / API

- Chat shows `⚙ tool(args)` and results inline; approval prompt `[y]es / [n]o / [a]lways`.
- `jarvis tools` — list tools and their permission · `jarvis audit` — recent tool calls ·
  `jarvis notify "text"` — test Windows notifications.
- API: SSE adds `tool_call` / `tool_result` events. Tools needing approval are
  denied over the API for now (no UI to approve yet).

## 8. Evals

New `tools` category runs in a **sandbox** (temp folders with fixture files,
in-memory DB, fake app launcher, auto-approve) so evals never touch your real
files or open apps. Checks: `tools_called`, `tools_called_any`, `no_tools`,
`tool_args`.

## 9. Done when

Automated: registry/specs, permissions (incl. arg-dependent risk, session allow),
path sandboxing (no escaping roots), every tool, agent loop (multi-step, denial,
unknown tool, bad args, max steps, cancellation), persistence of tool messages,
API events, tool eval checks.

On your PC:
- [ ] "What's in my Downloads folder?" lists real files
- [ ] "Create a file shopping.txt on my Desktop with milk and eggs" asks first, then creates it
- [ ] "Open Notepad" / "open youtube.com" work
- [ ] "Remind me in 1 minute to stand up" → Windows notification a minute later
- [ ] "Actually make it 2 minutes" updates the same reminder (the Tuesday/Thursday fix)
- [ ] `jarvis audit` shows everything that happened
- [ ] `jarvis eval --category tools` passes on qwen3:8b
