# Phase 1 — Conversation Engine

## 1. Goal

Jarvis holds real conversations: it remembers what was said earlier in a
conversation, survives restarts (you can resume yesterday's chat), keeps the
prompt inside the model's context window, and gives us an **eval harness** to
choose models with data instead of vibes.

Still no long-term memory or tools. Those are Phases 2–3.

## 2. Architecture

```
 CLI / API
    │
    ▼
 ConversationManager ──► ConversationStore (SQLite: data/jarvis.db)
    │        │
    │        └──► PromptBuilder (system prompt + date/time + history trimmed to budget)
    ▼
 ModelManager (role → provider:model, + per-role options e.g. think=false)
    ▼
 OllamaProvider
```

`JarvisApp.chat_stream()` stays as the **stateless** path (used by `/v1/chat` and
evals). `ConversationManager.send()` is the **stateful** path (CLI, future voice).
Both share the same `PromptBuilder`, so evals test exactly what you talk to.

## 3. New / changed files

```
src/jarvis/
  storage/db.py              Database (connection, WAL, migrations)
  storage/conversations.py   ConversationStore, Conversation, StoredMessage
  conversation/prompt.py     PromptBuilder, estimate_tokens, BuiltPrompt
  conversation/manager.py    ConversationManager
  evals/cases.py             EvalCase, load_cases, checks
  evals/runner.py            EvalRunner, results, report
evals/cases.toml             ~35 test cases
```

## 4. Database

```sql
conversations(id INTEGER PK, title TEXT, created_at TEXT, updated_at TEXT)
messages(id INTEGER PK, conversation_id → conversations ON DELETE CASCADE,
         role TEXT, content TEXT, thinking TEXT, created_at TEXT,
         model TEXT, prompt_tokens INT, completion_tokens INT,
         ttft_ms REAL, answer_ttft_ms REAL, interrupted INT)
schema_version(version INT)
```

Migrations are an ordered list of SQL scripts; `schema_version` records how far
we've applied. Phase 3 appends memory tables the same way.

## 5. Key behaviours

- **Context window:** budget = `context_tokens − reserve_tokens − system prompt`.
  History is added newest→oldest until the budget is full; the latest user
  message is always included. Dropped turns are counted (Phase 3 will summarise
  them into memory instead of forgetting). `context_tokens` is also sent to
  Ollama as `num_ctx`, so the model's window matches our budget.
- **Interruptions are saved:** Ctrl+C keeps the partial answer, flagged
  `interrupted`, so the history stays truthful.
- **Per-role options:** `[models.options.chat] think = false` etc. Thinking off by
  default for chat (speed); toggle with `/think on`.
- **Latency:** `answer_ttft_ms` = time to first *answer* token (excludes
  thinking). That's the number voice cares about.

## 6. Interfaces

```python
class ConversationStore:
    create(title=None) -> Conversation
    get(id) / latest() / list(limit) / rename(id, title) / delete(id)
    add_message(conversation_id, ...) -> StoredMessage
    messages(conversation_id) -> list[StoredMessage]

class PromptBuilder:
    build(history: list[Message]) -> BuiltPrompt(messages, dropped, est_tokens)

class ConversationManager:
    new(title=None) -> Conversation
    send(conversation_id, text, *, role="chat", options=None) -> AsyncIterator[StreamChunk]
```

## 7. HTTP API (new)

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/conversations` | Recent conversations |
| POST | `/v1/conversations` | Create `{title?}` |
| GET | `/v1/conversations/{id}` | Conversation + messages |
| DELETE | `/v1/conversations/{id}` | Delete |
| POST | `/v1/conversations/{id}/messages` | `{content, stream?}` → SSE/JSON; reply is saved |

## 8. CLI

`jarvis chat [--resume [ID]]` · in chat: `/new`, `/list`, `/resume ID`,
`/think on|off`, `/model REF`, `/help`, `/exit` · `jarvis history [ID]` ·
`jarvis eval [--models a,b] [--category X] [--think] [-v]`

## 9. Evals

`evals/cases.toml` — each case has a prompt, optional prior turns, and
deterministic checks: `contains_all`, `contains_any`, `not_contains`, `regex`,
`max_words`, `min_words`, `number`, `json_keys`, `no_markdown`.

Categories: persona, concise, instruction, reasoning, factual, multi-turn,
structured, honesty, voice.

`jarvis eval --models ollama:qwen3:14b,ollama:qwen3:8b` prints pass rate per
category plus median answer latency and tok/s per model, and saves full results
to `data/evals/<timestamp>.json`. Rerun after every prompt/model change.

## 10. Done when

Automated: store CRUD + cascade delete; migrations idempotent; prompt trimming
keeps newest turns and the last user message; interrupted replies saved;
conversation API; every eval check; runner against a scripted fake model.

On your PC:
- [ ] Chat, quit, `jarvis chat --resume` → Jarvis remembers the conversation
- [ ] Ctrl+C mid-answer → partial answer visible in `jarvis history <id>`
- [ ] `/think on` shows reasoning, `/think off` answers faster
- [ ] `jarvis eval` runs on 2+ models; we pick the default chat model from it

## 11. Future hooks

- Dropped-turn counter → Phase 3 summarisation into long-term memory.
- `messages` table gains `tool_calls` in Phase 2 (new migration).
- Eval cases grow with every phase (tool-use cases in Phase 2, memory recall in 3).
