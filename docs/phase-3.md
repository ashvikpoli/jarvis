# Phase 3 — Long-term Memory

## 1. Goal

Jarvis remembers **you** across conversations: your name, work, people in your
life, preferences, projects — picked up naturally as you talk, or when you say
"remember that…". It uses them without being asked, and you can see, correct and
delete everything it remembers.

All memory stays on your PC (`data/jarvis.db`).

## 2. How it works

```
 you: "My sister Priya is visiting next month"
   │
   ├─► (1) RECALL  embed the message → nearest memories (+ keyword match)
   │        → added to what the model sees:  <memory> #12 User's sister is Priya … </memory>
   │
   ├─► reply streams as usual (tools, approvals… unchanged)
   │
   └─► (2) LEARN (background, after the reply)
            extractor LLM sees your message + related existing memories
            → operations: add "User has a sister named Priya" / update #7 / delete #3
            → "💾 Remembered: User has a sister named Priya"
```

**Two layers in the prompt**

| Layer | What | Where in the prompt | Why there |
|---|---|---|---|
| Core | identity facts (name, where you live, work) — always included | system prompt | changes rarely → Ollama's prompt cache survives |
| Recalled | the few memories relevant to *this* message | prepended to your latest message | changes every turn, so kept at the end |

## 3. Memory types

`profile` (who you are) · `preference` (likes, dislikes, how you want things) ·
`person` (people and who they are to you) · `project` (ongoing work, goals) ·
`other`.

**Never stored as memories:** passwords/codes (those are *notes*, saved only when
you ask), transient states ("I'm tired today"), one-off tasks, the assistant's own
words, questions.

## 4. Components

```
src/jarvis/memory/
  vectors.py     float32 packing + cosine (no numpy)
  store.py       MemoryStore: CRUD, vector + keyword search, usage tracking
  service.py     MemoryService: embed, save-with-dedupe, recall → prompt context
  extractor.py   LLM extraction → add/update/delete ops; background worker thread
src/jarvis/tools/memory.py   memory_save / memory_search / memory_update / memory_forget
```

- **Embeddings:** `ollama:qwen3-embedding:0.6b` (role `embed`, 639 MB). If it's not
  installed, recall falls back to keyword matching and doctor tells you.
- **Dedupe:** saving a memory ≥ 0.9 similar to an existing one updates it instead.
- **Extraction** runs on the chat model (same `num_ctx`, so no reload), JSON-schema
  output, temperature 0, in a background thread so replies never wait. Only runs
  when your message could contain a personal fact (mentions I/my/me/we…).

## 5. Data (migration 3)

```sql
memories(id, text, category, source ['told'|'learned'], conversation_id,
         created_at, updated_at, last_used_at, use_count,
         embedding BLOB, embed_model TEXT)
```

## 6. Your controls

| Where | Command |
|---|---|
| chat | `/memory` list · `/remember TEXT` · `/forget ID` · `/memory off` / `on` (this session) |
| chat, natural | "what do you remember about me?" · "forget that I live in Hyderabad" |
| CLI | `jarvis memory` · `jarvis memory search Q` · `jarvis memory forget ID` · `jarvis memory export FILE` · `jarvis memory clear` |
| config | `[memory] enabled`, `auto_learn`, `top_k`, `min_score`, `core_limit` |
| API | `GET/POST /v1/memories`, `PATCH/DELETE /v1/memories/{id}` |

## 7. Evals

- `memory` category (sandbox has seeded memories): uses recalled facts, saves on
  "remember…", forgets on "forget…", lists on "what do you know about me".
- `memory-extract` cases test the extractor directly: right facts pulled out, no
  junk from small talk, updates instead of duplicates, no passwords.

## 8. Done when

Automated: vectors, store CRUD/search/dedupe, keyword fallback, recall ranking &
threshold, prompt layering (core in system, recalled in last message, raw message
stored), extractor ops (add/update/delete, skips junk), worker thread, memory tools,
API, eval kinds.

On your PC:
- [ ] `ollama pull qwen3-embedding:0.6b`; doctor shows memory ✓
- [ ] Tell Jarvis a few facts; see "💾 Remembered"; `/new`; ask about them → it knows
- [ ] "Actually I moved to Bangalore" → the old city memory is *updated*, not duplicated
- [ ] "What do you remember about me?" → accurate list
- [ ] "Forget where I work" → gone from `/memory`
- [ ] `jarvis eval --category memory,memory-extract` ≥ 90%
