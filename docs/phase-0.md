# Phase 0 — Lean Foundation

## 1. Goal

Get a skeleton that talks. By the end of Phase 0 you can type to Jarvis in a
terminal (or over HTTP) and get a **streamed** answer from a local model on your
RTX 5070 Ti, **interrupt it mid-sentence** with Ctrl+C, and swap models or
providers purely through config.

Nothing "smart" yet — no memory, no tools. But the two properties voice will
need later (streaming + cancellation) are baked in from day one.

**Deliberately NOT in Phase 0** (and why):

| Deferred | To | Reason |
|---|---|---|
| SQLite / conversation storage | Phase 1 | Nothing to persist yet |
| Vector DB (ChromaDB) | Phase 3 (Memory) | Only memory needs it |
| Event bus | When 2+ components need pub/sub | Premature otherwise |
| DI framework | Never (probably) | Constructor injection + one composition root is enough |
| GUI | Phase 8 | CLI + HTTP API is sufficient to build the brain |

## 2. Architecture

```
            ┌──────────────┐     ┌──────────────┐
            │  CLI (REPL)  │     │ FastAPI (HTTP│  ← future: voice, widget, phone
            └──────┬───────┘     │  + SSE)      │
                   │             └──────┬───────┘
                   └────────┬───────────┘
                            ▼
                    ┌───────────────┐
                    │   JarvisApp   │  composition root: builds & owns everything
                    └───────┬───────┘
                            ▼
                    ┌───────────────┐
                    │ ModelManager  │  role → (provider, model), warmup/unload,
                    └───────┬───────┘  latency metrics (TTFT, tok/s)
                ┌───────────┼────────────┐
                ▼           ▼            ▼
         OllamaProvider  EchoProvider  (future) Anthropic/OpenAI provider
                │
                ▼
          Ollama (native Windows, CUDA) → Qwen/Gemma/etc on GPU
```

Key idea: **interfaces talk to `JarvisApp`, never to a provider directly.**
Code asks for a *role* (`"chat"`, later `"planner"`, `"vision"`, `"fast"`), and
config decides which model fills it. This is what makes hybrid local/cloud
routing and VRAM swapping possible later without touching callers.

## 3. Folder structure

```
jarvis/
├── pyproject.toml            # uv-managed, Python ≥3.11 (3.12 recommended)
├── config/
│   ├── default.toml          # committed defaults
│   └── local.toml            # your overrides (git-ignored)
├── docs/phase-0.md
├── src/jarvis/
│   ├── __main__.py           # python -m jarvis
│   ├── app.py                # JarvisApp (composition root)
│   ├── core/
│   │   ├── config.py         # Settings (TOML + env vars)
│   │   ├── logging.py        # console + rotating JSON file logs
│   │   └── types.py          # Message, StreamChunk, Usage
│   ├── providers/
│   │   ├── base.py           # ModelProvider interface
│   │   ├── ollama.py         # Ollama implementation
│   │   ├── echo.py           # offline fake, for tests/demos
│   │   └── manager.py        # ModelManager + ModelRef
│   └── interfaces/
│       ├── cli.py            # `jarvis chat | serve | doctor`
│       └── api.py            # FastAPI app
└── tests/
```

## 4. Configuration

Precedence (highest wins): **env vars → `config/local.toml` → `config/default.toml` → code defaults.**
Env vars use prefix `JARVIS_` and `__` for nesting, e.g.
`JARVIS_MODELS__ROLES__CHAT="ollama:qwen3:8b"`.

Model references are `"<provider>:<model>"` — split on the first colon, so
Ollama tags like `qwen3:14b` work: `ollama:qwen3:14b`.

## 5. Interfaces (contracts)

```python
class ModelProvider(ABC):
    name: str
    async def stream_chat(messages, model, *, options=None) -> AsyncIterator[StreamChunk]
    async def list_models() -> list[str]
    async def loaded_models() -> list[LoadedModel]     # what's in VRAM now
    async def load(model) -> None                        # warm up
    async def unload(model) -> None                      # free VRAM
    async def health() -> ProviderHealth
    async def aclose() -> None

class ModelManager:
    def resolve(role) -> tuple[ModelProvider, str]
    async def stream_chat(messages, role="chat", **opts) -> AsyncIterator[StreamChunk]
    async def warmup(role) / unload(role)
    async def health() -> dict[str, ProviderHealth]

class JarvisApp:
    @classmethod
    def create(settings=None, providers=None) -> JarvisApp
    async def chat_stream(messages, role="chat") -> AsyncIterator[StreamChunk]
    async def aclose()
```

**Cancellation contract:** every stream must stop promptly when its asyncio task
is cancelled and must release the HTTP connection. Voice barge-in in Phase 7
relies on this.

## 6. HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | App + provider health |
| GET | `/v1/models` | Models available per provider, and which are loaded |
| POST | `/v1/chat` | `{messages, role?, stream?}` → SSE stream or JSON |

SSE events: `{"type":"token","text":…}`, `{"type":"thinking","text":…}`,
`{"type":"done","usage":{…}}`, `{"type":"error","message":…}`.

## 7. Data flow (one turn)

```
user text → CLI → JarvisApp.chat_stream(history)
         → ModelManager.resolve("chat") → OllamaProvider.stream_chat
         → POST /api/chat (NDJSON stream) → StreamChunk… → CLI prints tokens
Ctrl+C  → asyncio task cancelled → httpx stream closed → Ollama stops generating
done    → Usage{ttft_ms, tokens, tok/s} logged
```

## 8. Testing — Phase 0 is done when

Automated (`uv run pytest`):
- [ ] Config loads defaults, TOML override, and env override in the right order
- [ ] `ModelRef.parse` handles `ollama:qwen3:14b`
- [ ] Echo provider streams; cancelling mid-stream stops it
- [ ] Ollama provider parses NDJSON (content + thinking + final usage) against a mocked server
- [ ] API `/health`, `/v1/models`, `/v1/chat` (JSON and SSE) work

Manual, on your PC:
- [ ] `uv run jarvis doctor` is all green (Ollama reachable, model pulled, GPU visible)
- [ ] `uv run jarvis chat` streams a reply; first token < 1 s once warm
- [ ] Ctrl+C mid-reply stops generation and returns to the prompt (doesn't exit)
- [ ] Changing the model in `config/local.toml` switches models with no code change
- [ ] `uv run jarvis serve` + `curl` to `/v1/chat` streams

## 9. Future expansion hooks

- **Roles** → Phase 1 adds `fast`/`planner`; Phase 5 adds `vision` (manager swaps VRAM).
- **Provider interface** → add `AnthropicProvider` for hybrid routing with no caller changes.
- **StreamChunk + cancellation** → Phase 4 voice pipes tokens straight into TTS and barges in.
- **FastAPI** → every future UI (voice daemon, tray widget) is just another client.
- **Usage metrics** → become the latency dashboard for the voice budget.

## 10. Git

`main` + short-lived feature branches; tag each finished phase (`phase-0`).
