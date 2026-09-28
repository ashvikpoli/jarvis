# Jarvis

A local-first personal AI assistant for Windows, built layer by layer.

**Current phase: 1 — Conversation.** Streaming, interruptible chat with a local model;
conversations saved in SQLite and resumable; context-window management; an eval
harness for choosing models. Specs: [`docs/phase-0.md`](docs/phase-0.md),
[`docs/phase-1.md`](docs/phase-1.md).

## Setup (Windows, native — not WSL)

We run natively on Windows because later phases control the Windows desktop.

1. **NVIDIA driver** — current Game Ready or Studio driver for the RTX 5070 Ti.
   Check: `nvidia-smi` in PowerShell shows the GPU.
2. **Ollama** — install from <https://ollama.com/download/windows>. It runs as a
   background app and uses CUDA automatically.
3. **uv** (Python package/project manager) — in PowerShell:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
4. **Git** — <https://git-scm.com/download/win> (if you don't have it).
5. **Project**:
   ```powershell
   cd jarvis
   uv sync                   # creates .venv, installs Python 3.12 if needed + deps
   ollama pull qwen3:14b     # ~9 GB; or whichever chat model you pick
   uv run jarvis doctor      # everything should be green
   uv run jarvis chat
   ```

## Usage

```powershell
uv run jarvis chat                          # talk to Jarvis (new conversation)
uv run jarvis chat --resume                 # continue the most recent conversation
uv run jarvis chat --resume 3               # continue conversation #3
uv run jarvis chat --model ollama:qwen3:8b  # try another model
uv run jarvis chat --show-thinking          # turn reasoning on and show it
uv run jarvis chat --echo                   # offline, no model needed
uv run jarvis history                       # list conversations
uv run jarvis history 3                     # print conversation #3
uv run jarvis eval                          # score the chat model on evals/cases.toml
uv run jarvis eval --models ollama:qwen3:14b,ollama:qwen3:8b   # compare models
uv run jarvis serve                         # HTTP API on http://127.0.0.1:8765/docs
uv run jarvis doctor                        # diagnose setup, VRAM fit, latency
uv run pytest                               # tests
```

In chat: `/new`, `/list`, `/resume ID`, `/title TEXT`, `/think on|off`,
`/model REF`, `/help`, `/exit`. **Ctrl+C while Jarvis is answering interrupts
it** (the partial answer is saved); Ctrl+C at the prompt quits.

After every reply you'll see `answer in X ms · Y tok/s` — "answer in" is the time
to the first word of the actual answer, the number that matters for voice.

Stream from the API:

```powershell
curl.exe -N http://127.0.0.1:8765/v1/chat -H "Content-Type: application/json" `
  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'
```

## Configuration

Copy only the keys you want to change into `config/local.toml` (git-ignored):

```toml
[models]
roles = { chat = "ollama:qwen3:8b" }

[models.options.chat]
think = true    # let the model reason first (slower, sometimes smarter)

[conversation]
context_tokens = 16384   # longer memory within a chat; costs more VRAM
```

Env vars override everything: `$env:JARVIS_MODELS__ROLES__CHAT = "ollama:qwen3:8b"`.

Data lives in `data/` (git-ignored): `jarvis.db` (conversations),
`logs/jarvis.log` (JSON lines with latency for every reply), `evals/` (eval results).

## Layout

```
src/jarvis/
  app.py              composition root (wires everything)
  core/               config, logging, shared types
  providers/          model backends + ModelManager (roles → models, per-role options)
  storage/            SQLite database + conversation store
  conversation/       ConversationManager, PromptBuilder (context window)
  evals/              eval checks + runner
  interfaces/         cli.py, api.py
evals/cases.toml      eval cases — add one whenever Jarvis gets something wrong
```

## Roadmap

0 Foundation · 1 Conversation + evals · 2 Tools (MCP) + permissions · 3 Memory ·
4 Voice MVP · 5 Desktop control + vision · 6 Automation · 7 Advanced voice ·
8 Polish + packaging
