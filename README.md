# Jarvis

A local-first personal AI assistant for Windows, built layer by layer.

**Current phase: 0 — Foundation.** Streaming chat with a local model, interruptible,
with model choice driven by config. See [`docs/phase-0.md`](docs/phase-0.md).

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
uv run jarvis chat                          # talk to Jarvis
uv run jarvis chat --model ollama:qwen3:8b  # try another model
uv run jarvis chat --echo                   # offline, no model needed
uv run jarvis chat --show-thinking          # show reasoning traces
uv run jarvis serve                         # HTTP API on http://127.0.0.1:8765/docs
uv run jarvis doctor                        # diagnose setup, VRAM fit, latency
uv run pytest                               # tests
```

In chat: `/model <ref>`, `/reset`, `/help`, `/exit`. **Ctrl+C while Jarvis is
answering interrupts it**; Ctrl+C at the prompt quits.

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

[providers.ollama]
think = false   # faster answers from thinking models
```

Env vars override everything: `$env:JARVIS_MODELS__ROLES__CHAT = "ollama:qwen3:8b"`.

Logs: `data/logs/jarvis.log` (JSON lines, includes time-to-first-token and tok/s
for every reply).

## Layout

```
src/jarvis/
  app.py              composition root (wires everything)
  core/               config, logging, shared types
  providers/          model backends + ModelManager (roles → models)
  interfaces/         cli.py, api.py
```

## Roadmap

0 Foundation · 1 Conversation + evals · 2 Tools (MCP) + permissions · 3 Memory ·
4 Voice MVP · 5 Desktop control + vision · 6 Automation · 7 Advanced voice ·
8 Polish + packaging
