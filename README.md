# Jarvis

A local-first personal AI assistant for Windows, built layer by layer.

**Current phase: 2 — Tools.** Jarvis can work with your files, open apps and
websites, check your PC's status, and keep notes and reminders — every action goes
through a permission check and is recorded in an audit log. Built on Phase 1's
saved, resumable conversations and eval harness.
Specs: [phase 0](docs/phase-0.md) · [phase 1](docs/phase-1.md) · [phase 2](docs/phase-2.md).
Full test checklist: [docs/testing.md](docs/testing.md).

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
   ollama pull qwen3:8b      # ~5 GB; the default chat model (picked by evals)
   uv run jarvis doctor      # everything should be green
   uv run jarvis chat
   ```

## Usage

```powershell
uv run jarvis chat                          # talk to Jarvis (new conversation)
uv run jarvis chat --resume                 # continue the most recent conversation
uv run jarvis chat --resume 3               # continue conversation #3
uv run jarvis chat --model ollama:qwen3:14b # try another model
uv run jarvis chat --show-thinking          # turn reasoning on and show it
uv run jarvis chat --echo                   # offline, no model needed
uv run jarvis history                       # list conversations
uv run jarvis history 3                     # print conversation #3
uv run jarvis eval                          # score the chat model on evals/cases.toml
uv run jarvis eval --models ollama:qwen3:14b,ollama:qwen3:8b   # compare models
uv run jarvis tools                         # what Jarvis can do + what needs your OK
uv run jarvis audit                         # every tool call Jarvis has made
uv run jarvis reminders                     # upcoming reminders
uv run jarvis apps [name]                   # installed apps Jarvis can open
uv run jarvis notify                        # test Windows notifications
uv run jarvis serve                         # HTTP API on http://127.0.0.1:8765/docs
uv run jarvis doctor                        # diagnose setup, VRAM fit, latency
uv run pytest                               # tests
```

In chat: `/new`, `/list`, `/resume ID`, `/title TEXT`, `/think on|off`,
`/tools`, `/reminders`, `/model REF`, `/help`, `/exit`. **Ctrl+C while Jarvis is answering interrupts
it** (the partial answer is saved); Ctrl+C at the prompt quits.

After every reply you'll see `answer in X ms · Y tok/s` — "answer in" is the time
to the first word of the actual answer, the number that matters for voice.

Stream from the API:

```powershell
curl.exe -N http://127.0.0.1:8765/v1/chat -H "Content-Type: application/json" `
  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'
```

## What Jarvis can do (Phase 2)

Try: *"what's in my Downloads folder?"*, *"find my tax receipt"*, *"create
shopping.txt on my Desktop with milk and eggs"*, *"open Spotify"*, *"open
youtube.com"* (opens the app instead if one is installed, unless you say "website"), *"how much RAM am I using?"*, *"note that the plumber's number is
555-0142"*, *"remind me in 10 minutes to stretch"* → *"actually make it 20"*.

| Permission | Tools |
|---|---|
| runs automatically | list/search/read files, system info, time, list apps, notes, reminders, open apps & websites |
| asks you first | create files/folders, move/rename, delete (→ Recycle Bin), overwrite |

- File access is limited to **Documents, Desktop, Downloads** (OneDrive-aware). Change
  in `[tools.files] roots`.
- At an approval prompt: `y` = once, `n` = no, `a` = always for this session (not
  offered for delete/overwrite).
- Reminders pop up as Windows notifications **while Jarvis is running** (`chat` or
  `serve`). A background service comes in Phase 6.

## Configuration

Copy only the keys you want to change into `config/local.toml` (git-ignored):

```toml
[models]
roles = { chat = "ollama:qwen3:14b" }

[models.options.chat]
think = true    # let the model reason first (slower, sometimes smarter)

[conversation]
context_tokens = 16384   # longer memory within a chat; costs more VRAM

[tools.permissions.overrides]
files_write = "allow"    # stop asking before creating new files
files_delete = "deny"    # never delete

[tools.files]
roots = ["Documents", "Desktop", "Downloads", "D:\\Projects"]
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
  agent/              AgentLoop: model ⇄ tools, permissions, audit
  tools/              tool framework + files, system/apps, notes/reminders
  evals/              eval checks + runner
  interfaces/         cli.py, api.py
evals/cases.toml      eval cases — add one whenever Jarvis gets something wrong
```

## Roadmap

0 Foundation ✅ · 1 Conversation + evals ✅ · 2 Tools + permissions ✅ · 3 Memory ·
4 Voice MVP · 5 Desktop control + vision · 6 Automation · 7 Advanced voice ·
8 Polish + packaging
