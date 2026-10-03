# Jarvis

A local-first personal AI assistant for Windows, built layer by layer.

**Current phase: 4.5 — Obsidian second brain.** Everything Jarvis knows (memory,
notes, reminders, conversations, a daily journal) lives as Markdown pages in your
Obsidian vault — edit them and Jarvis learns from your edits; it also searches the
notes you write. Phase 4: tap **Right Ctrl**, speak, and Jarvis answers out loud in a
British voice — everything local (faster-whisper on the GPU, Kokoro TTS).
Built on Phase 3's long-term memory, Phase 2's tools (files, apps, notes, reminders —
every action permission-checked and audited) and Phase 1's saved conversations and evals.
Specs: [phase 0](docs/phase-0.md) · [phase 1](docs/phase-1.md) · [phase 2](docs/phase-2.md) · [phase 3](docs/phase-3.md) · [phase 4](docs/phase-4.md) · [phase 4.5](docs/phase-4.5.md).
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
   ollama pull qwen3-embedding:0.6b   # 639 MB; lets memory search by meaning
   uv run jarvis doctor      # everything should be green
   uv run jarvis chat
   uv run jarvis voice setup # voice models: ~350 MB voice + ~1.5 GB speech recognition
   uv run jarvis voice test  # hear Jarvis, check the mic
   uv run jarvis voice       # talk: tap Right Ctrl, speak, pause
   ```

## Usage

```powershell
uv run jarvis chat                          # talk to Jarvis (new conversation)
uv run jarvis chat --resume                 # continue the most recent conversation
uv run jarvis chat --resume 3               # continue conversation #3
uv run jarvis chat --model ollama:qwen3:14b # try another model
uv run jarvis chat --show-thinking          # turn reasoning on and show it
uv run jarvis chat --echo                   # offline, no model needed
uv run jarvis voice                         # voice mode (--resume, --voice bm_lewis, --no-speak)
uv run jarvis voice setup | test            # download voice models | self-test speakers, Whisper, mic
uv run jarvis voice say "Hello sir"         # hear a sentence (--save hello.wav)
uv run jarvis voice voices --play           # hear the voices
uv run jarvis voice devices                 # microphones & speakers
uv run jarvis vault                         # Obsidian vault status (sync | search TEXT | rebuild | open)
uv run jarvis history                       # list conversations
uv run jarvis history 3                     # print conversation #3
uv run jarvis eval                          # score the chat model on evals/cases.toml
uv run jarvis eval --models ollama:qwen3:14b,ollama:qwen3:8b   # compare models
uv run jarvis memory                        # what Jarvis remembers about you
uv run jarvis memory search sister          # search memories
uv run jarvis memory forget 12              # delete a memory (also: clear, export FILE)
uv run jarvis tools                         # what Jarvis can do + what needs your OK
uv run jarvis audit                         # every tool call Jarvis has made
uv run jarvis reminders                     # upcoming reminders
uv run jarvis apps [name]                   # installed apps Jarvis can open
uv run jarvis notify                        # test Windows notifications
uv run jarvis serve                         # HTTP API on http://127.0.0.1:8765/docs
uv run jarvis doctor                        # diagnose setup, VRAM fit, latency
uv run python -m pytest                     # tests (python -m: Smart App Control blocks pytest.exe)
```

In chat: `/new`, `/list`, `/resume ID`, `/title TEXT`, `/think on|off`,
`/memory` (`/memory off|on`), `/remember TEXT`, `/forget ID`,
`/tools`, `/reminders`, `/model REF`, `/help`, `/exit`. **Ctrl+C while Jarvis is answering interrupts
it** (the partial answer is saved); Ctrl+C at the prompt quits.

After every reply you'll see `answer in X ms · Y tok/s` — "answer in" is the time
to the first word of the actual answer, the number that matters for voice.

Stream from the API:

```powershell
curl.exe -N http://127.0.0.1:8765/v1/chat -H "Content-Type: application/json" `
  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'
```

## Obsidian second brain (Phase 4.5)

Your vault (`Documents\obsidian_notes`) stays in sync while Jarvis runs:
**Home**, **About me**, a page per person / project / topic, **Notes**, **Reminders**
(a checklist), a **Journal** page per day and every **conversation**. Start with
`uv run jarvis vault sync`, then open the vault in Obsidian.

- Jarvis's part of a page sits between `%% jarvis:start %%` and `%% jarvis:end %%`.
  Edit, add or delete lines there and Jarvis updates its memory (delete a *line* to
  make it forget — deleting a whole page keeps the facts). Tick a reminder to finish
  it, or add `- [ ] Pay rent 📅 2026-11-01 9am`.
- Everything outside the markers is yours: Jarvis never rewrites it — and searches
  it (*"what did I write about pricing?"*).
- Drop files into **Inbox/** — Jarvis learns from them and files them in Notes.
- Passwords and codes never leave Jarvis (🔒 in the vault).
- Details and safety rules: [docs/phase-4.5.md](docs/phase-4.5.md).

## Voice (Phase 4)

`uv run jarvis voice`, then **tap Right Ctrl** (from any app — the terminal doesn't
need focus), speak, and pause: Jarvis stops listening by itself after ~0.7 s of quiet
and answers out loud. **Tap again while it's talking** to cut it off and speak;
press **Enter** to just stop it. Typing still works and is answered aloud.

- Everything works as in chat — tools, memory, reminders (spoken too). When
  something needs your OK, Jarvis asks out loud: say *"yes"*, *"no"* or *"always"*,
  or type `y` / `n` / `a`.
- After each turn: `heard in 180 ms · answer in 40 ms · spoke 650 ms after you
  stopped` — the last number is the delay you feel.
- Only a clean tap counts: Right Ctrl + C in another app doesn't start listening.
  Change the key in `[voice] hotkey` (e.g. `"f9"` or `"<ctrl>+<alt>+j"`).
- Voice: `bm_george` (British male). Hear the others with `jarvis voice voices --play`;
  switch with `/voice bm_lewis` or `[voice] tts_voice`.
- If it cuts you off mid-sentence, raise `[voice] silence_ms` (e.g. 1000). If noise
  keeps it listening, raise `vad_threshold` to 0.6–0.7.
- Speech recognition runs on the GPU (~1.5 GB VRAM) and falls back to the CPU if
  CUDA isn't working — `jarvis voice setup` says which.

## Memory (Phase 3)

Just talk: *"I'm Adi, I run Exinent"*, *"my sister Priya lives in Pune"*, *"I prefer
short answers"*. After a reply you'll see **💾 Remembered: …** when Jarvis learns
something (it learns in the background, so replies never wait). In a new
conversation it already knows. Corrections update the old memory instead of adding
a duplicate (*"I moved to Bangalore"* → ✏️ Updated).

- *"What do you remember about me?"*, *"forget that I have a sister"*, `/memory`,
  `/forget ID`, `jarvis memory export me.json`.
- Never remembered as memories: passwords/codes (use notes), passing moods
  ("tired today"), one-off requests, things only Jarvis said.
- Stays on your PC in `data/jarvis.db`. Turn off for a session with `/memory off`,
  or permanently with `[memory] enabled = false`; stop learning with `auto_learn = false`.

## What Jarvis can do (Phase 2)

Try: *"what's in my Downloads folder?"*, *"find my tax receipt"*, *"create
shopping.txt on my Desktop with milk and eggs"*, *"open Spotify"*, *"open
youtube.com"* (opens the app instead if one is installed, unless you say "website"), *"how much RAM am I using?"*, *"note that the plumber's number is
555-0142"*, *"remind me in 10 minutes to stretch"* → *"actually make it 20"*.

| Permission | Tools |
|---|---|
| runs automatically | list/search/read/open files, system info, time, list apps, notes, reminders, open apps & websites |
| asks you first | create files/folders, move/rename, delete (→ Recycle Bin), overwrite |

**File zones**

| Where | Read | Create / change / delete |
|---|---|---|
| Your folders: Documents, Desktop, Downloads, this Jarvis folder | ✅ | asks first |
| Everywhere else on the PC | ✅ read-only | asks **every time** (red, "outside your folders", no "always") |
| Private data: browser profiles, SSH keys, Windows credentials, password managers | ❌ | ❌ |
| Windows, Program Files, Jarvis's own `config/` and `data/` | ✅ | ❌ never, even with approval |

Jarvis can't edit its own settings — otherwise it could grant itself permissions.
Change your folders in `[tools.files] roots`; set `outside_writes = "deny"` to forbid
changes outside them entirely, or `read_anywhere = false` to restrict reading too.

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
roots = ["Documents", "Desktop", "Downloads", "Jarvis", "D:\\Projects"]  # list them all
outside_writes = "deny"
```

If Windows says *"An Application Control policy has blocked this file"*, that's Smart
App Control blocking an unsigned launcher in `.venv\Scripts`. Use the module form
instead: `uv run python -m jarvis chat`, `uv run python -m pytest`. Don't disable
Smart App Control (it can't be re-enabled without a reset).

Env vars override everything: `$env:JARVIS_MODELS__ROLES__CHAT = "ollama:qwen3:8b"`.

Data lives in `data/` (git-ignored): `jarvis.db` (conversations, memories, notes, reminders),
`logs/jarvis.log` (JSON lines with latency for every reply), `evals/` (eval results),
`models/` (voice models).

## Layout

```
src/jarvis/
  app.py              composition root (wires everything)
  core/               config, logging, shared types
  providers/          model backends + ModelManager (roles → models, per-role options)
  storage/            SQLite database + conversation store
  conversation/       ConversationManager, PromptBuilder (context window)
  agent/              AgentLoop: model ⇄ tools, permissions, audit
  memory/             long-term memory: store, embeddings, recall, background learning
  voice/              mic + VAD, Whisper STT, Kokoro TTS, hotkey, VoiceSession
  vault/              Obsidian pages ⇄ database (two-way sync), vault search index
  tools/              tool framework + files, system/apps, notes/reminders, memory
  evals/              eval checks + runner
  interfaces/         cli.py, api.py
evals/cases.toml      eval cases — add one whenever Jarvis gets something wrong
```

## Roadmap

0 Foundation ✅ · 1 Conversation + evals ✅ · 2 Tools + permissions ✅ · 3 Memory (testing) ·
4 Voice MVP (testing) · 4.5 Obsidian second brain (testing) · 5 Desktop control + vision · 6 Automation · 7 Advanced voice ·
8 Polish + packaging
