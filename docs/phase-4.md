# Phase 4 — Voice MVP

## 1. Goal

Talk to Jarvis and hear it answer. **Tap Right Ctrl** (anywhere in Windows — no
need to focus the terminal), say what you want, and pause. Jarvis stops listening
on its own, works out what you said, answers in a British voice, and can still use
every tool from Phase 2 and the memory from Phase 3. Tap again while it's talking
to cut it off and say something else.

Everything runs on your PC. Nothing is sent anywhere.

## 2. How a turn flows

```
 tap Right Ctrl ─► 🔊 "listening" beep
   │
   ├─ Microphone (always open, 16 kHz) ─► Silero VAD, every 32 ms: speech?
   │     starts when you start talking · stops after 0.7 s of quiet
   │     (0.3 s before you started is kept, so the first word isn't clipped)
   │
   ├─ faster-whisper  distil-large-v3.5 on the GPU (float16)   ~0.1–0.3 s
   │
   ├─ conversation (same as chat: memory recall, tools, approvals, saved history)
   │     + a "voice mode" note in the system prompt: short, plain spoken answers
   │
   ├─ reply streams ─► sentence splitter ─► Kokoro TTS (CPU) ─► speakers
   │     sentence 1 plays while sentence 2 is being generated;
   │     the first chunk is cut at a comma so Jarvis starts talking sooner
   │
   └─ "heard in 180 ms · answer in 40 ms · spoke 650 ms after you stopped"
```

**The number that matters** is *spoke N ms after you stopped* — from the end of
your speech (after the 0.7 s pause) to Jarvis's first sound. Target: < 1 s.

## 3. Choices

| | Choice | Why |
|---|---|---|
| Talk key | **tap** Right Ctrl, auto-stop on silence | your pick; a clean tap only — Right Ctrl + C etc. don't trigger |
| Hearing | faster-whisper `distil-large-v3.5`, English | fast, accurate, ~1.5 GB VRAM; `large-v3-turbo` if you ever want other languages |
| GPU | float16 (CTranslate2 disables int8 on RTX 50) | CUDA libraries come from pip (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, ~1.2 GB); falls back to CPU automatically |
| Silence detection | Silero VAD v6 (ships inside faster-whisper, run with onnxruntime) | no PyTorch needed |
| Voice | Kokoro-82M, voice `bm_george` (British male) | best local quality at this size, real time on the CPU, leaves the GPU to the LLM |
| Kokoro runtime | our own ~150-line runner | the `kokoro-onnx` package doesn't install on Python 3.14; same pipeline (espeak-ng phonemes → ONNX) |
| Approvals | asked out loud — answer "yes", "no", "always", or type y / n / a | your voice can't grant destructive actions "always", same as chat |
| Not hearing itself | half-duplex: listening stops Jarvis talking; right after it spoke, the pre-roll is skipped and the first ~0.25 s (beep + speaker buffer) can't trigger speech | real echo cancellation comes with barge-in in Phase 7 |
| Barge-in | by key (tap while it talks) or Enter | talking over Jarvis to interrupt comes in Phase 7 (needs echo cancellation) |

## 4. Components

```
src/jarvis/voice/
  vad.py         SileroVAD (streaming), Endpointer (start/stop logic), utterance_slice
  audio.py       Microphone (always-open input + record()), Speaker (queued clips, instant stop), earcons
  stt.py         WhisperSTT: cuda → cpu fallback, warm-up, hallucination filter, CUDA DLL setup
  tts.py         KokoroTTS, model download, SpeechQueue (synthesize-ahead worker thread)
  text.py        speech_text (markdown/paths/URLs → speakable), SentenceSplitter, parse_approval
  hotkey.py      global hotkey via pynput: tap-a-key or key combo
  session.py     VoiceSession: event loop (taps + typed lines), turns, interrupts, voice approvals
  components.py  builds the pieces from settings; wav writing; word accuracy for self-tests
```

`ConversationManager.send()` gained `system_note=` (the voice-mode style). It's
constant for the session, so Ollama's prompt cache still works.

## 5. Files

Models live in `data/models/` (never in git):

| File | Size | From |
|---|---|---|
| `kokoro/kokoro-v1.0.onnx` + `voices-v1.0.bin` | 325 + 28 MB | GitHub releases of kokoro-onnx |
| `whisper/…distil-large-v3.5…` | ~1.5 GB | Hugging Face |

`jarvis voice setup` downloads both once.

## 6. Commands

| Command | What |
|---|---|
| `jarvis voice setup` | download the models, load speech recognition (shows GPU/CPU) |
| `jarvis voice test` | speaks a sentence, feeds it back through Whisper (round-trip accuracy), then a mic check |
| `jarvis voice` | voice mode (`--resume [ID]`, `--model`, `--voice NAME`, `--no-speak`) |
| `jarvis voice say "text"` | hear any sentence (`--voice`, `--save out.wav`) |
| `jarvis voice voices [--play]` | the voices, and what they sound like |
| `jarvis voice devices` | microphones & speakers, to pick one in config |

In voice mode: tap to talk · tap while it talks = interrupt · Enter = stop talking ·
type messages as usual · `/new` · `/voice NAME` · `/voices` · `/mute` `/unmute` ·
`/exit` or Ctrl+C.

Config — `[voice]` in `config/local.toml`: `hotkey`, `stt_model`, `stt_device`,
`hotwords`, `silence_ms`, `start_timeout_s`, `vad_threshold`, `input_device`,
`tts_voice`, `tts_speed`, `tts_model`, `output_device`, `earcons`, `speak`, `style`.

## 7. VRAM

~6 GB other apps + qwen3:8b ~5–6 GB + embeddings ~1 GB + Whisper ~1.5–2 GB ≈ 14–15 GB
of 16. Kokoro runs on the CPU. If `jarvis doctor` shows the chat model partly on CPU
after voice starts, set `stt_model = "distil-small.en"` or `stt_device = "cpu"`.

## 8. Done when

Automated (cloud + your PC): text cleanup, sentence splitting, yes/no parsing, VAD
endpointing, mic recording loop, tap detection, CUDA DLL discovery, cuda→cpu
fallback, speech queue (order, stop, errors), voice sessions end to end with fakes
(turn, nothing heard, typed input, approval yes/no/unclear/typed, interrupt +
re-listen, partial reply saved), and — once the models are downloaded — real Kokoro
speech detected by the real VAD.

On your PC:
- [ ] `uv sync`, `uv run jarvis voice setup` → "Speech recognition: … on cuda (float16)"
- [ ] `uv run jarvis voice test` → hear the sentence; round trip ≥ 80 %; mic hears you
- [ ] `uv run jarvis voice` → greeting; tap Right Ctrl, "what time is it?" → spoken answer
- [ ] "Open Spotify" → opens; "remind me in 2 minutes to stretch" → reminder spoken + toast
- [ ] "Create a file test.txt on my desktop" → asks out loud → "yes" → created
- [ ] Ask for something long, tap Right Ctrl mid-answer → stops, listens
- [ ] Right Ctrl + C in another app does **not** start listening
- [ ] "spoke … ms after you stopped" mostly under 1000 ms
- [ ] `jarvis doctor` still shows the chat model 100 % on GPU while voice is running
