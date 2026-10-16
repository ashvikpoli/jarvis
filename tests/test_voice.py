"""Phase 4 — voice. Pure logic is tested directly; audio/models are replaced by fakes.

Tests that need the real Kokoro files run only when they're downloaded
(`jarvis voice setup`, or JARVIS_TEST_KOKORO_DIR).
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from jarvis.core.config import VoiceSettings, load_settings
from jarvis.core.types import StreamChunk, ToolCall, Usage
from jarvis.providers.scripted import ScriptedProvider
from jarvis.voice import components as vc
from jarvis.voice.audio import Microphone, collect, resample, tone
from jarvis.voice.hotkey import TapDetector, describe
from jarvis.voice.session import Event, VoiceSession
from jarvis.voice.stt import Transcript, clean_transcript, cuda_library_dirs
from jarvis.voice.text import SentenceSplitter, parse_approval, speech_text
from jarvis.voice.tts import SpeechQueue, split_phonemes, trim_silence, voice_language
from jarvis.voice.vad import FRAME, EndpointConfig, Endpointer, utterance_slice

# ---------------------------------------------------------------- text


@pytest.mark.parametrize(
    "raw, spoken",
    [
        ("**Done!** Saved it.", "Done! Saved it."),
        ("Open [the docs](https://x.com/a) now", "Open the docs now"),
        ("See https://www.youtube.com/watch?v=abc", "See youtube.com"),
        (r"Saved to C:\Users\ashvi\Desktop\notes.txt", "Saved to notes.txt"),
        ("- milk\n- eggs", "milk. eggs"),
        ("## Weather\nIt's 31°C ☀️", "Weather. It's 31 degrees Celsius"),
        ("Meeting at 3:45 and 9:05, lunch at 1:00.", "Meeting at 3 45 and 9 oh 5, lunch at 1."),
        ("Run `ls` then:\n```\nrm -rf x\n```", "Run ls then: (the code is on screen)"),
        ("Tom & Jerry", "Tom and Jerry"),
    ],
)
def test_speech_text(raw, spoken):
    assert speech_text(raw) == spoken


def _stream(text: str, splitter: SentenceSplitter) -> list[str]:
    out: list[str] = []
    for i, word in enumerate(text.split(" ")):
        out += splitter.feed(word if i == 0 else " " + word)
    return out + splitter.flush()


def test_sentence_splitter_streams_sentences():
    s = SentenceSplitter()
    parts = _stream("Sure thing. Dr. Rao said the U.S. rate is 3.5 percent. Anything else?", s)
    assert parts == ["Sure thing.", "Dr. Rao said the U.S. rate is 3.5 percent.", "Anything else?"]


def test_sentence_splitter_cuts_first_chunk_at_comma_for_speed():
    s = SentenceSplitter()
    parts = _stream(
        "Right away, I have opened Spotify and started your focus playlist for you now.", s
    )
    assert parts[0] == "Right away,"
    assert " ".join(parts) == (
        "Right away, I have opened Spotify and started your focus playlist for you now."
    )


def test_sentence_splitter_code_blocks_and_no():
    s = SentenceSplitter()
    text = "No. That deletes it. Here you go:\n```python\nimport os\nprint(1)\n```\nDone."
    parts: list[str] = []
    for ch in text:  # worst case: one character at a time
        parts += s.feed(ch)
    parts += s.flush()
    assert parts[:3] == ["No.", "That deletes it.", "Here you go:"]
    assert parts[3].startswith("```") and parts[3].endswith("```")  # one piece
    assert speech_text(parts[3]) == "(the code is on screen)" and parts[4] == "Done."


def test_sentence_splitter_newlines_and_long_runs():
    s = SentenceSplitter(max_chars=40)
    parts = _stream("Title here\nand then a very long clause, with a comma, " + "word " * 30, s)
    assert parts[0] == "Title here"
    assert all(len(p) <= 60 for p in parts)
    assert " ".join(parts).split() == ("Title here and then a very long clause, with a comma, "
                                       + "word " * 30).split()  # fmt: skip


@pytest.mark.parametrize(
    "text, allow, expected",
    [
        ("Yes.", True, "once"),
        ("yeah go ahead", True, "once"),
        ("Sure, do it", True, "once"),
        ("No.", True, "deny"),
        ("nope, don't", True, "deny"),
        ("Cancel that", True, "deny"),
        ("Yes, always.", True, "session"),
        ("always", False, None),
        ("y", True, "once"),
        ("n", True, "deny"),
        ("a", True, "session"),
        ("No wait, yes do it", True, "once"),
        ("I'm not sure", True, None),
        ("not okay", True, "deny"),
        ("that is not correct", True, "deny"),
        ("I don't think that's right", True, "deny"),
        ("maybe", True, None),
        ("y.", True, "once"),
        ("What's the weather?", True, None),
        ("", True, None),
    ],
)
def test_parse_approval(text, allow, expected):
    assert parse_approval(text, allow_session=allow) == expected


# ---------------------------------------------------------------- VAD / endpointing


def _run(ep: Endpointer, probs: list[float]) -> list[str | None]:
    return [ep.push(p) for p in probs]


def test_endpointer_start_and_end():
    cfg = EndpointConfig(silence_ms=320, start_frames=2)  # 320 ms = 10 frames
    ep = Endpointer(cfg)
    events = _run(ep, [0.0] * 5 + [0.9] * 20 + [0.0] * 12)
    assert events.index("start") == 6  # second speech frame
    assert ep.started_at == 5
    assert events[-1] == "end" and events.index("end") == 25 + 9
    assert ep.last_speech == 24
    sl = utterance_slice(ep)
    pre, tail = round(300 / 32), round(200 / 32)
    assert sl == slice(max(0, 5 - pre), 25 + tail)


def test_endpointer_hysteresis_keeps_short_dips():
    ep = Endpointer(EndpointConfig(silence_ms=160))  # 5 frames
    events = _run(ep, [0.9] * 5 + [0.4] * 20 + [0.9] * 3)  # 0.4 > neg threshold 0.35
    assert "end" not in events


def test_endpointer_timeout_and_short_blips():
    ep = Endpointer(EndpointConfig(start_timeout_s=0.32))
    assert _run(ep, [0.0] * 10)[-1] == "timeout"
    assert utterance_slice(ep) is None
    ep = Endpointer(EndpointConfig(silence_ms=96, min_speech_ms=200))
    _run(ep, [0.9, 0.9, 0.9] + [0.0] * 5)  # ~96 ms of "speech": a click
    assert ep.finished == "end" and utterance_slice(ep) is None


def test_endpointer_max_length():
    ep = Endpointer(EndpointConfig(max_s=0.64))  # 20 frames
    events = _run(ep, [0.9] * 40)
    assert events.index("end") == 19  # 20 frames after speech started at frame 0


class FakeVAD:
    """Frames whose samples are ≥ 0.5 count as speech."""

    def reset(self):
        pass

    def __call__(self, frame) -> float:
        return float(frame[0])


def _frames(pattern: str) -> list[np.ndarray]:
    return [np.full(FRAME, 0.9 if c == "#" else 0.0, dtype=np.float32) for c in pattern]


def test_microphone_record_with_fake_stream():
    mic = Microphone(vad_factory=FakeVAD)
    mic._stream = object()  # pretend it's open; we feed frames ourselves
    started = []

    def feed():  # like the audio callback, while record() is running
        time.sleep(0.05)
        for f in _frames("...." + "#" * 15 + "." * 30):
            mic._q.put(f)

    threading.Thread(target=feed, daemon=True).start()
    audio = mic.record(
        threading.Event(), EndpointConfig(silence_ms=320), on_start=lambda: started.append(1)
    )
    assert started == [1]
    assert audio is not None and audio.dtype == np.float32
    assert audio.size % FRAME == 0 and (audio > 0.5).sum() == 15 * FRAME


def test_microphone_record_stop_before_speech_returns_none():
    mic = Microphone(vad_factory=FakeVAD)
    mic._stream = object()
    stop = threading.Event()
    threading.Timer(0.15, stop.set).start()
    assert mic.record(stop, EndpointConfig()) is None


def test_collect_and_resample_and_tone():
    ep = Endpointer(EndpointConfig(silence_ms=96))
    frames = _frames("..########...")
    for f in frames:
        ep.push(FakeVAD()(f))
    audio = collect(frames, ep)
    assert audio.size == 13 * FRAME
    assert resample(np.ones(24000, np.float32), 24000, 16000).size == 16000
    assert 0 < tone([880], 50).max() <= 0.2


def test_microphone_mute_and_no_seed():
    mic = Microphone(vad_factory=FakeVAD)
    mic._stream = object()
    mic._ring.extend(_frames("#" * 8))  # e.g. the tail of Jarvis's own voice

    def feed():
        time.sleep(0.05)
        for f in _frames("#" * 5 + "." * 20):  # only "our beep" — muted
            mic._q.put(f)

    threading.Thread(target=feed, daemon=True).start()
    cfg = EndpointConfig(silence_ms=320, start_timeout_s=0.8)
    assert mic.record(threading.Event(), cfg, seed=False, mute_s=0.2) is None
    assert len(mic._ring) == 0  # the pre-roll never outlives a recording


def test_speaker_queue_epoch_and_first_audio():
    from jarvis.voice.audio import Speaker

    spk = Speaker()
    spk._stream = object()  # no real device: drive the callback by hand
    out = np.zeros((100, 1), dtype=np.float32)
    spk.mark_first_audio()
    spk.play(np.ones(150, dtype=np.float32), 24000)
    assert spk.busy
    spk._callback(out, 100, None, None)
    assert out[:, 0].sum() == 100 and spk.first_audio_at is not None
    spk._callback(out, 100, None, None)
    assert out[:, 0].sum() == 50 and not spk.busy
    old = spk.epoch
    spk.stop()
    spk.play(np.ones(10, dtype=np.float32), 24000, epoch=old)  # stale: dropped
    assert not spk.busy and spk.recently_played()


# ---------------------------------------------------------------- hotkey


def test_tap_detector():
    now = [0.0]
    tap = TapDetector("ctrl_r", clock=lambda: now[0])
    tap.down("ctrl_r")
    tap.down("ctrl_r")  # auto-repeat
    now[0] = 0.2
    assert tap.up("ctrl_r") is True
    tap.down("ctrl_r")
    tap.down("c")  # Right Ctrl + C is a shortcut, not a tap
    assert tap.up("ctrl_r") is False
    tap.down("ctrl_r")
    now[0] = 2.0  # held too long
    assert tap.up("ctrl_r") is False
    assert tap.up("x") is False
    tap.down("ctrl_r")
    tap.other()  # Ctrl + click (e.g. open a link in a new tab) isn't a tap
    assert tap.up("ctrl_r") is False
    assert describe("ctrl_l") == "Left Ctrl" and describe("f9") == "F9"
    assert describe("<ctrl>+<alt>+j") == "Ctrl+Alt+J"


# ---------------------------------------------------------------- STT helpers


def test_clean_transcript():
    assert clean_transcript("  Open   Discord. ") == "Open Discord."
    assert clean_transcript("Thanks for watching!") == ""
    assert clean_transcript("Thank you.", 0.1) == "Thank you."  # real speech
    assert clean_transcript("Thank you.", 0.7) == ""  # Whisper doubted it


def test_cuda_library_dirs(tmp_path, monkeypatch):
    b = tmp_path / "nvidia" / "cublas" / "bin"
    b.mkdir(parents=True)
    (b / "cublas64_12.dll").write_bytes(b"")
    (tmp_path / "nvidia" / "empty" / "bin").mkdir(parents=True)
    monkeypatch.setattr("sys.path", [str(tmp_path)])
    assert cuda_library_dirs() == [b]


def test_word_accuracy():
    assert vc.word_accuracy("remind me at six", "Remind me at six.") == 1.0
    assert vc.word_accuracy("remind me at six", "remind me") == 0.5


def test_voice_settings_and_whisper_absent(tmp_path):
    settings = load_settings(Path(__file__).resolve().parents[1] / "config")
    assert settings.voice.hotkey == "ctrl_l" and settings.voice.tts_voice == "bm_george"
    settings.app.data_dir = tmp_path
    assert vc.whisper_downloaded(settings) is False


# ---------------------------------------------------------------- TTS helpers


def test_tts_helpers():
    assert voice_language("bm_george") == "en-gb" and voice_language("am_adam") == "en-us"
    parts = split_phonemes("ab. " * 200, limit=50)
    assert all(0 < len(p) <= 50 for p in parts)
    assert "".join(parts).replace(" ", "") == ("ab. " * 200).replace(" ", "")
    a = np.concatenate([np.zeros(5000), np.ones(100) * 0.5, np.zeros(5000)]).astype(np.float32)
    assert trim_silence(a, margin=10).size == 120
    assert trim_silence(np.zeros(100)).size == 0


class FakeSpeaker:
    def __init__(self):
        self.played: list[np.ndarray] = []
        self.earcons: list[str] = []
        self.first_audio_at = None
        self._mark = False
        self.stops = 0
        self.epoch = 0

    def play(self, audio, rate=None, *, epoch=None):
        if epoch is not None and epoch != self.epoch:
            return
        self.played.append(audio)
        if self._mark:
            self.first_audio_at, self._mark = time.perf_counter(), False

    def mark_first_audio(self):
        self.first_audio_at, self._mark = None, True

    def stop(self):
        self.stops += 1
        self.epoch += 1

    def earcon(self, name):
        self.earcons.append(name)
        return 0.14

    latency = 0.05

    def recently_played(self, within_s=0.6):
        return bool(self.played)

    @property
    def busy(self):
        return False

    def wait(self, timeout=None):
        return True


class FakeTTS:
    sample_rate = 24000

    def __init__(self, delay: float = 0.0):
        self.said: list[str] = []
        self.delay = delay
        self.voice = "bm_george"

    def synthesize(self, text, voice=None, speed=None):
        time.sleep(self.delay)
        self.said.append(text)
        return np.full(240, 0.1, dtype=np.float32)

    def set_voice(self, voice):
        if voice == "nope":
            raise ValueError("unknown voice")
        self.voice = voice


def test_speech_queue_order_stop_and_errors():
    tts, spk = FakeTTS(delay=0.01), FakeSpeaker()
    q = SpeechQueue(tts, spk)
    q.reset_stats()
    for s in ("One.", "**Two.**", "Three."):
        q.say(s)
    assert q.wait(5)
    assert tts.said == ["One.", "Two.", "Three."]
    assert q.first_audio_at is not None and q.stats.sentences == 3
    q.say("")  # nothing speakable: ignored
    tts.delay = 0.2
    q.say("Long one.")
    q.say("Dropped.")
    time.sleep(0.05)
    q.stop()  # barge-in: queued/in-flight speech is discarded
    assert q.wait(5)
    assert "Dropped." not in tts.said and len(spk.played) == 3

    def boom(*a, **k):
        raise RuntimeError("espeak missing")

    tts.synthesize = boom
    q.say("Hi.")
    assert q.wait(5) and q.errors == ["espeak missing"]
    q.close()


# ---------------------------------------------------------------- the voice session


class FakeMic:
    def __init__(self, clips):
        self.clips = list(clips)
        self.records = 0
        self.calls: list[dict] = []

    def record(self, stop, cfg=None, *, on_start=None, seed=True, mute_s=0.0):
        self.records += 1
        self.calls.append({"seed": seed, "mute_s": mute_s})
        clip = self.clips.pop(0) if self.clips else None
        return None if clip is None else np.zeros(16000, dtype=np.float32)


class FakeSTT:
    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, audio):
        return Transcript(text=self.texts.pop(0) if self.texts else "", ms=12.0, audio_s=1.0)

    def describe(self):
        return "fake"


class SlowProvider(ScriptedProvider):
    """Streams a word every 80 ms so there's time to interrupt."""

    async def stream_chat(self, messages, model, *, options=None, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        for w in "This answer is long and will be cut off before it ends".split():
            await asyncio.sleep(0.08)
            yield StreamChunk(text=w + " ")
        yield StreamChunk(done=True, usage=Usage(answer_ttft_ms=5, total_ms=50))


@pytest.fixture
def voice(make_app):
    from rich.console import Console

    def build(turns=(), heard=(), clips=None, provider=None):
        app, scripted = make_app(turns)
        if provider is not None:
            app.models.providers["scripted"] = provider
        tts, spk = FakeTTS(), FakeSpeaker()
        speech = SpeechQueue(tts, spk)
        mic = FakeMic(clips if clips is not None else ["clip"] * len(heard))
        session = VoiceSession(
            app,
            VoiceSettings(),
            stt=FakeSTT(heard),
            mic=mic,
            speech=speech,
            console=Console(record=True, width=120),
            read_stdin=False,
        )
        return session, app, provider or scripted, tts

    return build


async def _drive(session: VoiceSession, *events, limit_s: float = 5.0):
    """Run the session, feed events (strings: "tap", "sleep:0.1", or typed text), then exit."""
    runner = asyncio.create_task(session.run())
    await asyncio.sleep(0)
    for ev in events:
        if ev.startswith("sleep:"):
            await asyncio.sleep(float(ev[6:]))
        elif ev == "tap":
            session._on_event(Event("tap"))
        else:
            session.post("text", ev)
            await asyncio.sleep(0)
    session.post("text", "/exit")
    await asyncio.wait_for(runner, limit_s)
    session.speech.wait(2)
    out = session.console.export_text()
    if os.environ.get("JARVIS_SHOW_VOICE"):  # see what the terminal would show
        print(out)
    return out


async def test_voice_turn_speaks_reply_and_saves(voice):
    session, app, provider, tts = voice(
        turns=["It's noon, sir. Anything else?"], heard=["What time is it?"]
    )
    out = await _drive(session, "tap")
    assert "you › What time is it?" in out and "jarvis › It's noon" in out
    assert tts.said == ["It's noon, sir.", "Anything else?"]
    msgs = app.store.messages(session.conv.id)
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert "Voice mode" in provider.calls[0]["messages"][0].content  # style note in system
    assert "heard in 12 ms" in out and "answer in 5 ms" in out
    assert session.speech.speaker.earcons[:2] == ["listen", "done"]


async def test_second_tap_right_after_recording_is_ignored(voice):
    session, app, provider, tts = voice(
        turns=["It's noon."], heard=["What time is it?"], provider=SlowProvider()
    )
    real = session.mic.record

    def record(*a, **k):  # the user taps "done" just as the pause ends it
        clip = real(*a, **k)
        session._loop.call_soon_threadsafe(session._on_event, Event("tap"))
        return clip

    session.mic.record = record
    out = await _drive(session, "tap", "sleep:1.5")
    assert "interrupted" not in out and "you › What time is it?" in out
    assert not [m for m in app.store.messages(session.conv.id) if m.interrupted]


async def test_listen_after_speaking_skips_seed_and_mutes(voice):
    session, app, provider, tts = voice(turns=["One. Two."], heard=["hi", "again"])
    await _drive(session, "tap", "sleep:0.6", "tap", "sleep:0.3")
    first, second = session.mic.calls
    assert first["seed"] is True  # nothing played yet
    assert second["seed"] is False and second["mute_s"] > 0.14  # Jarvis just spoke


async def test_reminder_is_spoken_when_idle(voice):
    session, app, provider, tts = voice()
    runner = asyncio.create_task(session.run())
    await asyncio.sleep(0.01)
    session.post("reminder", "stretch")
    await asyncio.sleep(0.2)
    session.post("text", "/exit")
    await asyncio.wait_for(runner, 5)
    session.speech.wait(2)
    assert tts.said == ["Reminder: stretch"]


async def test_nothing_heard_does_nothing(voice):
    session, app, provider, tts = voice(clips=[None])
    out = await _drive(session, "tap")
    assert "didn't hear anything" in out and not provider.calls
    assert session.speech.speaker.earcons == ["listen", "cancel"]


async def test_typed_message_is_answered_aloud_and_commands(voice):
    session, app, provider, tts = voice(turns=["Hello there, Adi."])
    out = await _drive(session, "/mute", "/unmute", "/voice nope", "hi", "sleep:0.3")
    assert "Spoken replies off" in out and "unknown voice" in out
    assert tts.said == ["Hello there, Adi."]


def _write_call(roots):
    return [
        ToolCall(name="files_write", arguments={"path": str(roots[1] / "hi.txt"), "content": "hi"})
    ]


async def test_voice_approval_yes(voice, roots):
    session, app, provider, tts = voice(
        turns=[_write_call(roots), "Done — it's on your desktop."],
        heard=["make a file hi.txt on my desktop", "yes please"],
    )
    out = await _drive(session, "tap", "sleep:0.5")
    assert (roots[1] / "hi.txt").read_text() == "hi"
    assert "→ approved" in out
    assert any(s.startswith("May I") for s in tts.said)


async def test_voice_approval_no_and_unclear(voice, roots):
    session, app, provider, tts = voice(
        turns=[_write_call(roots), "Okay, I won't."],
        heard=["make a file", "what's the weather", "no"],
    )
    out = await _drive(session, "tap", "sleep:0.5")
    assert not (roots[1] / "hi.txt").exists()
    assert "didn't catch a yes or no" in out and "→ declined" in out


async def test_typed_approval(voice, roots):
    session, app, provider, tts = voice(turns=[_write_call(roots), "Done."], heard=["make it"])
    session.mic.clips = ["clip", None]  # the spoken answer never comes…
    session.cfg.start_timeout_s = 0.1
    out = await _drive(session, "tap", "sleep:0.3", "y", "sleep:0.3")  # …so type it
    assert (roots[1] / "hi.txt").exists() and "→ approved" in out


async def test_tap_interrupts_and_listens_again(voice):
    session, app, provider, tts = voice(
        heard=["tell me a long story", "actually, stop"], provider=SlowProvider()
    )
    out = await _drive(session, "tap", "sleep:0.6", "tap", "sleep:1.5")
    assert "⏹ interrupted" in out and "you › actually, stop" in out
    msgs = app.store.messages(session.conv.id)
    interrupted = [m for m in msgs if m.interrupted]
    assert len(interrupted) == 1 and interrupted[0].content.startswith("This answer")
    assert [m.content for m in msgs if m.role == "user"] == [
        "tell me a long story",
        "actually, stop",
    ]


# ---------------------------------------------------------------- real models (optional)


def _kokoro_dir() -> Path | None:
    for d in (
        os.environ.get("JARVIS_TEST_KOKORO_DIR"),
        Path(__file__).resolve().parents[1] / "data" / "models" / "kokoro",
    ):
        if d and (Path(d) / "voices-v1.0.bin").exists() and (Path(d) / "kokoro-v1.0.onnx").exists():
            return Path(d)
    return None


@pytest.mark.skipif(_kokoro_dir() is None, reason="Kokoro files not downloaded")
def test_kokoro_speaks_and_vad_hears_it():
    from jarvis.voice.tts import KokoroTTS
    from jarvis.voice.vad import SileroVAD

    tts = KokoroTTS(_kokoro_dir())
    audio = tts.synthesize("Good evening. All systems are online.")
    assert 1.5 < audio.size / tts.sample_rate < 6
    x = resample(np.concatenate([np.zeros(12000), audio, np.zeros(24000)]), 24000, 16000)
    vad, ep = SileroVAD(), Endpointer(EndpointConfig())
    events = [ep.push(vad(f)) for f in x[: x.size // FRAME * FRAME].reshape(-1, FRAME)]
    assert "start" in events and "end" in events
    assert ep.started_at * 32 == pytest.approx(500, abs=200)  # speech began ~0.5 s in


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel: CUDA 'works' until the first transcribe."""

    created: list[tuple[str, str]] = []

    def __init__(self, name, device, compute_type, **kw):
        self.device = device
        FakeWhisperModel.created.append((device, compute_type))

    def transcribe(self, audio, **kw):
        if self.device == "cuda":
            raise RuntimeError("Could not load library cudnn_ops64_9.dll")
        self.kw = kw

        class Seg:
            text, no_speech_prob = " Open Spotify.", 0.01

        return iter([Seg()]), type("Info", (), {"language": "en"})()


def test_whisper_falls_back_to_cpu(monkeypatch):
    import faster_whisper

    from jarvis.voice.stt import WhisperSTT

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr("jarvis.voice.stt.cuda_devices", lambda: 1)
    FakeWhisperModel.created = []
    stt = WhisperSTT("distil-large-v3.5")
    result = stt.transcribe(np.zeros(16000, dtype=np.float32))
    assert FakeWhisperModel.created == [("cuda", "float16"), ("cpu", "int8")]
    assert stt.device == "cpu" and "cudnn" in stt.fallback_reason
    assert result.text == "Open Spotify." and stt.beam_size == 1
    assert stt.model.kw["hotwords"] == "Jarvis" and stt.model.kw["language"] == "en"
    assert stt.describe() == "distil-large-v3.5 on cpu (int8)"


class FakeSD:
    """Just enough of sounddevice for device selection."""

    def __init__(self, default=(-1, 1), devices=None):
        self.default = type("D", (), {"device": default, "hostapi": 0})()
        self._devices = devices or []

    def query_hostapis(self, i=None):
        apis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
        return apis if i is None else apis[i]

    def query_devices(self, i=None, kind=None):
        return self._devices if i is None else self._devices[i]


def _dev(name, ins, outs, api=0):
    return {"name": name, "max_input_channels": ins, "max_output_channels": outs,
            "hostapi": api, "default_samplerate": 48000.0}  # fmt: skip


def test_no_windows_default_mic_picks_a_real_one(monkeypatch):
    from jarvis.voice import audio

    devs = [
        _dev("Microsoft Sound Mapper - Input", 2, 0),
        _dev("Stereo Mix (Realtek Audio)", 2, 0),
        _dev("Speakers (Realtek Audio)", 0, 2),
        _dev("Microphone (USB Audio Device)", 1, 0, api=1),
        _dev("Microphone (USB Audio Device)", 1, 0, api=0),
    ]
    monkeypatch.setattr(audio, "_sd", lambda: FakeSD((-1, 2), devs))
    assert audio.find_device(None, "input") == 4  # a real mic, MME first
    assert audio.find_device(None, "output") is None  # Windows has a default speaker
    assert audio.default_device_names()[0].startswith("Microphone (USB Audio Device) (auto")

    monkeypatch.setattr(audio, "_sd", lambda: FakeSD((-1, 2), devs[:3]))
    with pytest.raises(audio.AudioError, match="no microphone set as the default"):
        audio.find_device(None, "input")
    monkeypatch.setattr(audio, "_sd", lambda: FakeSD((3, 2), devs))
    assert audio.find_device(None, "input") is None
