"""Text-to-speech with Kokoro (82M params, ONNX, runs on the CPU in real time).

A small Kokoro runner of our own (text → espeak-ng phonemes → token ids → ONNX
model with a voice style vector → 24 kHz audio). We don't use the kokoro-onnx
package because it doesn't install on Python 3.14; this is the same pipeline in
~150 lines. Model files come from the kokoro-onnx releases (Apache-2.0 weights).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
MAX_PHONEMES = 510
RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MODEL_FILES = {
    # name: approximate size in bytes (to sanity-check downloads)
    "kokoro-v1.0.onnx": 325_532_387,  # full precision — best quality (default)
    "kokoro-v1.0.fp16.onnx": 177_464_787,  # half the size, near-identical sound
    "kokoro-v1.0.int8.onnx": 92_361_271,  # smallest; slower on most CPUs
    "voices-v1.0.bin": 28_214_398,
}
VOICES_FILE = "voices-v1.0.bin"

# The English voices. First letter: a = American, b = British; second: f/m.
VOICES = {
    "bm_george": "British male — deep, calm (default)",
    "bm_lewis": "British male — warm, lower",
    "bm_daniel": "British male — crisp",
    "bm_fable": "British male — storyteller",
    "bf_emma": "British female",
    "bf_isabella": "British female — bright",
    "bf_alice": "British female — soft",
    "bf_lily": "British female — young",
    "am_michael": "American male",
    "am_adam": "American male — deep",
    "am_onyx": "American male — low",
    "am_fenrir": "American male — gravelly",
    "am_echo": "American male — light",
    "am_eric": "American male — steady",
    "am_liam": "American male — young",
    "am_puck": "American male — playful",
    "af_heart": "American female — the Kokoro favourite",
    "af_bella": "American female — expressive",
    "af_nicole": "American female — whispery",
    "af_sarah": "American female",
    "af_sky": "American female — light",
}


class TTSError(RuntimeError):
    pass


def voice_language(voice: str) -> str:
    return "en-gb" if voice.startswith("b") else "en-us"


def missing_files(model_dir: Path, model_file: str) -> list[str]:
    return [f for f in (model_file, VOICES_FILE) if not (model_dir / f).is_file()]


def download_models(model_dir: Path, model_file: str, progress=None) -> list[Path]:
    """Fetch any missing Kokoro files. `progress(name, done_bytes, total_bytes)`."""
    import httpx

    model_dir.mkdir(parents=True, exist_ok=True)
    got = []
    for name in missing_files(model_dir, model_file):
        url = f"{RELEASE}/{name}"
        tmp = model_dir / (name + ".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
            if r.status_code != 200:
                raise TTSError(f"download failed ({r.status_code}): {url}")
            total = int(r.headers.get("content-length") or MODEL_FILES.get(name, 0))
            done = 0
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(name, done, total)
        if total and tmp.stat().st_size != total:
            raise TTSError(f"{name}: incomplete download ({tmp.stat().st_size} of {total} bytes)")
        os.replace(tmp, model_dir / name)
        got.append(model_dir / name)
    return got


def _load_vocab() -> dict[str, int]:
    return json.loads((Path(__file__).parent / "kokoro_vocab.json").read_text(encoding="utf-8"))


def trim_silence(audio, threshold: float = 0.01, margin: int = 1200):
    """Drop leading/trailing near-silence (Kokoro pads each clip)."""
    import numpy as np

    a = np.asarray(audio, dtype=np.float32)
    loud = np.flatnonzero(np.abs(a) > threshold)
    if loud.size == 0:
        return a[:0]
    return a[max(0, loud[0] - margin) : min(a.size, loud[-1] + 1 + margin)]


def split_phonemes(ph: str, limit: int = MAX_PHONEMES) -> list[str]:
    """Cut an over-long phoneme string at sentence marks, commas, then spaces."""
    ph = ph.strip()
    if len(ph) <= limit:
        return [ph] if ph else []
    for marks in (".!?", ",;:", " "):
        cut = max(ph.rfind(m, 0, limit) for m in marks)
        if cut > 0:
            return [ph[: cut + 1].strip(), *split_phonemes(ph[cut + 1 :], limit)]
    return [ph[:limit], *split_phonemes(ph[limit:], limit)]


_espeak_lock = threading.Lock()  # espeak-ng has process-global state


class KokoroTTS:
    def __init__(
        self,
        model_dir: Path,
        *,
        model_file: str = "kokoro-v1.0.onnx",
        voice: str = "bm_george",
        speed: float = 1.0,
        threads: int = 0,
    ):
        self.model_dir = Path(model_dir)
        self.model_file = model_file
        self.voice = voice
        self.speed = speed
        self.threads = threads
        self.sample_rate = SAMPLE_RATE
        self._session = None
        self._voices = None
        self._backends: dict[str, object] = {}
        self._vocab = _load_vocab()
        self._lock = threading.Lock()

    # -- loading

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            missing = missing_files(self.model_dir, self.model_file)
            if missing:
                raise TTSError(
                    f"voice model files missing ({', '.join(missing)}) — run: jarvis voice setup"
                )
            import numpy as np
            import onnxruntime

            opts = onnxruntime.SessionOptions()
            opts.log_severity_level = 3
            if self.threads:
                opts.intra_op_num_threads = self.threads
            self._session = onnxruntime.InferenceSession(
                str(self.model_dir / self.model_file),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._voices = np.load(self.model_dir / VOICES_FILE)
            self._inputs = {i.name: i.type for i in self._session.get_inputs()}
            self._tokens_input = "input_ids" if "input_ids" in self._inputs else "tokens"
        self.set_voice(self.voice)

    def voices(self) -> list[str]:
        self.load()
        return sorted(self._voices.files)

    def set_voice(self, voice: str) -> None:
        if self._voices is not None and voice not in self._voices.files:
            raise TTSError(f"unknown voice {voice!r}. Try: {', '.join(VOICES)}")
        self.voice = voice

    def _backend(self, lang: str):
        if lang not in self._backends:
            import espeakng_loader
            from phonemizer.backend import EspeakBackend
            from phonemizer.backend.espeak.wrapper import EspeakWrapper

            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            self._backends[lang] = EspeakBackend(
                lang, preserve_punctuation=True, with_stress=True, language_switch="remove-flags"
            )
        return self._backends[lang]

    # -- synthesis

    def phonemize(self, text: str, lang: str | None = None) -> str:
        lang = lang or voice_language(self.voice)
        with _espeak_lock:
            ph = self._backend(lang).phonemize([text], strip=True)[0]
        return "".join(c for c in ph if c in self._vocab).strip()

    def synthesize(self, text: str, voice: str | None = None, speed: float | None = None):
        """Text → 24 kHz float32 mono audio (empty if there's nothing to say)."""
        import numpy as np

        self.load()
        voice = voice or self.voice
        if voice not in self._voices.files:
            raise TTSError(f"unknown voice {voice!r}")
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        style_table = self._voices[voice]  # (510, 1, 256): one style per phoneme count
        parts = []
        for ph in split_phonemes(self.phonemize(text, voice_language(voice))):
            ids = [self._vocab[c] for c in ph if c in self._vocab]
            if not ids:
                continue
            style = style_table[min(len(ids), len(style_table)) - 1]
            dtype = np.int32 if "int" in self._inputs.get("speed", "") else np.float32
            feeds = {
                self._tokens_input: np.array([[0, *ids, 0]], dtype=np.int64),
                "style": np.asarray(style, dtype=np.float32),
                "speed": np.array([speed or self.speed], dtype=dtype),
            }
            audio = np.asarray(self._session.run(None, feeds)[0], dtype=np.float32).reshape(-1)
            parts.append(trim_silence(audio))
            parts.append(np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts[:-1])


# ---------------------------------------------------------------- speaking a reply


@dataclass
class SpeechStats:
    first_audio_at: float | None = None  # perf_counter when the first word played
    synth_ms: float = 0.0
    sentences: int = 0


class SpeechQueue:
    """Speak sentences in order while later ones are still being written.

    `say()` is non-blocking; a worker thread synthesizes and hands audio to the
    speaker, so sentence 2 is generated while sentence 1 plays. `stop()` drops
    everything instantly (barge-in); `wait()` blocks until all speech has played.
    """

    def __init__(self, tts, speaker):
        import queue

        self.tts = tts
        self.speaker = speaker
        self._q: queue.Queue = queue.Queue()
        self._pending = 0
        self._cv = threading.Condition()
        self.stats = SpeechStats()
        self.errors: list[str] = []
        self._thread = threading.Thread(target=self._work, name="jarvis-tts", daemon=True)
        self._thread.start()

    def reset_stats(self) -> None:
        self.stats = SpeechStats()
        self.speaker.mark_first_audio()

    def say(self, text: str) -> None:
        from jarvis.voice.text import speech_text

        spoken = speech_text(text)
        if not spoken:
            return
        with self._cv:
            self._pending += 1
        # Tagged with the speaker's epoch: after stop() (barge-in) it's stale and dropped.
        self._q.put((self.speaker.epoch, spoken))

    def _work(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            epoch, text = item
            try:
                if epoch == self.speaker.epoch:
                    t0 = time.perf_counter()
                    audio = self.tts.synthesize(text)
                    self.stats.synth_ms += (time.perf_counter() - t0) * 1000
                    if epoch == self.speaker.epoch and len(audio):
                        # play() re-checks the epoch under the speaker's lock
                        self.speaker.play(audio, self.tts.sample_rate, epoch=epoch)
                        self.stats.sentences += 1
            except Exception as e:  # never kill the worker; show the problem instead
                log.exception("speech failed")
                self.errors.append(str(e))
            finally:
                with self._cv:
                    self._pending -= 1
                    self._cv.notify_all()

    @property
    def first_audio_at(self) -> float | None:
        return self.speaker.first_audio_at

    @property
    def busy(self) -> bool:
        return self._pending > 0 or self.speaker.busy

    def wait(self, timeout: float | None = None) -> bool:
        """Until everything queued has been synthesized and played."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while self._pending > 0:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return False
                self._cv.wait(left)
        left = None if deadline is None else max(0, deadline - time.monotonic())
        return self.speaker.wait(left)

    def stop(self) -> None:
        self.speaker.stop()

    def close(self) -> None:
        self.stop()
        self._q.put(None)
