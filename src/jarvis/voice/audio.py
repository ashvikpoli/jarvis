"""Microphone and speakers (sounddevice / PortAudio), plus small audio helpers."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from jarvis.voice.vad import FRAME, SAMPLE_RATE, EndpointConfig, Endpointer, utterance_slice

log = logging.getLogger(__name__)

TTS_RATE = 24_000


class AudioError(RuntimeError):
    pass


def resample(audio, src_rate: int, dst_rate: int):
    """Linear resampling — plenty for speech going into Whisper or out of a speaker."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or audio.size == 0:
        return audio
    n = max(1, round(audio.size * dst_rate / src_rate))
    x = np.linspace(0, audio.size - 1, n)
    return np.interp(x, np.arange(audio.size), audio).astype(np.float32)


def tone(freqs: list[float], ms: int = 90, rate: int = TTS_RATE, volume: float = 0.18):
    """A soft beep (or a little rising/falling pair) used as listen / done cues."""
    import numpy as np

    parts = []
    for f in freqs:
        n = int(rate * ms / 1000)
        t = np.arange(n) / rate
        env = np.minimum(1, np.minimum(t, t[::-1]) * 1000 / 12)  # 12 ms fade in/out
        parts.append((volume * env * np.sin(2 * np.pi * f * t)).astype(np.float32))
    return np.concatenate(parts)


EARCONS: dict[str, tuple[list[float], int]] = {
    "listen": ([660, 880], 70),  # rising: I'm listening
    "done": ([880], 45),  # got it
    "cancel": ([520, 390], 70),  # falling: heard nothing
}


def rms(audio) -> float:
    import numpy as np

    a = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def _sd():
    try:
        import sounddevice
    except (ImportError, OSError) as e:  # OSError: PortAudio library missing
        raise AudioError(f"audio isn't available: {e}") from e
    return sounddevice


def _device(spec: str | int | None) -> int | str | None:
    if spec in (None, "", "default"):
        return None
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return int(spec)
    return spec  # a (part of a) name — resolved by find_device() when the stream opens


def find_device(spec: int | str | None, kind: str) -> int | None:
    """Device index for a number or part of a name.

    Windows lists every device once per audio API (MME, DirectSound, WASAPI…), so a
    bare name is ambiguous for PortAudio; prefer the default API's entry.
    """
    if spec is None or isinstance(spec, int):
        return spec
    sd = _sd()
    want = spec.lower()
    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    default_api = sd.default.hostapi
    matches = [
        (i, d)
        for i, d in enumerate(sd.query_devices())
        if want in d["name"].lower() and d[channels] > 0
    ]
    if not matches:
        raise AudioError(f"no {kind} device matching {spec!r} — see: jarvis voice devices")
    matches.sort(key=lambda m: m[1]["hostapi"] != default_api)  # default API first
    return matches[0][0]


def list_devices() -> list[dict[str, Any]]:
    sd = _sd()
    out = []
    default_in, default_out = sd.default.device
    for i, d in enumerate(sd.query_devices()):
        hostapi = sd.query_hostapis(d["hostapi"])["name"]
        out.append(
            {
                "index": i,
                "name": d["name"],
                "hostapi": hostapi,
                "inputs": d["max_input_channels"],
                "outputs": d["max_output_channels"],
                "rate": d["default_samplerate"],
                "default_input": i == default_in,
                "default_output": i == default_out,
            }
        )
    return out


def default_device_names() -> tuple[str | None, str | None]:
    sd = _sd()
    names = []
    for kind in ("input", "output"):
        try:
            names.append(sd.query_devices(kind=kind)["name"])
        except Exception:
            names.append(None)
    return names[0], names[1]


# ---------------------------------------------------------------- microphone


class Microphone:
    """An always-open 16 kHz mono input stream.

    Keeping it open means no start-up delay and ~0.3 s of audio from *before* speech
    was detected, so the first word never gets clipped. Frames are only kept while
    `record()` is running (plus the short pre-roll ring).
    """

    def __init__(self, device: str | int | None = None, vad_factory: Callable | None = None):
        self.device = _device(device)
        self._vad_factory = vad_factory
        self._stream = None
        self._rate = SAMPLE_RATE
        self._q: queue.Queue = queue.Queue(maxsize=2000)
        self._recording = threading.Event()
        self._ring: deque = deque(maxlen=16)  # ~0.5 s of frames while idle
        self._pending = None  # leftover samples when the device rate isn't 16 kHz
        self._closed = False
        self.level = 0.0  # last frame RMS, for meters

    # -- stream

    def start(self) -> None:
        if self._stream is not None:
            return
        if self._closed:
            raise AudioError("the microphone was closed")
        sd = _sd()
        import numpy as np

        self._np = np
        self.device = find_device(self.device, "input")
        try:
            self._stream = self._open(sd, SAMPLE_RATE, FRAME)
        except Exception as e:  # some devices only do their native rate
            log.info("mic can't open at 16 kHz (%s); using its native rate", e)
            info = sd.query_devices(self.device, kind="input")
            self._rate = int(info["default_samplerate"])
            block = round(FRAME * self._rate / SAMPLE_RATE)
            try:
                self._stream = self._open(sd, self._rate, block)
            except Exception as e2:
                raise AudioError(f"couldn't open the microphone: {e2}") from e2
        self._stream.start()

    def _open(self, sd, rate: int, block: int):
        return sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            blocksize=block,
            device=self.device,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status) -> None:  # PortAudio thread
        data = indata[:, 0].copy()
        if self._rate != SAMPLE_RATE:
            data = resample(data, self._rate, SAMPLE_RATE)
            if self._pending is not None:
                data = self._np.concatenate([self._pending, data])
            n = data.size // FRAME * FRAME
            self._pending = data[n:]
            frames_out = [data[i : i + FRAME] for i in range(0, n, FRAME)]
        else:
            frames_out = [data]
        for f in frames_out:
            if f.size != FRAME:
                continue
            if self._recording.is_set():
                try:
                    self._q.put_nowait(f)
                except queue.Full:
                    pass
            else:
                self._ring.append(f)

    def close(self) -> None:
        self._closed = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    # -- recording

    def record(
        self,
        stop: threading.Event,
        cfg: EndpointConfig | None = None,
        *,
        on_start: Callable[[], None] | None = None,
        seed: bool = True,
        mute_s: float = 0.0,
    ):
        """Block until you finish speaking. Returns 16 kHz float32 audio, or None if
        nothing was said (timeout, too short, or `stop` set before speech began).

        `seed`: start with the ~0.25 s heard just before (so a word said right on the
        tap isn't lost) — pass False when Jarvis was just talking, or it hears itself.
        `mute_s`: ignore speech for this long at the start (our own "listening" beep
        and whatever is still in the speaker's buffer).
        """
        self.start()
        cfg = cfg or EndpointConfig()
        vad = self._vad_factory() if self._vad_factory else _default_vad()
        vad.reset()
        ep = Endpointer(cfg)
        frames: list = []
        while True:  # drop anything stale
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        # Seed with the idle ring (≤0.25 s) so a quick "open…" right after the tap counts.
        seeded: list = []
        for _ in range(3 if seed else 0):  # the audio thread may append meanwhile
            try:
                seeded = list(self._ring)[-8:]
                break
            except RuntimeError:
                continue
        muted_frames = len(seeded) + round(mute_s * 1000 / 32)
        self._recording.set()
        try:
            source = iter(seeded)
            while not stop.is_set():
                frame = next(source, None)
                if frame is None:
                    try:
                        frame = self._q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                frames.append(frame)
                self.level = rms(frame)
                prob = vad(frame)
                if len(frames) <= muted_frames and len(frames) > len(seeded):
                    prob = 0.0  # our own beep / speaker tail
                event = ep.push(prob)
                if event == "start" and on_start:
                    on_start()
                elif event in ("end", "timeout"):
                    break
        finally:
            self._recording.clear()
            self._ring.clear()  # never reuse audio from before this recording
        return collect(frames, ep)


def collect(frames: list, ep: Endpointer):
    """The utterance audio out of all recorded frames, or None if nothing was said."""
    import numpy as np

    sl = utterance_slice(ep)
    if sl is None:
        return None
    return np.concatenate(frames[sl]).astype(np.float32)


_VAD = None


def _default_vad():
    from jarvis.voice.vad import SileroVAD

    global _VAD
    if _VAD is None:
        _VAD = SileroVAD()
    return _VAD


# ---------------------------------------------------------------- speaker


class Speaker:
    """An always-open output stream playing queued clips; `stop()` is instant."""

    def __init__(self, device: str | int | None = None, rate: int = TTS_RATE):
        self.device = _device(device)
        self.rate = rate
        self._stream = None
        self._lock = threading.Lock()
        self._clips: deque = deque()
        self._pos = 0
        self._idle = threading.Event()
        self._idle.set()
        self.first_audio_at: float | None = None  # perf_counter of the first sample played
        self.last_audio_at = 0.0  # perf_counter when sound last went out
        self.epoch = 0  # bumped by stop(); clips queued for an older epoch are dropped
        self._mark_next = False

    def start(self) -> None:
        if self._stream is not None:
            return
        sd = _sd()
        import numpy as np

        self._np = np
        self.device = find_device(self.device, "output")
        try:
            self._stream = self._open(sd, self.rate)
        except Exception as e:  # e.g. WASAPI only accepts the device's own mix rate
            info = sd.query_devices(self.device, kind="output")
            native = int(info["default_samplerate"])
            if native == self.rate:
                raise AudioError(f"couldn't open the speakers: {e}") from e
            log.info("speaker can't do %s Hz (%s); using %s Hz", self.rate, e, native)
            try:
                self._stream = self._open(sd, native)
                self.rate = native  # play() resamples to this
            except Exception as e2:
                self._stream = None
                raise AudioError(f"couldn't open the speakers: {e2}") from e2

    def _open(self, sd, rate: int):
        stream = sd.OutputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
            blocksize=0,
            latency="low",
        )
        stream.start()
        return stream

    @property
    def latency(self) -> float:
        """Seconds between handing audio to PortAudio and it leaving the speaker."""
        try:
            return float(self._stream.latency) if self._stream is not None else 0.1
        except Exception:
            return 0.1

    def recently_played(self, within_s: float = 0.6) -> bool:
        return self.busy or time.perf_counter() - self.last_audio_at < within_s + self.latency

    def _callback(self, outdata, frames, time_info, status) -> None:  # PortAudio thread
        out = outdata[:, 0]
        out.fill(0)
        filled = 0
        with self._lock:
            while filled < frames and self._clips:
                clip = self._clips[0]
                n = min(frames - filled, clip.size - self._pos)
                out[filled : filled + n] = clip[self._pos : self._pos + n]
                if self._mark_next and n > 0:
                    self.first_audio_at = time.perf_counter()
                    self._mark_next = False
                filled += n
                self._pos += n
                self.last_audio_at = time.perf_counter()
                if self._pos >= clip.size:
                    self._clips.popleft()
                    self._pos = 0
            if not self._clips:
                self._idle.set()

    def play(self, audio, rate: int | None = None, *, epoch: int | None = None) -> None:
        """Queue a clip (float32 mono). Returns immediately.

        With `epoch`, the clip is dropped if stop() was called since that epoch —
        so a sentence synthesized just before a barge-in never plays.
        """
        self.start()
        clip = resample(audio, rate or TTS_RATE, self.rate)
        if clip.size == 0:
            return
        with self._lock:
            if epoch is not None and epoch != self.epoch:
                return
            self._clips.append(clip)
            self._idle.clear()

    def mark_first_audio(self) -> None:
        """Record when the next queued sample actually reaches the speakers."""
        with self._lock:
            self.first_audio_at = None
            self._mark_next = True

    def stop(self) -> None:
        with self._lock:
            self.epoch += 1
            self._clips.clear()
            self._pos = 0
            self._mark_next = False
            self._idle.set()

    @property
    def busy(self) -> bool:
        return not self._idle.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    def earcon(self, name: str) -> float:
        """Play a cue; returns its length in seconds."""
        freqs, ms = EARCONS[name]
        clip = tone(freqs, ms, TTS_RATE)
        self.play(clip, TTS_RATE)
        return clip.size / TTS_RATE

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
