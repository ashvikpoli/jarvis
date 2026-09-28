"""Voice activity detection: is someone speaking right now, and have they finished?

`SileroVAD` scores 32 ms frames (the Silero model faster-whisper already ships, run
with onnxruntime — no torch). `Endpointer` turns those scores into "started talking"
and "stopped talking" — pure logic, so it's unit-tested with made-up scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16_000
FRAME = 512  # samples per VAD frame = 32 ms at 16 kHz
FRAME_MS = FRAME * 1000 / SAMPLE_RATE


def silero_model_path() -> Path:
    from faster_whisper.utils import get_assets_path

    return Path(get_assets_path()) / "silero_vad_v6.onnx"


class SileroVAD:
    """Streaming Silero VAD: call with consecutive 512-sample frames → speech probability."""

    CONTEXT = 64

    def __init__(self, path: Path | None = None):
        import numpy as np
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._np = np
        self.session = onnxruntime.InferenceSession(
            str(path or silero_model_path()), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.reset()

    def reset(self) -> None:
        np = self._np
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.CONTEXT, dtype=np.float32)

    def __call__(self, frame) -> float:
        np = self._np
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.shape[0] != FRAME:
            raise ValueError(f"VAD frames must be {FRAME} samples, got {frame.shape[0]}")
        x = np.concatenate([self._context, frame])[None, :]
        out, self._h, self._c = self.session.run(None, {"input": x, "h": self._h, "c": self._c})
        self._context = frame[-self.CONTEXT :].copy()
        return float(np.asarray(out).reshape(-1)[0])


@dataclass
class EndpointConfig:
    threshold: float = 0.5  # probability above which a frame is speech
    silence_ms: int = 700  # this much quiet after speech = you're done
    start_timeout_s: float = 6.0  # give up if you don't start talking within this
    max_s: float = 30.0  # hard cap on one utterance
    min_speech_ms: int = 200  # shorter blips (a cough, a click) are ignored
    start_frames: int = 2  # consecutive speech frames needed to count as "started"
    preroll_ms: int = 300  # audio kept from just before speech started
    tail_ms: int = 200  # audio kept after speech ended


class Endpointer:
    """Feed one score per frame; `push()` returns "start", "end", "timeout" or None."""

    def __init__(self, cfg: EndpointConfig | None = None):
        self.cfg = cfg or EndpointConfig()
        c = self.cfg
        self.neg_threshold = max(c.threshold - 0.15, 0.05)
        self._silence_frames = max(1, round(c.silence_ms / FRAME_MS))
        self._timeout_frames = round(c.start_timeout_s * 1000 / FRAME_MS)
        self._max_frames = round(c.max_s * 1000 / FRAME_MS)
        self.frames = 0
        self.started_at: int | None = None  # frame index where speech began
        self.last_speech: int | None = None
        self.speech_frames = 0
        self._run = 0
        self._quiet = 0
        self.finished: str | None = None

    @property
    def started(self) -> bool:
        return self.started_at is not None

    def push(self, prob: float) -> str | None:
        if self.finished:
            return self.finished
        i = self.frames
        self.frames += 1
        c = self.cfg
        if not self.started:
            self._run = self._run + 1 if prob >= c.threshold else 0
            if self._run >= c.start_frames:
                self.started_at = i - c.start_frames + 1
                self.last_speech = i
                self.speech_frames = self._run
                return "start"
            if self.frames >= self._timeout_frames:
                self.finished = "timeout"
                return "timeout"
            return None
        # speaking: hysteresis — only a clearly quiet frame counts as silence
        if prob >= self.neg_threshold:
            self._quiet = 0
            if prob >= c.threshold:
                self.speech_frames += 1
                self.last_speech = i
        else:
            self._quiet += 1
        if self._quiet >= self._silence_frames or self.frames - self.started_at >= self._max_frames:
            self.finished = "end"
            return "end"
        return None

    @property
    def enough_speech(self) -> bool:
        return self.speech_frames * FRAME_MS >= self.cfg.min_speech_ms


def utterance_slice(ep: Endpointer) -> slice | None:
    """Which frames (of all frames pushed) make up the utterance: pre-roll + speech + tail.

    None when nothing usable was said (never started, or too short to be speech).
    """
    if ep.started_at is None or ep.last_speech is None or not ep.enough_speech:
        return None
    pre = round(ep.cfg.preroll_ms / FRAME_MS)
    tail = round(ep.cfg.tail_ms / FRAME_MS)
    return slice(max(0, ep.started_at - pre), min(ep.frames, ep.last_speech + 1 + tail))
