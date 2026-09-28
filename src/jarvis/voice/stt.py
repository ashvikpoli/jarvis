"""Speech-to-text with faster-whisper (CTranslate2), on the GPU when it can be."""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Whisper was trained on subtitles, so on noise it "hears" these.
ALWAYS_JUNK = {
    "thanks for watching", "thank you for watching", "please subscribe",
    "subtitles by the amara org community", "",
}  # fmt: skip
# Real words people do say — dropped only when Whisper itself doubts there was speech.
MAYBE_JUNK = {"you", "thank you", "thanks", "bye", "so", "oh", "uh", "um", "hmm", "i"}


class STTError(RuntimeError):
    pass


def cuda_library_dirs() -> list[Path]:
    """Folders holding the pip-installed NVIDIA DLLs (cuBLAS, cuDNN) on Windows."""
    dirs: list[Path] = []
    for base in map(Path, sys.path):
        nvidia = base / "nvidia"
        if not nvidia.is_dir():
            continue
        for pkg in sorted(nvidia.iterdir()):
            for sub in ("bin", "lib"):
                d = pkg / sub
                if d.is_dir() and any(d.glob("*.dll")):
                    dirs.append(d)
    return dirs


_dll_setup_done = False


def setup_cuda_dlls() -> list[Path]:
    """Let CTranslate2 find cuBLAS/cuDNN from the nvidia-* pip packages (Windows)."""
    global _dll_setup_done
    if os.name != "nt":
        return []
    dirs = cuda_library_dirs()
    if not _dll_setup_done:
        for d in dirs:
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass
        # CTranslate2 loads cuBLAS by name at first use, which searches PATH.
        os.environ["PATH"] = os.pathsep.join([*map(str, dirs), os.environ.get("PATH", "")])
        _dll_setup_done = True
    return dirs


def cuda_devices() -> int:
    try:
        setup_cuda_dlls()
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


@dataclass
class Transcript:
    text: str
    ms: float  # how long transcription took
    audio_s: float
    language: str | None = None
    no_speech_prob: float | None = None


def clean_transcript(text: str, no_speech_prob: float | None = None) -> str:
    text = " ".join(text.split()).strip()
    bare = re.sub(r"[^\w' ]+", "", text.lower()).strip()
    if bare in ALWAYS_JUNK:
        return ""
    if bare in MAYBE_JUNK and (no_speech_prob or 0) > 0.4:
        return ""
    return text


class WhisperSTT:
    """Loads lazily; `load()` picks cuda → cpu and warms up so the first real use is fast."""

    def __init__(
        self,
        model: str = "distil-large-v3.5",
        *,
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = "en",
        beam_size: int | None = None,
        hotwords: str | None = "Jarvis",
        download_root: Path | None = None,
        cpu_threads: int = 0,
    ):
        self.model_name = model
        self.device_pref = device
        self.compute_pref = compute_type
        self.language = language
        self.beam_size_pref = beam_size
        self.hotwords = hotwords or None
        self.download_root = download_root
        self.cpu_threads = cpu_threads
        self.model = None
        self.device: str | None = None
        self.compute_type: str | None = None
        self.fallback_reason: str | None = None
        self._lock = threading.Lock()

    @property
    def beam_size(self) -> int:
        if self.beam_size_pref:
            return self.beam_size_pref
        return 5 if self.device == "cuda" else 1  # greedy keeps CPU latency down

    def _candidates(self) -> list[tuple[str, str]]:
        dev, ct = self.device_pref, self.compute_pref
        gpu = (dev == "cuda") or (dev == "auto" and cuda_devices() > 0)
        out: list[tuple[str, str]] = []
        if gpu:
            # int8 is disabled on RTX 50 (Blackwell) in CTranslate2, so float16.
            out.append(("cuda", "float16" if ct == "auto" else ct))
        if dev != "cuda":
            out.append(("cpu", "int8" if ct == "auto" else ct))
        return out

    def load(self) -> None:
        with self._lock:
            if self.model is not None:
                return
            import numpy as np

            setup_cuda_dlls()
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise STTError(f"faster-whisper isn't installed: {e}") from e
            errors = []
            for device, compute in self._candidates():
                try:
                    model = WhisperModel(
                        self.model_name,
                        device=device,
                        compute_type=compute,
                        download_root=str(self.download_root) if self.download_root else None,
                        cpu_threads=self.cpu_threads,
                    )
                    # Warm-up: also the moment missing CUDA DLLs (cuDNN) show up.
                    segments, _ = model.transcribe(
                        np.zeros(16000, dtype=np.float32), language=self.language, beam_size=1
                    )
                    list(segments)
                except Exception as e:
                    log.warning("whisper on %s/%s failed: %s", device, compute, e)
                    errors.append(f"{device}: {e}")
                    continue
                self.model, self.device, self.compute_type = model, device, compute
                if errors:
                    self.fallback_reason = errors[0]
                return
            raise STTError("couldn't load the speech model — " + "; ".join(errors))

    def transcribe(self, audio) -> Transcript:
        """16 kHz mono float32 → text ("" if it was noise)."""
        self.load()
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # we already cut the audio to speech
            condition_on_previous_text=False,
            without_timestamps=True,
            hotwords=self.hotwords,
        )
        segs = list(segments)
        nsp = max((s.no_speech_prob for s in segs), default=None)
        text = clean_transcript("".join(s.text for s in segs), nsp)
        return Transcript(
            text=text,
            ms=(time.perf_counter() - t0) * 1000,
            audio_s=len(audio) / 16000,
            language=getattr(info, "language", None),
            no_speech_prob=nsp,
        )

    def describe(self) -> str:
        if self.model is None:
            return f"{self.model_name} (not loaded)"
        return f"{self.model_name} on {self.device} ({self.compute_type})"
