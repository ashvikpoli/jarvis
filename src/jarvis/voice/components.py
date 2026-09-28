"""Builds the voice pieces from settings (the voice part of the composition root)."""

from __future__ import annotations

import wave
from pathlib import Path

from jarvis.core.config import Settings


def models_dir(settings: Settings) -> Path:
    return settings.app.data_dir / "models"


def kokoro_dir(settings: Settings) -> Path:
    return models_dir(settings) / "kokoro"


def whisper_dir(settings: Settings) -> Path:
    return models_dir(settings) / "whisper"


def make_stt(settings: Settings):
    from jarvis.voice.stt import WhisperSTT

    v = settings.voice
    return WhisperSTT(
        v.stt_model,
        device=v.stt_device,
        compute_type=v.stt_compute_type,
        language=v.language or None,
        beam_size=v.stt_beam_size or None,
        hotwords=v.hotwords,
        download_root=whisper_dir(settings),
    )


def make_tts(settings: Settings):
    from jarvis.voice.tts import KokoroTTS

    v = settings.voice
    return KokoroTTS(
        kokoro_dir(settings), model_file=v.tts_model, voice=v.tts_voice, speed=v.tts_speed
    )


def make_mic(settings: Settings):
    from jarvis.voice.audio import Microphone

    return Microphone(settings.voice.input_device)


def make_speaker(settings: Settings):
    from jarvis.voice.audio import Speaker

    return Speaker(settings.voice.output_device)


def whisper_downloaded(settings: Settings) -> bool:
    try:
        from faster_whisper.utils import download_model

        download_model(
            settings.voice.stt_model,
            local_files_only=True,
            cache_dir=str(whisper_dir(settings)),
        )
        return True
    except Exception:
        return False


def download_whisper(settings: Settings) -> str:
    from faster_whisper.utils import download_model

    return download_model(settings.voice.stt_model, cache_dir=str(whisper_dir(settings)))


def save_wav(path: Path, audio, rate: int) -> None:
    import numpy as np

    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def word_accuracy(expected: str, heard: str) -> float:
    """Share of expected words that were heard, in order (simple, for self-tests)."""
    import difflib
    import re

    a = re.findall(r"[a-z0-9']+", expected.lower())
    b = re.findall(r"[a-z0-9']+", heard.lower())
    if not a:
        return 1.0
    matched = sum(m.size for m in difflib.SequenceMatcher(None, a, b).get_matching_blocks())
    return matched / len(a)
