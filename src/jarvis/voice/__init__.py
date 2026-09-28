"""Voice: tap a key, speak, hear Jarvis answer (Phase 4).

    hotkey tap → Microphone + VAD (auto-stop) → WhisperSTT → conversation → sentences
    → KokoroTTS → Speaker

Heavy libraries (numpy, onnxruntime, ctranslate2, sounddevice, pynput) are imported
lazily inside the classes that need them, so the rest of Jarvis starts fast and the
pure-logic parts are testable anywhere.
"""
