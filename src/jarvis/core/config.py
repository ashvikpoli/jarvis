"""Settings: code defaults <- config/default.toml <- config/local.toml <- JARVIS_* env vars."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class AppSettings(BaseModel):
    name: str = "Jarvis"
    data_dir: Path = Path("data")
    system_prompt: str = "You are Jarvis, a concise, capable personal assistant."


class LoggingSettings(BaseModel):
    level: str = "INFO"
    console_level: str = "WARNING"
    json_file: bool = True


class ModelsSettings(BaseModel):
    roles: dict[str, str] = Field(
        default_factory=lambda: {"chat": "ollama:qwen3:8b", "embed": "ollama:qwen3-embedding:0.6b"}
    )
    # Per-role generation options, e.g. {"chat": {"think": False, "temperature": 0.7}}
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ConversationSettings(BaseModel):
    context_tokens: int = 8192  # prompt budget; also sent to Ollama as num_ctx
    reserve_tokens: int = 1024  # room kept free for the reply
    include_datetime: bool = True


# Jarvis's own data and opening things are low-risk, so they don't ask by default.
DEFAULT_PERMISSION_OVERRIDES = {
    "notes_add": "allow",
    "reminders_set": "allow",
    "reminders_update": "allow",
    "reminders_cancel": "allow",
    "apps_open": "allow",
    "web_open": "allow",
    "web_search": "allow",
    "files_open": "allow",
    "memory_save": "allow",
    "memory_update": "allow",
    "memory_forget": "allow",
}


class ToolPermissions(BaseModel):
    safe: str = "allow"
    change: str = "ask"
    destructive: str = "ask"
    # tool name -> allow|ask|deny. Your entries are merged over the defaults.
    overrides: dict[str, str] = Field(default_factory=dict)

    @property
    def effective_overrides(self) -> dict[str, str]:
        return {**DEFAULT_PERMISSION_OVERRIDES, **self.overrides}


# Never read or written: secrets that would let anyone impersonate the user.
DEFAULT_NO_ACCESS = [
    "~/.ssh",
    "~/.aws",
    "~/.azure",
    "~/.gnupg",
    "~/.kube",
    "~/.docker",
    "%APPDATA%/Microsoft/Credentials",
    "%LOCALAPPDATA%/Microsoft/Credentials",
    "%APPDATA%/Microsoft/Protect",
    "%APPDATA%/Microsoft/Crypto",
    "%LOCALAPPDATA%/Microsoft/Vault",
    "%LOCALAPPDATA%/Google/Chrome/User Data",
    "%LOCALAPPDATA%/Microsoft/Edge/User Data",
    "%LOCALAPPDATA%/BraveSoftware",
    "%APPDATA%/Mozilla/Firefox/Profiles",
    "%APPDATA%/Opera Software",
    "%LOCALAPPDATA%/Vivaldi",
    "%APPDATA%/discord/Local Storage",
    "%APPDATA%/Bitwarden",
    "%LOCALAPPDATA%/1Password",
    "%APPDATA%/KeePass",
    "%APPDATA%/KeePassXC",
]
# Never changed, even with approval (reading is fine).
DEFAULT_NO_WRITE = [
    "%SystemRoot%",
    "%ProgramFiles%",
    "%ProgramFiles(x86)%",
    "%ProgramData%",
    "C:/$Recycle.Bin",
    "C:/System Volume Information",
    "C:/Recovery",
    "C:/Boot",
]


class FileToolSettings(BaseModel):
    # Your folders: read freely, changes ask. Known names (Documents, Desktop, Downloads,
    # Pictures, Music, Videos — OneDrive-aware) or "Jarvis" (this project) or paths.
    roots: list[str] = Field(
        default_factory=lambda: ["Documents", "Desktop", "Downloads", "Jarvis"]
    )
    read_anywhere: bool = True  # read-only access to the rest of the PC
    outside_writes: str = "ask"  # changes outside your folders: ask (every time) | deny
    no_access: list[str] = Field(default_factory=lambda: list(DEFAULT_NO_ACCESS))
    no_write: list[str] = Field(default_factory=lambda: list(DEFAULT_NO_WRITE))
    max_read_chars: int = 20_000


class ToolsSettings(BaseModel):
    enabled: bool = True
    max_steps: int = 15
    disabled: list[str] = Field(default_factory=list)  # tool names to leave out entirely
    permissions: ToolPermissions = Field(default_factory=ToolPermissions)
    files: FileToolSettings = Field(default_factory=FileToolSettings)
    notifications: bool = True


class MemorySettings(BaseModel):
    enabled: bool = True
    auto_learn: bool = True  # learn facts from conversation in the background
    top_k: int = 5  # memories recalled per message
    min_score: float = 0.5  # similarity needed to recall (0..1)
    core_limit: int = 15  # "profile" memories always included
    dedupe_threshold: float = 0.9  # saving something this similar updates instead
    query_prefix: str | None = None  # embedding instruction; None = model default


class VoiceSettings(BaseModel):
    hotkey: str = "ctrl_l"  # tap Left Ctrl to talk; or a combo like "<ctrl>+<alt>+j"
    # speech-to-text (faster-whisper)
    stt_model: str = "distil-large-v3.5"  # English; "large-v3-turbo" = multilingual
    stt_device: str = "auto"  # auto (GPU if it works, else CPU) | cuda | cpu
    stt_compute_type: str = "auto"  # auto = float16 on GPU, int8 on CPU
    stt_beam_size: int = 0  # 0 = auto (5 on GPU, 1 on CPU)
    language: str = "en"
    hotwords: str = "Jarvis"  # words Whisper should expect (names, jargon)
    # listening
    silence_ms: int = 700  # this much quiet after you speak = you're done
    start_timeout_s: float = 6.0  # stop listening if you don't start within this
    max_listen_s: float = 30.0
    vad_threshold: float = 0.5  # 0..1; raise it if background noise triggers listening
    input_device: str = ""  # "" = Windows default; or a number / part of the name
    # speaking (Kokoro)
    speak: bool = True
    tts_voice: str = "bm_george"
    tts_speed: float = 1.0
    tts_model: str = "kokoro-v1.0.onnx"  # or kokoro-v1.0.fp16.onnx (half the size)
    output_device: str = ""
    earcons: bool = True  # little beeps when listening starts / nothing was heard
    # added to the system prompt in voice mode
    style: str = (
        "Voice mode: the user is talking to you and hears your reply spoken aloud. "
        "Reply in 1-3 short sentences of plain speech: no markdown, lists, tables, emoji "
        "or code unless asked. Don't read out file paths or URLs in full. Their words "
        "come from speech recognition, so if something sounds garbled, make a sensible "
        "guess or ask."
    )


class VaultSettings(BaseModel):
    """Obsidian vault: Jarvis's memory, notes, reminders and chats as Markdown pages."""

    enabled: bool = True
    # A folder name Windows knows ("Documents/…", OneDrive-aware) or a full path.
    path: str = "Documents/obsidian_notes"
    poll_s: float = 3.0  # how often to sync
    quiet_s: float = 4.0  # leave a file alone for this long after you edit it
    conversations: bool = True  # a page per conversation
    journal: bool = True  # a page per day
    inbox: bool = True  # learn from files dropped in Inbox/
    search: bool = True  # index your notes for vault_search
    search_conversations: bool = False  # also search chat transcripts
    user_name: str = "You"  # how you're labelled in transcripts


class DesktopSettings(BaseModel):
    """Seeing the screen and using mouse & keyboard (Windows)."""

    enabled: bool = True
    max_image_side: int = 1280  # screenshots are scaled to this before the vision model
    element_limit: int = 80  # controls listed per window
    # Never looked at or controlled; add apps by their .exe name. None = built-in list.
    blocked_apps: list[str] | None = None
    no_typing_apps: list[str] | None = None  # never typed into (terminals); None = built-in
    takeover_px: int = 40  # moving the mouse this far yourself stops Jarvis


class OllamaSettings(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    keep_alive: str = "30m"
    timeout_s: float = 120
    think: bool | None = None


class ProvidersSettings(BaseModel):
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


def default_config_dir() -> Path:
    if env := os.environ.get("JARVIS_CONFIG_DIR"):
        return Path(env)
    cwd = Path.cwd() / "config"
    return cwd if (cwd / "default.toml").exists() else REPO_ROOT / "config"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def read_toml_layers(config_dir: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name in ("default.toml", "local.toml"):
        path = config_dir / name
        if path.exists():
            with path.open("rb") as f:
                merged = deep_merge(merged, tomllib.load(f))
    return merged


class _TomlLayersSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], data: dict[str, Any]):
        super().__init__(settings_cls)
        self._data = data

    def get_field_value(self, field, field_name):  # pragma: no cover - unused
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JARVIS_", env_nested_delimiter="__", extra="ignore"
    )

    app: AppSettings = Field(default_factory=AppSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    vault: VaultSettings = Field(default_factory=VaultSettings)
    desktop: DesktopSettings = Field(default_factory=DesktopSettings)
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    # Set by load_settings(); not read from config.
    config_dir: Path = Field(default_factory=default_config_dir, exclude=True)

    _toml_data: ClassVar[dict[str, Any]] = {}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Earlier sources win.
        return (init_settings, env_settings, _TomlLayersSource(settings_cls, cls._toml_data))


def load_settings(config_dir: Path | None = None, **overrides: Any) -> Settings:
    """Load settings from the TOML layers in `config_dir` plus env vars."""
    config_dir = Path(config_dir) if config_dir else default_config_dir()
    Settings._toml_data = read_toml_layers(config_dir)
    try:
        settings = Settings(config_dir=config_dir, **overrides)
    finally:
        Settings._toml_data = {}
    if not settings.app.data_dir.is_absolute():  # relative to the repo, not the cwd
        settings.app.data_dir = (config_dir.parent / settings.app.data_dir).resolve()
    return settings
