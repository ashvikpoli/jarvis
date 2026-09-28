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
    roles: dict[str, str] = Field(default_factory=lambda: {"chat": "ollama:qwen3:8b"})
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


class FileToolSettings(BaseModel):
    # Known folder names (resolved via Windows, OneDrive-aware) or explicit paths.
    roots: list[str] = Field(default_factory=lambda: ["Documents", "Desktop", "Downloads"])
    max_read_chars: int = 20_000


class ToolsSettings(BaseModel):
    enabled: bool = True
    max_steps: int = 6
    disabled: list[str] = Field(default_factory=list)  # tool names to leave out entirely
    permissions: ToolPermissions = Field(default_factory=ToolPermissions)
    files: FileToolSettings = Field(default_factory=FileToolSettings)
    notifications: bool = True


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
