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
    roles: dict[str, str] = Field(default_factory=lambda: {"chat": "ollama:qwen3:14b"})


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
    if not settings.app.data_dir.is_absolute():
        settings.app.data_dir = (config_dir.parent / settings.app.data_dir).resolve()
    return settings
