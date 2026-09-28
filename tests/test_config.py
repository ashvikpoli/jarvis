from pathlib import Path

from jarvis.core.config import load_settings


def write(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def test_defaults_file(tmp_path):
    write(tmp_path / "default.toml", '[models]\nroles = { chat = "ollama:a:1" }\n')
    s = load_settings(tmp_path)
    assert s.models.roles["chat"] == "ollama:a:1"
    assert s.server.port == 8765  # code default survives


def test_local_overrides_default_deeply(tmp_path):
    write(tmp_path / "default.toml", "[server]\nhost = '127.0.0.1'\nport = 1\n")
    write(tmp_path / "local.toml", "[server]\nport = 2\n")
    s = load_settings(tmp_path)
    assert s.server.port == 2
    assert s.server.host == "127.0.0.1"


def test_env_overrides_toml(tmp_path, monkeypatch):
    write(tmp_path / "default.toml", "[server]\nport = 1\n")
    monkeypatch.setenv("JARVIS_SERVER__PORT", "3")
    monkeypatch.setenv("JARVIS_MODELS__ROLES__CHAT", "echo:echo")
    s = load_settings(tmp_path)
    assert s.server.port == 3
    assert s.models.roles["chat"] == "echo:echo"


def test_data_dir_resolved_relative_to_repo(tmp_path):
    cfg = tmp_path / "config"
    cfg.mkdir()
    write(cfg / "default.toml", "[app]\ndata_dir = 'data'\n")
    s = load_settings(cfg)
    assert s.app.data_dir == (tmp_path / "data").resolve()


def test_repo_default_config_parses():
    s = load_settings(Path(__file__).parents[1] / "config")
    assert "chat" in s.models.roles
