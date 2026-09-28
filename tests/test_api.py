import json

import pytest
from fastapi.testclient import TestClient

from jarvis.app import JarvisApp
from jarvis.core.config import load_settings
from jarvis.interfaces.api import create_api
from jarvis.providers import EchoProvider


@pytest.fixture
def client(tmp_path):
    (tmp_path / "default.toml").write_text('[models]\nroles = { chat = "echo:echo" }\n')
    settings = load_settings(tmp_path)
    jarvis = JarvisApp.create(settings, providers={"echo": EchoProvider(delay_s=0)})
    with TestClient(create_api(settings, jarvis)) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["ok"] and r["roles"]["chat"] == "echo:echo"


def test_models(client):
    assert client.get("/v1/models").json()["echo"]["available"] == ["echo"]


def test_chat_json(client):
    r = client.post(
        "/v1/chat", json={"messages": [{"role": "user", "content": "ping"}], "stream": False}
    )
    assert r.status_code == 200
    assert r.json()["message"]["content"] == "You said: ping"


def test_chat_sse(client):
    with client.stream(
        "POST", "/v1/chat", json={"messages": [{"role": "user", "content": "ping"}]}
    ) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "You said: ping"
    assert events[-1]["type"] == "done"


def test_chat_unknown_role_400(client):
    r = client.post(
        "/v1/chat", json={"messages": [{"role": "user", "content": "x"}], "role": "vision"}
    )
    assert r.status_code == 400


def test_system_prompt_injected(tmp_path):
    captured = {}

    class Spy(EchoProvider):
        async def stream_chat(self, messages, model, *, options=None):
            captured["messages"] = list(messages)
            async for c in super().stream_chat(messages, model, options=options):
                yield c

    (tmp_path / "default.toml").write_text('[models]\nroles = { chat = "echo:echo" }\n')
    settings = load_settings(tmp_path)
    jarvis = JarvisApp.create(settings, providers={"echo": Spy(delay_s=0)})
    with TestClient(create_api(settings, jarvis)) as c:
        c.post("/v1/chat", json={"messages": [{"role": "user", "content": "x"}], "stream": False})
    assert captured["messages"][0].role == "system"
