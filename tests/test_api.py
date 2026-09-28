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
    jarvis = JarvisApp.create(
        settings, providers={"echo": EchoProvider(delay_s=0)}, db_path=":memory:"
    )
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
    jarvis = JarvisApp.create(settings, providers={"echo": Spy(delay_s=0)}, db_path=":memory:")
    with TestClient(create_api(settings, jarvis)) as c:
        c.post("/v1/chat", json={"messages": [{"role": "user", "content": "x"}], "stream": False})
    assert captured["messages"][0].role == "system"


def test_conversation_lifecycle(client):
    conv = client.post("/v1/conversations", json={}).json()
    cid = conv["id"]
    r = client.post(f"/v1/conversations/{cid}/messages", json={"content": "hi", "stream": False})
    assert r.json()["message"]["content"] == "You said: hi"

    with client.stream("POST", f"/v1/conversations/{cid}/messages", json={"content": "again"}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1]["type"] == "done"

    detail = client.get(f"/v1/conversations/{cid}").json()
    assert detail["conversation"]["title"] == "hi"
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"] * 2

    listed = client.get("/v1/conversations").json()
    assert listed[0]["id"] == cid and listed[0]["message_count"] == 4

    assert client.delete(f"/v1/conversations/{cid}").status_code == 204
    assert client.get(f"/v1/conversations/{cid}").status_code == 404


def test_send_to_missing_conversation_404(client):
    r = client.post("/v1/conversations/999/messages", json={"content": "x"})
    assert r.status_code == 404


def test_tool_events_over_sse(make_app):
    from jarvis.core.types import ToolCall

    app, _ = make_app(
        [
            [ToolCall(name="notes_add", arguments={"text": "buy milk"})],
            [ToolCall(name="files_write", arguments={"path": "Desktop/x.txt", "content": "x"})],
            "Saved your note; I couldn't create the file without your OK.",
        ]
    )
    with TestClient(create_api(app.settings, app)) as c:
        cid = c.post("/v1/conversations", json={}).json()["id"]
        with c.stream("POST", f"/v1/conversations/{cid}/messages", json={"content": "go"}) as r:
            events = [json.loads(x[6:]) for x in r.iter_lines() if x.startswith("data: ")]
        tool_events = [(e["type"], e["name"], e["status"]) for e in events if "name" in e]
        assert tool_events == [
            ("tool_call", "notes_add", "call"),
            ("tool_result", "notes_add", "result"),
            ("tool_call", "files_write", "call"),
            ("tool_result", "files_write", "denied"),  # needs approval; none over HTTP
        ]
        tools = {t["name"]: t["permission"] for t in c.get("/v1/tools").json()}
        assert tools["files_list"] == "allow" and tools["files_delete"] == "ask"
        assert [a["tool"] for a in c.get("/v1/audit").json()] == ["files_write", "notes_add"]
