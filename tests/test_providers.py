import asyncio
import json

import httpx
import pytest

from jarvis.core.config import OllamaSettings
from jarvis.core.types import Message
from jarvis.providers import EchoProvider, ModelManager, ModelRef, OllamaProvider, ProviderError

# ----------------------------------------------------------- ModelRef / manager


def test_modelref_splits_on_first_colon():
    ref = ModelRef.parse("ollama:qwen3:14b")
    assert (ref.provider, ref.model) == ("ollama", "qwen3:14b")
    assert str(ref) == "ollama:qwen3:14b"


@pytest.mark.parametrize("bad", ["qwen3", ":x", "ollama:"])
def test_modelref_rejects_bad(bad):
    with pytest.raises(ValueError):
        ModelRef.parse(bad)


def test_manager_unknown_provider():
    m = ModelManager({"echo": EchoProvider()}, {"chat": "ollama:x"})
    with pytest.raises(ProviderError, match="isn't enabled"):
        m.resolve("chat")


def test_manager_unknown_role():
    m = ModelManager({"echo": EchoProvider()}, {"chat": "echo:echo"})
    with pytest.raises(ProviderError, match="No model"):
        m.resolve("vision")


# ----------------------------------------------------------- echo


async def test_echo_streams_then_done():
    m = ModelManager({"echo": EchoProvider(delay_s=0)}, {"chat": "echo:echo"})
    chunks = [c async for c in m.stream_chat([Message.user("hi there")])]
    assert "".join(c.text for c in chunks) == "You said: hi there"
    assert chunks[-1].done and chunks[-1].usage.completion_tokens == 4
    assert sum(c.done for c in chunks) == 1


async def test_stream_cancellation_stops_promptly():
    provider = EchoProvider(delay_s=0.05)
    received: list[str] = []

    async def consume():
        async for c in provider.stream_chat([Message.user("word " * 200)], "echo"):
            received.append(c.text)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.12)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert 0 < len(received) < 10


# ----------------------------------------------------------- ollama (mocked server)


def ndjson(*objs) -> bytes:
    return b"".join(json.dumps(o).encode() + b"\n" for o in objs)


def make_ollama(handler) -> OllamaProvider:
    return OllamaProvider(OllamaSettings(think=False), transport=httpx.MockTransport(handler))


async def test_ollama_stream_parses_content_thinking_usage():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"role": "assistant", "content": "", "thinking": "hmm"}, "done": False},
                {"message": {"role": "assistant", "content": "Hel"}, "done": False},
                {"message": {"role": "assistant", "content": "lo"}, "done": False},
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "prompt_eval_count": 12,
                    "eval_count": 2,
                },
            ),
        )

    p = make_ollama(handler)
    chunks = [c async for c in p.stream_chat([Message.user("hi")], "qwen3:14b")]
    await p.aclose()

    assert seen["body"]["model"] == "qwen3:14b"
    assert seen["body"]["stream"] is True
    assert seen["body"]["think"] is False
    assert "".join(c.text for c in chunks) == "Hello"
    assert chunks[0].thinking == "hmm"
    final = chunks[-1]
    assert final.done and final.usage.prompt_tokens == 12 and final.usage.completion_tokens == 2
    assert final.usage.ttft_ms is not None


async def test_ollama_error_line_raises():
    p = make_ollama(lambda r: httpx.Response(200, content=ndjson({"error": "model not found"})))
    with pytest.raises(ProviderError, match="model not found"):
        [c async for c in p.stream_chat([Message.user("hi")], "nope")]
    await p.aclose()


async def test_ollama_http_error_raises():
    p = make_ollama(lambda r: httpx.Response(404, text="not found"))
    with pytest.raises(ProviderError, match="404"):
        [c async for c in p.stream_chat([Message.user("hi")], "nope")]
    await p.aclose()


async def test_ollama_unreachable_is_friendly():
    def handler(request):
        raise httpx.ConnectError("refused")

    p = make_ollama(handler)
    health = await p.health()
    assert not health.ok and "Is Ollama running" in health.detail
    await p.aclose()


async def test_ollama_models_and_unload():
    calls = []

    def handler(request: httpx.Request):
        calls.append((request.method, request.url.path, request.content))
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:14b"}]})
        if request.url.path == "/api/ps":
            return httpx.Response(
                200, json={"models": [{"name": "qwen3:14b", "size": 10, "size_vram": 10}]}
            )
        return httpx.Response(200, json={})

    p = make_ollama(handler)
    assert await p.list_models() == ["qwen3:14b"]
    loaded = await p.loaded_models()
    assert loaded[0].vram_bytes == 10
    await p.unload("qwen3:14b")
    assert json.loads(calls[-1][2]) == {"model": "qwen3:14b", "keep_alive": 0}
    await p.aclose()


async def test_ollama_option_routing_and_answer_ttft():
    bodies = []

    def handler(request: httpx.Request):
        bodies.append(json.loads(request.content))
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={})
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"content": "", "thinking": "hmm"}, "done": False},
                {"message": {"content": "Hi"}, "done": False},
                {"message": {"content": ""}, "done": True, "eval_count": 1},
            ),
        )

    m = ModelManager(
        {"ollama": make_ollama(handler)},
        {"chat": "ollama:qwen3:14b"},
        role_options={"chat": {"think": True, "temperature": 0.2}},
        default_options={"num_ctx": 8192},
    )
    await m.warmup("chat")
    chunks = [c async for c in m.stream_chat([Message.user("x")])]
    await m.aclose()

    load, chat = bodies
    assert load["options"] == {"num_ctx": 8192, "temperature": 0.2}  # same ctx => no reload
    assert "think" not in load
    assert chat["think"] is True
    assert chat["options"] == {"num_ctx": 8192, "temperature": 0.2}
    usage = chunks[-1].usage
    assert usage.answer_ttft_ms >= usage.ttft_ms


async def test_ollama_tool_calls_roundtrip():
    from jarvis.core.types import ToolCall

    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=ndjson(
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "files_list",
                                    "arguments": {"path": "Documents"},
                                }
                            },
                            {"function": {"name": "system_time", "arguments": "{}"}},
                        ],
                    },
                    "done": False,
                },
                {"message": {"content": ""}, "done": True, "eval_count": 3},
            ),
        )

    p = make_ollama(handler)
    history = [
        Message.user("hi"),
        Message.assistant("", [ToolCall(name="notes_add", arguments={"text": "x"})]),
        Message.tool("Saved note #1.", "notes_add", "abc"),
    ]
    tools = [{"type": "function", "function": {"name": "files_list", "parameters": {}}}]
    chunks = [c async for c in p.stream_chat(history, "qwen3:8b", tools=tools)]
    await p.aclose()

    body = seen["body"]
    assert body["tools"] == tools
    assert body["messages"][1]["tool_calls"] == [
        {"function": {"name": "notes_add", "arguments": {"text": "x"}}}
    ]
    assert body["messages"][2] == {
        "role": "tool",
        "content": "Saved note #1.",
        "tool_name": "notes_add",
    }
    calls = chunks[0].tool_calls
    assert [c.name for c in calls] == ["files_list", "system_time"]
    assert calls[0].arguments == {"path": "Documents"} and calls[1].arguments == {}
    assert chunks[-1].done
