import asyncio
from contextlib import aclosing
from datetime import datetime

import pytest

from jarvis.conversation import ConversationManager, ConversationNotFound, PromptBuilder, make_title
from jarvis.core.types import Message
from jarvis.providers import EchoProvider, ModelManager
from jarvis.storage import ConversationStore, Database


def fixed_clock():
    return datetime(2026, 9, 28, 10, 30)


def words_estimator(text: str) -> int:
    return len(text.split())


# ------------------------------------------------------------------ prompt builder


def test_prompt_includes_system_and_datetime():
    pb = PromptBuilder("You are Jarvis.", clock=fixed_clock)
    built = pb.build([Message.user("hi")])
    assert built.messages[0].role == "system"
    assert "You are Jarvis." in built.messages[0].content
    assert "28 September 2026" in built.messages[0].content
    assert built.messages[-1].content == "hi"
    assert built.dropped == 0


def test_prompt_trims_oldest_first_and_keeps_last():
    pb = PromptBuilder(
        "sys",
        context_tokens=100,
        reserve_tokens=0,
        include_datetime=False,
        estimator=words_estimator,
    )
    # budget: 100 - 30 (note reserve) - (1 + 4 system) = 65; each msg = 10 words + 4 = 14
    history = [Message.user(f"m{i} " + "w " * 9) for i in range(10)]
    built = pb.build(history)
    kept = [m.content.split()[0] for m in built.messages[1:]]
    assert kept == ["m6", "m7", "m8", "m9"]
    assert built.dropped == 6
    assert "6 earlier messages" in built.messages[0].content


def test_prompt_always_keeps_last_message_even_if_huge():
    pb = PromptBuilder(
        "sys",
        context_tokens=50,
        reserve_tokens=0,
        include_datetime=False,
        estimator=words_estimator,
    )
    built = pb.build([Message.user("old"), Message.user("w " * 500)])
    assert len(built.messages) == 2 and built.dropped == 1


def test_prompt_respects_caller_system_message():
    pb = PromptBuilder("default", include_datetime=False)
    built = pb.build([Message.system("custom"), Message.user("hi")])
    assert built.messages[0].content == "custom"
    assert len(built.messages) == 2


def test_make_title():
    assert make_title("  hello   world ") == "hello world"
    assert make_title("x" * 100).endswith("…") and len(make_title("x" * 100)) == 60


# ------------------------------------------------------------------ manager


def make_manager(delay=0.0) -> ConversationManager:
    models = ModelManager({"echo": EchoProvider(delay_s=delay)}, {"chat": "echo:echo"})
    store = ConversationStore(Database(":memory:"))
    return ConversationManager(models, store, PromptBuilder("sys", clock=fixed_clock))


async def collect(stream):
    async with aclosing(stream) as s:
        return [c async for c in s]


async def test_send_saves_both_turns_and_titles():
    cm = make_manager()
    conv = cm.new()
    chunks = await collect(cm.send(conv.id, "hello there"))
    assert "".join(c.text for c in chunks) == "You said: hello there"
    msgs = cm.store.messages(conv.id)
    assert [(m.role, m.content) for m in msgs] == [
        ("user", "hello there"),
        ("assistant", "You said: hello there"),
    ]
    assert msgs[1].model == "echo:echo" and not msgs[1].interrupted
    assert msgs[1].answer_ttft_ms is not None
    assert cm.store.get(conv.id).title == "hello there"


async def test_history_is_sent_to_model():
    cm = make_manager()
    seen = {}
    original = cm.models.stream_chat

    async def spy(messages, role="chat", **kw):
        seen["messages"] = list(messages)
        async for c in original(messages, role, **kw):
            yield c

    cm.models.stream_chat = spy
    conv = cm.new()
    await collect(cm.send(conv.id, "one"))
    await collect(cm.send(conv.id, "two"))
    contents = [m.content for m in seen["messages"][1:]]
    assert contents == ["one", "You said: one", "two"]


async def test_interrupted_reply_is_saved_partial():
    cm = make_manager(delay=0.05)
    conv = cm.new()

    async def consume():
        await collect(cm.send(conv.id, "word " * 100))

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    msgs = cm.store.messages(conv.id)
    assert msgs[-1].role == "assistant" and msgs[-1].interrupted
    assert 0 < len(msgs[-1].content) < len("You said: " + "word " * 100)


async def test_unknown_conversation():
    cm = make_manager()
    with pytest.raises(ConversationNotFound):
        await collect(cm.send(999, "hi"))
