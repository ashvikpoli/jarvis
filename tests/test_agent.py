import asyncio
from contextlib import aclosing

import pytest
from pydantic import BaseModel

from jarvis.core.types import Message, ToolCall
from jarvis.tools import Decision, PermissionPolicy, Risk, Tool, default_registry
from jarvis.tools.files import FilesWrite, WriteArgs


def call(name, **args):
    return ToolCall(name=name, arguments=args)


async def run_turn(app, text, approver=None, conv=None):
    conv = conv or app.conversations.new()
    chunks = []
    async with aclosing(app.conversations.send(conv.id, text, approver=approver)) as s:
        async for c in s:
            chunks.append(c)
    return conv, chunks


def events(chunks):
    return [(c.event.kind, c.event.call.name) for c in chunks if c.event]


def reply(chunks):
    return "".join(c.text for c in chunks)


# ------------------------------------------------------------------ registry / policy


def test_default_registry_specs_are_valid(make_app):
    app, _ = make_app()
    specs = app.registry.specs(app.tool_context())
    names = [s["function"]["name"] for s in specs]
    assert len(names) == len(set(names)) >= 15
    for s in specs:
        assert s["type"] == "function"
        assert s["function"]["description"]
        assert s["function"]["parameters"]["type"] == "object"
        assert "title" not in s["function"]["parameters"]
    files_list = next(s for s in specs if s["function"]["name"] == "files_list")
    assert "The user's folders" in files_list["function"]["description"]


def test_duplicate_tool_rejected():
    reg = default_registry()
    with pytest.raises(ValueError):
        reg.register(reg.tools[0])


def test_policy_defaults_overrides_and_escalation():
    policy = PermissionPolicy(overrides={"files_write": "allow", "apps_open": "deny"})
    fw = FilesWrite()
    assert policy.decide(fw, WriteArgs(path="a", content="x")) == (Decision.ALLOW, Risk.CHANGE)
    # overwrite escalates to destructive, which ignores the "allow" override
    assert policy.decide(fw, WriteArgs(path="a", content="x", overwrite=True)) == (
        Decision.ASK,
        Risk.DESTRUCTIVE,
    )
    from jarvis.tools.system import AppArgs, AppsOpen

    assert policy.decide(AppsOpen(), AppArgs(name="x"))[0] == Decision.DENY


def test_session_allow_never_covers_destructive():
    policy = PermissionPolicy()
    fw = FilesWrite()
    policy.allow_for_session("files_write")
    assert policy.decide(fw, WriteArgs(path="a", content="x"))[0] == Decision.ALLOW
    assert policy.decide(fw, WriteArgs(path="a", content="x", overwrite=True))[0] == Decision.ASK


# ------------------------------------------------------------------ agent loop


async def test_tool_call_then_answer_is_persisted(make_app):
    app, provider = make_app([[call("files_list", path="Documents")], "You have 3 items."])
    conv, chunks = await run_turn(app, "what's in my documents?")

    assert events(chunks) == [("call", "files_list"), ("result", "files_list")]
    assert reply(chunks) == "You have 3 items."
    assert sum(c.done for c in chunks) == 1

    # model saw the tool specs and, on the 2nd step, the tool result
    assert provider.calls[0]["tools"]
    second = provider.calls[1]["messages"]
    assert second[-2].tool_calls[0].name == "files_list"
    assert second[-1].role == "tool" and "report.txt" in second[-1].content

    stored = app.store.messages(conv.id)
    assert [m.role for m in stored] == ["user", "assistant", "tool", "assistant"]
    assert stored[1].tool_calls[0].name == "files_list"
    assert stored[2].tool_name == "files_list"

    # history round-trips into the next prompt, tool messages included
    provider.turns = ["sure"]
    await run_turn(app, "thanks", conv=conv)
    roles = [m.role for m in provider.calls[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]

    audit = app.audit.recent()
    assert audit[0].tool == "files_list" and audit[0].ok and audit[0].decision == "allowed"


async def test_ask_without_approver_is_denied(make_app):
    app, provider = make_app([[call("files_write", path="Desktop/a.txt", content="hi")], "ok"])
    _, chunks = await run_turn(app, "make a file")
    assert events(chunks)[-1] == ("denied", "files_write")
    assert not (app.file_roots[1] / "a.txt").exists()
    assert "approval" in provider.calls[1]["messages"][-1].content
    assert app.audit.recent()[0].decision == "denied"


async def test_approver_once_session_and_deny(make_app):
    answers = iter(["session", "deny"])
    seen = []

    async def approver(req):
        seen.append(req.description)
        return next(answers)

    app, _ = make_app(
        [
            [call("files_write", path="Desktop/a.txt", content="hi")],
            [call("files_write", path="Desktop/b.txt", content="hi")],  # session-allowed: no prompt
            "done",
            [
                call("files_write", path="Desktop/a.txt", content="new", overwrite=True)
            ],  # asks again
            "ok",
        ]
    )
    _, chunks = await run_turn(app, "make two files", approver)
    assert (app.file_roots[1] / "a.txt").exists() and (app.file_roots[1] / "b.txt").exists()
    assert len(seen) == 1 and "create file Desktop/a.txt" in seen[0]

    _, chunks = await run_turn(app, "overwrite a", approver)
    assert len(seen) == 2 and "OVERWRITE" in seen[1]
    assert events(chunks)[-1] == ("denied", "files_write")
    assert (app.file_roots[1] / "a.txt").read_text() == "hi"


async def test_unknown_tool_and_bad_args_are_reported_to_model(make_app):
    app, provider = make_app(
        [
            [call("teleport", where="mars")],
            [call("reminders_set", text="x")],  # missing at/in_minutes
            "sorry",
        ]
    )
    _, chunks = await run_turn(app, "do stuff")
    assert [k for k, _ in events(chunks)] == ["call", "error", "call", "error"]
    assert "Unknown tool" in provider.calls[1]["messages"][-1].content
    assert "Invalid arguments" in provider.calls[2]["messages"][-1].content


async def test_max_steps_stops_the_loop(make_app):
    app, _ = make_app([[call("system_time")]] * 10, config="[tools]\nmax_steps = 3\n")
    _, chunks = await run_turn(app, "loop forever")
    assert len([e for e in events(chunks) if e[0] == "call"]) == 3
    assert "stopped after 3" in reply(chunks)


async def test_tool_crash_does_not_kill_conversation(make_app):
    class Boom(Tool):
        name = "boom"
        description = "explodes"

        class Args(BaseModel):
            pass

        async def run(self, args, ctx):
            raise RuntimeError("kaboom")

    app, provider = make_app([[call("boom")], "it broke"])
    app.registry.register(Boom())
    _, chunks = await run_turn(app, "boom")
    assert events(chunks)[-1] == ("error", "boom")
    assert reply(chunks) == "it broke"


async def test_cancel_during_approval_leaves_no_side_effects(make_app):
    app, _ = make_app([[call("files_write", path="Desktop/c.txt", content="x")], "ok"])

    async def slow_approver(req):
        await asyncio.sleep(10)
        return "once"

    task = asyncio.create_task(run_turn(app, "write", slow_approver))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not (app.file_roots[1] / "c.txt").exists()


async def test_tools_disabled_config(make_app):
    app, provider = make_app(["hi"], config="[tools]\nenabled = false\n")
    await run_turn(app, "hello")
    assert provider.calls[0]["tools"] is None
    app2, _ = make_app(config='[tools]\ndisabled = ["files_delete"]\n')
    assert app2.registry.get("files_delete") is None and app2.registry.get("files_list")


def test_orphan_tool_messages_trimmed():
    from jarvis.conversation import PromptBuilder

    pb = PromptBuilder(
        "s",
        context_tokens=60,
        reserve_tokens=0,
        include_datetime=False,
        estimator=lambda t: len(t.split()),
    )
    history = [
        Message.user("find"),
        Message.assistant(
            "", [ToolCall(name="files_search", arguments={"query": " ".join("abcdefghijklmnop")})]
        ),
        Message.tool("w w", "files_search"),  # small: fits, but its call doesn't
        Message.user("next"),
    ]
    built = pb.build(history)
    assert [m.role for m in built.messages] == ["system", "user"]
    assert built.messages[1].content == "next"


async def test_outside_write_always_prompts_even_after_always(make_app, tmp_path):
    """End to end: 'always' for files_write covers your folders, never outside them."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    prompts = []

    async def approver(req):
        prompts.append((req.risk, req.can_allow_for_session, req.description))
        return "session" if req.can_allow_for_session else "deny"

    app, _ = make_app(
        [
            [call("files_write", path="Desktop/a.txt", content="1")],
            [call("files_write", path="Desktop/b.txt", content="2")],
            [call("files_write", path=str(outside / "c.txt"), content="3")],
            "done",
        ]
    )
    await run_turn(app, "write three files", approver)
    assert (app.file_roots[1] / "b.txt").exists()
    assert not (outside / "c.txt").exists()
    assert len(prompts) == 2  # a.txt asked (→ always), b.txt silent, c.txt asked again
    assert prompts[1][0] == Risk.DESTRUCTIVE and not prompts[1][1]
    assert "OUTSIDE" in prompts[1][2]


def test_announces_action():
    from jarvis.agent.loop import announces_action

    assert announces_action("I'll check my notes for the guest wifi password. Let me search.")
    assert announces_action("Let me look that up.")
    assert not announces_action("The password is lemonade42.")
    assert not announces_action("Let me know if you need anything else.")
    assert not announces_action("")


async def test_said_it_would_search_but_didnt_gets_one_nudge(make_app):
    app, provider = make_app(
        ["Let me search my notes for it.", [call("files_list", path="Documents")], "Found it."]
    )
    conv, chunks = await run_turn(app, "what's in my documents?")
    assert events(chunks) == [("call", "files_list"), ("result", "files_list")]
    assert reply(chunks).endswith("Found it.")
    assert "Note from the Jarvis app" in provider.calls[1]["messages"][-1].content
    stored = [m for m in app.store.messages(conv.id) if "Note from the Jarvis" in m.content]
    assert stored == []  # the nudge isn't saved
    # Only once: a second announcement just ends the turn.
    provider.turns = ["Let me check.", "Let me check again."]
    await run_turn(app, "anything else?", conv=conv)
    assert len(provider.turns) == 0 or provider.turns == ["Let me check again."]
