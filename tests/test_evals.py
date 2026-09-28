from collections.abc import AsyncIterator

import pytest

from jarvis.app import JarvisApp
from jarvis.core.config import load_settings
from jarvis.core.types import ProviderHealth, StreamChunk, Usage
from jarvis.evals import Checks, EvalRunner, load_cases, print_report, run_checks, save_results
from jarvis.evals.cases import EvalCase, clean_reply
from jarvis.providers.base import ModelProvider


@pytest.mark.parametrize(
    "reply,checks,ok",
    [
        ("Canberra.", Checks(contains_all=["canberra"]), True),
        ("Sydney", Checks(contains_all=["Canberra"]), False),
        ("It's red", Checks(contains_any=["red", "blue"]), True),
        ("I turned the lights off", Checks(not_contains=["turned the lights"]), False),
        ("Yes, because...", Checks(regex=r"^\W*yes\b"), True),
        ("one two three", Checks(max_words=2), False),
        ("one", Checks(min_words=2), False),
        ("The answer is 1,408.", Checks(number=1408), True),
        ("The answer is 408.5", Checks(number=408), False),
        ('Sure: ```json\n{"name": "Ana", "age": 30}\n```', Checks(json_keys=["name", "age"]), True),
        ('{"name": "Ana"}', Checks(json_keys=["name", "age"]), False),
        ("not json", Checks(json_keys=["a"]), False),
        ("Plain sentence. Another one.", Checks(no_markdown=True), True),
        ("**Bold** claim", Checks(no_markdown=True), False),
        ("Tips:\n- one\n- two", Checks(no_markdown=True), False),
        ("", Checks(), False),
    ],
)
def test_checks(reply, checks, ok):
    assert (run_checks(reply, checks) == []) is ok


def test_clean_reply_strips_think_tags():
    assert clean_reply("<think>hmm</think>\nHello") == "Hello"


def test_repo_cases_load_and_are_valid():
    cases = load_cases()
    assert len(cases) >= 30
    assert {"persona", "reasoning", "honesty", "voice", "multi-turn"} <= {c.category for c in cases}


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[[case]]\nid="a"\ncategory="x"\nprompt="p"\n' * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_cases(p)


class ScriptedProvider(ModelProvider):
    """Replies from a dict keyed by the last user message."""

    name = "scripted"

    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.unloaded: list[str] = []

    async def stream_chat(
        self, messages, model, *, options=None, tools=None
    ) -> AsyncIterator[StreamChunk]:
        reply = self.answers.get(messages[-1].content, "no idea")
        yield StreamChunk(text=reply)
        yield StreamChunk(
            done=True, usage=Usage(completion_tokens=5, ttft_ms=10, answer_ttft_ms=10, total_ms=110)
        )

    async def list_models(self):
        return ["m"]

    async def unload(self, model):
        self.unloaded.append(model)

    async def health(self):
        return ProviderHealth(ok=True)


async def test_runner_scores_and_reports(tmp_path):
    (tmp_path / "default.toml").write_text('[models]\nroles = { chat = "scripted:main" }\n')
    settings = load_settings(tmp_path)
    provider = ScriptedProvider({"2+2?": "4", "Capital of France?": "Berlin"})
    app = JarvisApp.create(settings, providers={"scripted": provider}, db_path=":memory:")
    cases = [
        EvalCase(id="math", category="reasoning", prompt="2+2?", checks=Checks(number=4)),
        EvalCase(
            id="geo",
            category="factual",
            prompt="Capital of France?",
            checks=Checks(contains_all=["Paris"]),
        ),
    ]
    result = await EvalRunner(app).run_model("scripted:other", cases)
    await app.aclose()

    assert result.passed == 1 and result.pass_rate == 0.5
    assert result.by_category() == {"reasoning": (1, 1), "factual": (0, 1)}
    assert result.median_tokens_per_s == pytest.approx(50)
    assert provider.unloaded == ["other"]  # non-chat model freed after the run
    assert result.options["num_ctx"] == 8192

    from rich.console import Console

    console = Console(record=True, width=120)
    print_report([result], console, verbose=True)
    out = console.export_text()
    assert "1/2 (50%)" in out and "geo" in out and "By category" in out

    path = save_results([result], tmp_path / "evals")
    assert path.exists() and '"pass_rate": 0.5' in path.read_text()


def test_tool_checks():
    from jarvis.core.types import ToolCall

    calls = [ToolCall(name="reminders_set", arguments={"text": "Stretch now", "in_minutes": 10})]
    ok = Checks(tools_called=["reminders_set"], tool_args={"reminders_set": {"text": "stretch"}})
    assert run_checks("Done.", ok, calls) == []
    assert run_checks("Done.", Checks(tools_called=["notes_add"]), calls)
    assert run_checks("Done.", Checks(tools_called_any=["notes_add", "reminders_set"]), calls) == []
    assert run_checks("Done.", Checks(tools_not_called=["reminders_set"]), calls)
    assert run_checks("Done.", Checks(no_tools=True), calls)
    assert run_checks("Done.", Checks(no_tools=True), []) == []
    bad_args = Checks(tool_args={"reminders_set": {"text": "dentist"}})
    assert "don't match" in run_checks("Done.", bad_args, calls)[0]


async def test_runner_uses_sandboxed_tools(tmp_path):
    from jarvis.core.types import ToolCall
    from jarvis.providers.scripted import ScriptedProvider

    (tmp_path / "default.toml").write_text('[models]\nroles = { chat = "scripted:main" }\n')
    provider = ScriptedProvider(
        [
            [ToolCall(name="files_read", arguments={"path": "Desktop/todo.txt"})],
            "You need to buy milk.",
        ]
    )
    app = JarvisApp.create(
        load_settings(tmp_path), providers={"scripted": provider}, db_path=":memory:", file_roots=[]
    )
    case = EvalCase(
        id="todo",
        category="tools",
        prompt="what's on my todo?",
        checks=Checks(tools_called=["files_read"], contains_all=["milk"]),
    )
    result = await EvalRunner(app).run_model("scripted:main", [case])
    await app.aclose()
    r = result.cases[0]
    assert r.passed, r.failures
    assert r.tool_calls[0].startswith("files_read")
    # the tool really read the sandbox fixture, not the user's disk
    assert "buy milk" in provider.calls[1]["messages"][-1].content


async def test_runner_memory_cases(tmp_path):
    import json as _json

    from jarvis.evals.cases import SeedMemory
    from jarvis.providers.scripted import ScriptedProvider

    (tmp_path / "default.toml").write_text(
        '[models]\nroles = { chat = "scripted:main", embed = "scripted:emb" }\n'
    )
    provider = ScriptedProvider(
        [
            "Your name is Adi.",
            _json.dumps(
                {
                    "operations": [
                        {
                            "op": "update",
                            "id": 1,
                            "text": "User lives in Bangalore.",
                            "category": "profile",
                        }
                    ]
                }
            ),
        ]
    )
    app = JarvisApp.create(
        load_settings(tmp_path), providers={"scripted": provider}, db_path=":memory:", file_roots=[]
    )
    runner = EvalRunner(app)
    chat = EvalCase(
        id="m",
        category="memory",
        prompt="What's my name?",
        memories="default",
        checks=Checks(contains_all=["Adi"]),
    )
    learn = EvalCase(
        id="l",
        category="memory-extract",
        kind="extract",
        prompt="I moved to Bangalore last month.",
        memories=[SeedMemory(text="User lives in Hyderabad.", category="profile")],
        checks=Checks(ops_any=["update"], memories_any=["Bangalore"]),
    )
    skip = EvalCase(
        id="s",
        category="memory-extract",
        kind="extract",
        prompt="What's the weather on Mars?",
        checks=Checks(memories_none=True),
    )
    result = await runner.run_model("scripted:main", [chat, learn, skip])
    real_store_count = app.memory_store.count()
    await app.aclose()
    assert [c.passed for c in result.cases] == [True, True, True], [
        c.failures for c in result.cases
    ]
    # seeded memories reached the model; the user's real memory store wasn't touched
    assert "User's name is Adi" in provider.calls[0]["messages"][0].content
    assert real_store_count == 0
    assert result.cases[1].reply == "update: User lives in Bangalore."
