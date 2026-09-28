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

    async def stream_chat(self, messages, model, *, options=None) -> AsyncIterator[StreamChunk]:
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
