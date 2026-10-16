import json
from contextlib import aclosing

import pytest

from jarvis.core.types import Message
from jarvis.memory.extractor import MemoryExtractor, MemoryWorker, worth_checking
from jarvis.memory.service import Embedder, MemoryService
from jarvis.memory.store import MemoryStore
from jarvis.memory.vectors import cosine, keyword_score, normalize, pack, toy_embed, unpack
from jarvis.providers import ModelManager
from jarvis.providers.scripted import ScriptedProvider
from jarvis.storage import Database

EMBED_CFG = '[models]\nroles = { chat = "scripted:main", embed = "scripted:emb" }\n'


def ops(*items) -> str:
    return json.dumps({"operations": list(items)})


def make_service(provider=None, *, embed=True, **kw) -> MemoryService:
    provider = provider or ScriptedProvider()
    roles = {"chat": "scripted:main", "memory": "scripted:main"}
    if embed:
        roles["embed"] = "scripted:emb"
    models = ModelManager({"scripted": provider}, roles)
    return MemoryService(MemoryStore(Database(":memory:")), Embedder(models, query_prefix=""), **kw)


# ------------------------------------------------------------------ vectors


def test_vectors_roundtrip_and_similarity():
    v = normalize([3, 4])
    assert unpack(pack(v)) == pytest.approx(v)
    assert cosine(v, v) == pytest.approx(1)
    assert cosine(toy_embed("my sister Priya"), toy_embed("sisters named priya")) > 0.8
    assert cosine(toy_embed("my sister Priya"), toy_embed("quarterly revenue")) < 0.2
    assert keyword_score("who is Priya?", "User's sister is named Priya.") == 1.0
    assert keyword_score("weather today", "User's sister is Priya.") == 0.0


# ------------------------------------------------------------------ store


def test_store_crud_and_stale_embeddings():
    s = MemoryStore(Database(":memory:"))
    m = s.add("User lives in Hyderabad.", category="profile", embedding=[1.0, 0.0], embed_model="x")
    assert s.get(m.id).category == "profile" and s.count() == 1
    assert s.missing_embeddings("x") == []
    s.update(m.id, category="profile")  # same text → embedding kept
    assert s.missing_embeddings("x") == []
    s.update(m.id, text="User lives in Bangalore.")  # text changed → needs re-embedding
    assert [x.id for x in s.missing_embeddings("x")] == [m.id]
    assert s.add("x", category="nonsense").category == "other"
    s.mark_used([m.id])
    assert s.get(m.id).use_count == 1
    assert s.delete(m.id) and not s.delete(m.id)
    assert s.clear() == 1


def test_store_search_vector_and_keyword_fallback():
    s = MemoryStore(Database(":memory:"))
    a = s.add(
        "User's sister is named Priya.",
        embedding=toy_embed("sister named Priya"),
        embed_model="toy",
    )
    s.add("User runs a company called Exinent.")  # no embedding → keyword only
    # vector path alone (keyword matching disabled)
    hits = s.search(
        "my sister",
        query_vector=toy_embed("sister Priya"),
        embed_model="toy",
        min_score=0.5,
        min_keyword=1.01,
    )
    assert hits[0].memory.id == a.id and hits[0].how == "vector"
    # with both, the better score wins (exact words → keyword 1.0)
    both = s.search("my sister", query_vector=toy_embed("my sister"), embed_model="toy")
    assert both[0].memory.id == a.id and both[0].score == 1.0
    kw = s.search(
        "what is Exinent",
        query_vector=toy_embed("what is Exinent"),
        embed_model="toy",
        min_score=0.5,
    )
    assert kw[0].memory.text.startswith("User runs") and kw[0].how == "keyword"
    # different embedding model → vectors ignored, keyword still works
    other = s.search("Priya", query_vector=[1.0] + [0.0] * 63, embed_model="other")
    assert other[0].how == "keyword"
    assert (
        s.search(
            "stock prices", query_vector=toy_embed("stock prices"), embed_model="toy", min_score=0.5
        )
        == []
    )


# ------------------------------------------------------------------ service


async def test_save_dedupes_exact_and_near():
    svc = make_service()
    m1, created = await svc.save("User's sister is named Priya.", category="person")
    assert created
    m2, created = await svc.save("user's sister is named priya.")
    assert not created and m2.id == m1.id
    m3, created = await svc.save("User's sisters named Priya")  # near-identical vector
    assert not created and m3.id == m1.id and svc.store.count() == 1
    _, created = await svc.save("User has a dog named Bruno.")
    assert created and svc.store.count() == 2


async def test_recall_without_embedding_model_uses_keywords():
    svc = make_service(embed=False)
    await svc.save("User runs a company called Exinent.", category="profile")
    assert svc.store.list()[0].embed_model is None
    hits = await svc.recall("tell me about Exinent")
    assert hits and hits[0].how == "keyword"


async def test_prepare_layers_core_and_recalled():
    svc = make_service()
    await svc.save("User's name is Adi.", category="profile")
    await svc.save("User's sister is named Priya.", category="person")
    await svc.save("User likes filter coffee.", category="preference")
    history = [
        Message.user("hello"),
        Message.assistant("hi"),
        Message.user("Is my sister visiting?"),
    ]
    out = await svc.prepare(history)
    assert "User's name is Adi" in out.system_extra  # core → system prompt
    last = out.messages[-1].content
    assert (
        last.startswith("<memory>") and "Priya" in last and last.endswith("Is my sister visiting?")
    )
    assert "coffee" not in last  # irrelevant memory not recalled
    assert out.messages[0].content == "hello"  # earlier messages untouched
    assert history[-1].content == "Is my sister visiting?"  # original list not mutated
    assert all(m.use_count == 1 for m in svc.store.list() if "coffee" not in m.text)

    svc.enabled = False
    off = await svc.prepare(history)
    assert off.system_extra is None and off.messages[-1].content == "Is my sister visiting?"


async def test_backfill_embeds_missing():
    svc = make_service(embed=False)
    await svc.save("User's name is Adi.")
    svc.embedder.role = "embed"
    svc.embedder.models.roles["embed"] = svc.embedder.models.roles["chat"].__class__(
        "scripted", "emb"
    )
    assert await svc.backfill() == 1
    assert svc.store.missing_embeddings("scripted:emb") == []


# ------------------------------------------------------------------ extractor


def test_worth_checking_filter():
    assert worth_checking("My sister Priya is visiting")
    assert worth_checking("Remember that the meeting is on Tuesday")
    assert not worth_checking("What's the weather on Mars?")
    assert not worth_checking("I'm")  # too short


async def test_extractor_add_update_delete_and_guards():
    provider = ScriptedProvider()
    svc = make_service(provider)
    home, _ = await svc.save("User lives in Hyderabad.", category="profile")
    bro, _ = await svc.save("User's brother is Rohan.", category="person")
    ex = MemoryExtractor(svc.embedder.models, svc)

    provider.turns = [
        ops(
            {
                "op": "update",
                "id": home.id,
                "text": "User lives in Bangalore.",
                "category": "profile",
            },
            {"op": "add", "text": "User started a new job at Infosys.", "category": "profile"},
            {"op": "add", "text": "User's wifi password is lemonade42."},  # secret → dropped
            {"op": "delete", "id": 9999},  # id it wasn't shown → ignored
        )
    ]
    changes = await ex.extract("I moved to Bangalore and my new job at Infosys starts Monday")
    assert [(c.op, c.text) for c in changes] == [
        ("update", "User lives in Bangalore."),
        ("add", "User started a new job at Infosys."),
    ]
    assert "lemonade42" not in json.dumps([m.text for m in svc.store.list()])
    # the model was told what's already known and asked for JSON
    call = provider.calls[-1]
    assert "User lives in Hyderabad." in call["messages"][1].content
    assert call["options"]["format"]["type"] == "object" and call["options"]["temperature"] == 0

    provider.turns = [ops({"op": "delete", "id": bro.id})]
    changes = await ex.extract("Please forget about my brother, we don't talk anymore")
    assert [c.op for c in changes] == ["delete"] and svc.store.get(bro.id) is None

    provider.turns = ["this is not json"]
    assert await ex.extract("I think my cat is cute") == []
    n = len(provider.calls)
    assert await ex.extract("What's the weather on Mars?") == []  # filtered: no LLM call
    assert len(provider.calls) == n


async def test_extractor_reports_duplicates():
    provider = ScriptedProvider()
    svc = make_service(provider)
    await svc.save("User's name is Adi.", category="profile")
    provider.turns = [ops({"op": "add", "text": "User's name is Adi.", "category": "profile"})]
    changes = await MemoryExtractor(svc.embedder.models, svc).extract("my name is Adi")
    assert [c.op for c in changes] == ["duplicate"] and svc.store.count() == 1


async def test_extractor_schema_forces_every_field_and_dry_run_saves_nothing():
    from jarvis.memory.extractor import EXTRACT_SCHEMA

    item = EXTRACT_SCHEMA["properties"]["operations"]["items"]
    # Constrained decoding only writes required fields — "op" alone gave texts-less adds.
    assert set(item["required"]) == set(item["properties"])
    provider = ScriptedProvider()
    svc = make_service(provider)
    ex = MemoryExtractor(svc.embedder.models, svc)
    provider.turns = [ops({"op": "add", "text": "User's name is Adi.", "category": "profile"})]
    assert await ex.extract("Hi, I'm Adi by the way.", force=True, dry_run=True) == []
    assert ex.last_ops[0]["text"] == "User's name is Adi." and '"add"' in ex.last_raw
    assert svc.store.count() == 0
    assert provider.calls[-1]["options"]["num_predict"] > 0  # can't loop forever
    provider.turns = [ops({"op": "add"})]  # no text → nothing saved, no crash
    assert await ex.extract("Hi, I'm Adi by the way.") == []


def test_worker_runs_in_background_thread():
    provider = ScriptedProvider([ops({"op": "add", "text": "User owns a dog named Bruno."})])
    svc = make_service(provider)
    got = []
    worker = MemoryWorker(lambda: MemoryExtractor(svc.embedder.models, svc), got.extend)
    assert worker.submit("What time is it?") is None  # filtered, no thread work
    fut = worker.submit("My dog Bruno loves walks")
    assert fut.result(timeout=5)[0].text == "User owns a dog named Bruno."
    worker.close()
    assert [c.op for c in got] == ["add"]


# ------------------------------------------------------------------ end to end


async def test_conversation_recalls_and_learns(make_app):
    app, provider = make_app(
        [
            "Hi Adi! Priya will love it.",
            ops(
                {
                    "op": "add",
                    "text": "User's sister Priya is visiting next month.",
                    "category": "person",
                }
            ),
        ],
    )
    app.models.set_role("embed", "scripted:emb")
    await app.memory.save("User's name is Adi.", category="profile")
    await app.memory.save("User's sister is named Priya.", category="person")
    learned = []
    app.memory_listeners.append(learned.extend)

    conv = app.conversations.new()
    text = "My sister Priya is visiting next month, any ideas?"
    async with aclosing(app.conversations.send(conv.id, text)) as s:
        [c async for c in s]
    app.memory_worker.flush()

    sent = provider.calls[0]["messages"]
    assert "User's name is Adi" in sent[0].content  # core fact in system prompt
    assert "<memory>" in sent[-1].content and "Priya" in sent[-1].content
    assert app.store.messages(conv.id)[0].content == text  # stored as typed
    assert [c.op for c in learned] == ["add"]
    assert any("visiting next month" in m.text for m in app.memory_store.list())
    await app.aclose()


async def test_memory_tools(make_app):
    from jarvis.tools.memory import (
        ForgetArgs,
        MemoryForget,
        MemorySave,
        MemorySearch,
        MemoryUpdate,
        SaveArgs,
        SearchArgs,
        UpdateArgs,
    )

    app, _ = make_app()
    ctx = app.tool_context()
    out = await MemorySave().run(
        SaveArgs(text="User's favourite team is Sunrisers.", category="preference"), ctx
    )
    assert out.startswith("Remembered memory #1")
    assert "Already knew" in await MemorySave().run(
        SaveArgs(text="User's favourite team is Sunrisers."), ctx
    )
    listed = await MemorySearch().run(SearchArgs(), ctx)
    assert listed["memories"][0]["text"] == "User's favourite team is Sunrisers."
    await MemoryUpdate().run(UpdateArgs(id=1, text="User's favourite team is CSK."), ctx)
    assert "CSK" in app.memory_store.get(1).text
    assert "Forgot" in await MemoryForget().run(ForgetArgs(id=1), ctx)
    from jarvis.tools import ToolError

    with pytest.raises(ToolError):
        await MemoryForget().run(ForgetArgs(id=1), ctx)
    app.memory.enabled = False
    with pytest.raises(ToolError, match="turned off"):
        await MemorySearch().run(SearchArgs(), ctx)
    await app.aclose()


async def test_remember_command_normalizes_or_falls_back(make_app):
    app, provider = make_app()
    app.models.set_role("embed", "scripted:emb")
    provider.turns = [
        ops({"op": "add", "text": "User's favourite colour is teal.", "category": "preference"})
    ]
    changes = await app.remember("my favourite colour is teal")
    assert [(c.op, c.text) for c in changes] == [("add", "User's favourite colour is teal.")]
    provider.turns = [ops()]  # learner finds nothing → saved as typed
    changes = await app.remember("garage opens with the blue remote")
    assert changes[0].text == "garage opens with the blue remote"
    await app.aclose()
