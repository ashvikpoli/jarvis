from jarvis.core.types import Usage
from jarvis.storage import ConversationStore, Database
from jarvis.storage.db import MIGRATIONS


def make_store() -> ConversationStore:
    return ConversationStore(Database(":memory:"))


def test_migrations_idempotent(tmp_path):
    db = Database(tmp_path / "j.db")
    assert db.schema_version == len(MIGRATIONS)
    assert db.migrate() == len(MIGRATIONS)
    db.close()
    assert Database(tmp_path / "j.db").schema_version == len(MIGRATIONS)


def test_create_list_rename():
    s = make_store()
    a = s.create()
    b = s.create("Second")
    assert a.title is None and b.title == "Second"
    s.rename(a.id, "First")
    assert s.get(a.id).title == "First"
    assert [c.id for c in s.list()][0] == a.id  # most recently updated first
    assert s.latest().id == a.id


def test_messages_roundtrip_and_counts():
    s = make_store()
    c = s.create()
    s.add_message(c.id, "user", "hi")
    m = s.add_message(
        c.id,
        "assistant",
        "hello",
        thinking="hmm",
        model="echo:echo",
        usage=Usage(prompt_tokens=3, completion_tokens=1, ttft_ms=5, answer_ttft_ms=9),
        interrupted=True,
    )
    assert m.interrupted and m.answer_ttft_ms == 9 and m.thinking == "hmm"
    msgs = s.messages(c.id)
    assert [x.role for x in msgs] == ["user", "assistant"]
    assert s.get(c.id).message_count == 2
    assert msgs[1].to_message().content == "hello"


def test_delete_cascades():
    s = make_store()
    c = s.create()
    s.add_message(c.id, "user", "hi")
    assert s.delete(c.id)
    assert s.get(c.id) is None
    assert s.messages(c.id) == []
    assert not s.delete(c.id)
