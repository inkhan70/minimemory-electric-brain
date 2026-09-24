from pathlib import Path

import pytest

from minimemory import MemoryQA
from minimemory.backends import MemoryBackend, SQLiteBackend
from minimemory.cli import main
from minimemory.packs import KnowledgePack


def test_fallback_search_and_validation(tmp_path):
    ai = MemoryQA(tmp_path / "memory.db")
    assert ai.ask("anything")["status"] == "EMPTY_MEMORY"
    assert ai.learn("What is Python?", "A programming language.")
    assert ai.ask("Tell me about Python", threshold=0.1)["status"] == "SUCCESS"
    with pytest.raises(ValueError):
        ai.learn("", "answer")
    with pytest.raises(ValueError):
        ai.ask("question", threshold=2)


def test_pack_round_trip_and_cli_export(tmp_path):
    ai = MemoryQA(tmp_path / "memory.db")
    ai.learn("q", "a")
    output = tmp_path / "pack"
    assert main(["--db", str(tmp_path / "memory.db"), "export", str(output)]) == 0
    assert KnowledgePack.from_file(str(output)).pairs == [("q", "a")]
    assert (output / "README.md").is_file()


def test_backends():
    for backend in (MemoryBackend(), SQLiteBackend(":memory:")):
        assert backend.learn_batch([("q", "a")]) == 1
        assert backend.count() == 1
        assert backend.get_all_pairs() == [("q", "a")]
        backend.clear()
        assert backend.count() == 0


def test_ai_chat_and_controlled_self_learning(tmp_path):
    ai = MemoryQA(tmp_path / "chat.db", semantic=False, min_auto_learn_confidence=0.8)

    calls = []
    def local_model(question, context):
        calls.append((question, context))
        return "Python is a programming language."

    first = ai.chat("What is Python?", local_model, threshold=0.9, auto_learn=True, approve=True)
    assert first["source"] == "ai"
    assert first["learned"] is True
    assert len(calls) == 1

    second = ai.chat("Tell me about Python", local_model, threshold=0.1)
    assert second["source"] == "memory"
    assert len(calls) == 1


def test_ai_learning_requires_approval_or_policy(tmp_path):
    ai = MemoryQA(tmp_path / "policy.db", semantic=False, min_auto_learn_confidence=0.9)
    assert not ai.learn_from_ai("q", "a", confidence=0.5)
    assert ai.count() == 0
    assert ai.learn_from_ai("q", "a", confidence=0.95)
    assert ai.count() == 1


def test_old_database_migrates(tmp_path):
    db = tmp_path / "old.db"
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE knowledge_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT UNIQUE NOT NULL,
            answer TEXT NOT NULL,
            usage_count INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.execute("INSERT INTO knowledge_memory(question, answer) VALUES (?, ?)", ("old", "memory"))
    ai = MemoryQA(db, semantic=False)
    assert ai.ask("old", threshold=0.1)["status"] == "SUCCESS"
