"""Portable end-to-end regression suite for minimemory.

This file intentionally uses only the core package and pytest. It is suitable
for GitHub checkouts and Android/Termux installations without network access,
model downloads, or external AI credentials.
"""
from pathlib import Path

import pytest

from minimemory import (
    LocalBrain,
    MemoryQA,
    QueryKind,
    QueryRoute,
    classify_query,
    read_documentation,
    update_documentation,
)


def test_qna_learn_search_ask_update_delete_health(tmp_path):
    db = tmp_path / "memory.db"
    with MemoryQA(db) as memory:
        memory_id = memory.learn("What is Python?", "Python is a programming language.", confidence=0.95)
        assert memory_id > 0
        assert memory.ask("What is Python?")
        assert memory.search("Python", top_k=5)
        assert memory.get(memory_id)["answer"].startswith("Python is")
        assert memory.delete(memory_id)
        assert memory.health()["ok"] is True


def test_associative_entities_relations_and_bundle(tmp_path):
    with LocalBrain(tmp_path / "brain.db") as brain:
        brain.associative.upsert_entity("place:denver-co", "Denver", entity_type="city", aliases=["Denver Colorado"])
        brain.associative.upsert_entity("country:usa", "United States", entity_type="country", aliases=["USA"])
        brain.associative.add_relation("place:denver-co", "located_in", "country:usa", confidence=0.95)
        mid = brain.remember("Denver is a city in Colorado.", memory_type="semantic", confidence=0.95)
        assert mid > 0
        assert brain.recall("Denver")
        bundle = brain.bundle("Denver")
        assert "memories" in bundle


def test_query_classifier_text_code_and_mixed():
    assert classify_query("Where is Denver?")["kind"] == QueryKind.TEXT.value
    assert classify_query("```python\nprint(\'hello\')\n```")["kind"] == QueryKind.CODE.value
    mixed = classify_query("Use Python to run:\n```python\ndef add(a, b):\n    return a + b\n```\n")
    assert mixed["kind"] == QueryKind.CODE.value


def test_local_brain_routes_qna_memory_code_programming(tmp_path):
    with LocalBrain(tmp_path / "brain.db", behavior_tracking=True) as brain:
        brain.memory.learn("Where is Denver?", "Denver is a city in Colorado.", confidence=1.0)
        brain.remember("The user is building an Android app with Termux.", confidence=0.95)
        brain.associative.add_code_component(
            "python:calculator", "Calculator", language="Python", purpose="simple calculator",
            code="def add(a, b):\n    return a + b\n", confidence=1.0,
        )
        brain.seed_programming_knowledge()

        qna = brain.ask("Where is Denver?")
        assert qna["route"] == QueryRoute.QNA.value
        assert "Colorado" in qna["answer"]

        mem = brain.ask("What do you remember about my Android app?")
        assert mem["route"] == QueryRoute.MEMORY.value
        assert "Android" in mem["answer"]

        code = brain.ask("```python\ndef add(a, b):\n    return a + b\n```")
        assert code["route"] == QueryRoute.CODE.value
        assert "return a + b" in code["answer"]

        prog = brain.ask("What does a Python function mean?")
        assert prog["route"] in {QueryRoute.PROGRAMMING.value, QueryRoute.CODE.value}


def test_ai_fallback_and_controlled_learning(tmp_path):
    with LocalBrain(tmp_path / "brain.db") as brain:
        calls = []

        def generator(question, context=None):
            calls.append((question, context))
            return "A locally generated answer."

        result = brain.ask("What is an unknown local topic?", generator=generator, auto_learn=False)
        assert result["route"] == "ai_fallback"
        assert result["answer"] == "A locally generated answer."
        assert calls


def test_behavior_evolution_and_sensitive_privacy(tmp_path):
    with LocalBrain(tmp_path / "brain.db", behavior_tracking=True) as brain:
        normal = brain.observe("I am working on Python and Termux")
        assert normal["stored"] is True
        sensitive = brain.observe("my password is super-secret")
        assert sensitive["stored"] is False
        status = brain.behavior_status()
        assert status["stored_events"] == 1
        assert status["evolved_topics"] >= 1
        evolved = brain.evolve(min_accesses=1)
        assert "topics" in evolved


def test_behavior_disabled_is_transient(tmp_path):
    with LocalBrain(tmp_path / "brain.db", behavior_tracking=False) as brain:
        result = brain.observe("Python project")
        assert result["stored"] is False
        assert brain.behavior_status()["stored_events"] == 0


def test_knowledge_pack_round_trip(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    pack = tmp_path / "brain.mmpack"
    with LocalBrain(source) as brain:
        brain.memory.learn("What is SQLite?", "SQLite is a local database engine.", confidence=0.95)
        brain.remember("SQLite is useful for local applications.", confidence=0.9)
        out = brain.export_knowledge(str(pack))
        assert Path(out).exists()
    with LocalBrain(target) as brain:
        inspected = brain.inspect_knowledge_pack(str(pack))
        assert inspected["format"] == "minimemory-mmpack"
        imported = brain.import_knowledge(str(pack))
        assert imported
        assert brain.memory.search("SQLite")


def test_documentation_requires_approval(tmp_path):
    doc = tmp_path / "documentation.md"
    blocked = update_documentation(str(doc), "notes", "Unapproved text")
    assert blocked["updated"] is False
    assert not doc.exists()


def test_documentation_update_is_section_scoped_and_repeatable(tmp_path):
    doc = tmp_path / "documentation.md"
    first = update_documentation(str(doc), "routing", "The router is deterministic.", approved=True, heading="Routing")
    assert first["action"] == "created"
    second = update_documentation(str(doc), "routing", "The router checks local evidence first.", approved=True, heading="Routing")
    assert second["action"] == "replaced"
    text = read_documentation(doc)
    assert text.count("minimemory:section:routing:start") == 1
    assert "checks local evidence first" in text
    assert "router is deterministic" not in text


def test_backup_and_health(tmp_path):
    db = tmp_path / "brain.db"
    backup = tmp_path / "backup.db"
    with LocalBrain(db) as brain:
        brain.memory.learn("A", "B", confidence=1.0)
        assert brain.health()["ok"] is True
        result = brain.memory.backup(str(backup))
        assert Path(result).exists()


def test_invalid_inputs_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        classify_query("")
    with pytest.raises(ValueError):
        LocalBrain(tmp_path / "brain.db").ask("")
    with pytest.raises(ValueError):
        update_documentation(tmp_path / "d.md", "bad name!", "x", approved=True)


def test_public_exports_and_version():
    import minimemory
    assert hasattr(minimemory, "LocalBrain")
    assert hasattr(minimemory, "update_documentation")
    assert isinstance(minimemory.__version__, str)
