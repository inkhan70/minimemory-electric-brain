import json
import sqlite3
from pathlib import Path

import pytest

from minimemory import LocalBrain, BehaviorCategory, PrivacyClass, classify_behavior, __version__


def test_behavior_classifier_never_persists_sensitive_query():
    obs = classify_behavior("my password and api key")
    assert obs.category == BehaviorCategory.SENSITIVE.value
    assert obs.privacy_class == PrivacyClass.SENSITIVE.value
    assert obs.persist is False
    assert obs.evolve is False
    assert obs.query_hash


def test_opt_in_behavior_evolution(tmp_path):
    brain = LocalBrain(tmp_path / "brain.db", behavior_tracking=True)
    try:
        result = brain.observe("Python FastAPI SQLite tutorial")
        assert result["stored"] is True
        assert brain.behavior_status()["stored_events"] == 1
        assert brain.behavior_status()["top_topics"]
        sensitive = brain.observe("show me my password")
        assert sensitive["stored"] is False
        assert brain.behavior_status()["stored_events"] == 1
    finally:
        brain.close()


def test_behavior_disabled_does_not_store(tmp_path):
    brain = LocalBrain(tmp_path / "brain.db", behavior_tracking=False)
    try:
        result = brain.observe("Python programming")
        assert result["stored"] is False
        assert brain.behavior_status()["stored_events"] == 0
    finally:
        brain.close()


def test_electric_pulse_and_consolidation(tmp_path):
    brain = LocalBrain(tmp_path / "brain.db", behavior_tracking=True)
    try:
        mid = brain.associative.remember_episode("Python is useful for automation", confidence=0.9)
        brain.associative.recall("Python", top_k=1)
        result = brain.evolve(min_accesses=1)
        assert result["consolidated"] >= 1
        pulse = brain.pulse()
        assert pulse["health"]["ok"] is True
        assert "evolution" in pulse
        assert mid > 0
    finally:
        brain.close()


def test_mmpack_round_trip_excludes_behavior(tmp_path):
    a_path = tmp_path / "a.db"
    b_path = tmp_path / "b.db"
    pack = tmp_path / "knowledge.mmpack"
    with LocalBrain(a_path, behavior_tracking=True) as a:
        a.memory.learn("What is Python?", "A programming language.", confidence=0.99)
        eid = a.associative.upsert_entity("python", "Python", entity_type="language")
        a.associative.remember("Python supports automation", confidence=0.95)
        a.observe("Python automation")
        assert eid > 0
        a.export_knowledge(str(pack))
    assert pack.exists()
    with LocalBrain(b_path) as b:
        info = b.inspect_knowledge_pack(str(pack))
        assert info["format"] == "minimemory-mmpack"
        result = b.import_knowledge(str(pack))
        assert result["qna_imported"] == 1
        assert result["behavior_imported"] == 0
        assert b.memory.count() == 1
        assert b.associative.stats()["entities"] == 1
        assert b.behavior_status()["stored_events"] == 0


def test_mmpack_tamper_detected(tmp_path):
    pack = tmp_path / "knowledge.mmpack"
    with LocalBrain(tmp_path / "a.db") as brain:
        brain.remember("Python is a language", confidence=0.9)
        brain.export_knowledge(str(pack))
    import zipfile
    tampered = tmp_path / "tampered.mmpack"
    with zipfile.ZipFile(pack, "r") as src, zipfile.ZipFile(tampered, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "memories.json":
                data = data.replace(b"Python", b"Ruby", 1)
            dst.writestr(name, data)
    with LocalBrain(tmp_path / "b.db") as brain:
        with pytest.raises(ValueError, match="checksum"):
            brain.inspect_knowledge_pack(str(tampered))


def test_version_bumped():
    assert __version__ == "0.8.0"

def test_research_evidence_pipeline_without_network(tmp_path):
    from minimemory.research import WebResearcher, Source
    engine = WebResearcher(max_results=2)
    sources = [
        Source("A", "https://one.example/a", text="A" * 100, ok=True),
        Source("B", "https://two.example/b", text="B" * 100, ok=True),
    ]
    ev = engine.evidence(sources)
    assert ev.status == "supported"
    assert ev.source_count == 2
    candidate = engine.candidate("q", "answer", {"evidence": ev.__dict__})
    assert candidate["source"] == "web-research"
    assert engine.validator_accepts(candidate, lambda c: c["confidence"] > 0.5)


def test_research_storage_requires_explicit_acceptance(tmp_path):
    with LocalBrain(tmp_path / "brain.db") as brain:
        result = {"evidence": {"score": 0.9, "source_count": 2, "domain_count": 2, "sources": ["https://a", "https://b"], "status": "supported", "note": "supported"}}
        assert brain.store_research("q", "a", result) is False
        assert brain.memory.count() == 0
        assert brain.store_research("q", "a", result, approved=True) is True
        assert brain.memory.count() == 1
