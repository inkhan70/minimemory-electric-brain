"""Regression tests for bugs found in the 0.8.2 code review."""
import gc
import json
import os
import sqlite3

import pytest

from minimemory import AssociativeMemory, FederatedBrain, KnowledgeGate, MemoryQA, TextPipeline
from minimemory.backends import SQLiteBackend
from minimemory.cli import main
from minimemory.federated import DomainStore
from minimemory.resource_router import internet_available


def _open_fds():
    path = "/proc/self/fd"
    return len(os.listdir(path)) if os.path.isdir(path) else None


def test_memoryqa_does_not_leak_connections(tmp_path):
    ai = MemoryQA(tmp_path / "m.db", semantic=False)
    ai.learn("q", "a")
    gc.disable()
    try:
        before = _open_fds()
        for _ in range(100):
            ai.count()
            ai.search("q")
        after = _open_fds()
    finally:
        gc.enable()
    if before is not None:
        assert after - before <= 2
    ai.close()


def test_sqlite_backend_does_not_leak_connections(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "b.db"))
    backend.learn_batch([("q", "a")])
    gc.disable()
    try:
        before = _open_fds()
        for _ in range(100):
            backend.count()
        after = _open_fds()
    finally:
        gc.enable()
    if before is not None:
        assert after - before <= 2


def test_qna_is_found_by_question_text(tmp_path):
    with FederatedBrain(tmp_path / "k") as brain:
        brain.remember_qna("What is photosynthesis?", "Plants turn light into sugar.", domain="science")
        result = brain.ask("photosynthesis", domains=["science"])
        assert result["status"] == "SUCCESS"


def test_unrelated_query_returns_nothing(tmp_path):
    with FederatedBrain(tmp_path / "k") as brain:
        brain.remember("Rivers flow to the sea.", domain="geography")
        assert brain.ask("quantum chromodynamics", domains=["geography"])["status"] == "EMPTY_MEMORY"
    with AssociativeMemory(str(tmp_path / "a.db")) as mem:
        mem.remember("Snowy mountain and a lake", confidence=0.9)
        assert mem.recall("quantum chromodynamics") == []


def test_mixed_case_domain_names_work(tmp_path):
    with FederatedBrain(tmp_path / "k") as brain:
        brain.remember("Rivers flow to the sea.", domain="Geography")
        result = brain.retrieve("rivers", domains=["GEOGRAPHY"])
        assert result["domains"] == ["geography"]
        assert result["results"]


def test_search_does_not_leave_open_transaction(tmp_path):
    store = DomainStore(tmp_path / "s.db")
    store.add("hello world fact")
    store.search("hello")
    assert store._conn.in_transaction is False
    store.close()


def test_routing_probe_does_not_count_as_access(tmp_path):
    with FederatedBrain(tmp_path / "k") as brain:
        brain.remember("Photosynthesis converts light to energy.", domain="science")
        brain.route("photosynthesis light")
        count = brain.stores["science"]._conn.execute("SELECT access_count FROM knowledge_items").fetchone()[0]
        assert count == 0


def test_legacy_import_missing_file_is_error_and_creates_nothing(tmp_path):
    missing = tmp_path / "nope.db"
    with FederatedBrain(tmp_path / "k") as brain:
        with pytest.raises(FileNotFoundError):
            brain.import_legacy_db(missing)
    assert not missing.exists()


def test_old_fts_index_is_migrated(tmp_path):
    root = tmp_path / "k"
    root.mkdir()
    db = root / "science.db"
    store = DomainStore(db)
    store.add_qna("What is osmosis?", "Water moving across a membrane.")
    store.close()
    conn = sqlite3.connect(db)
    # Simulate a 0.8.0-0.8.2 database: content-only external FTS + triggers.
    conn.executescript("""
        DROP TABLE knowledge_search;
        CREATE VIRTUAL TABLE knowledge_fts USING fts5(content, content='knowledge_items', content_rowid='id');
        CREATE TRIGGER knowledge_ai AFTER INSERT ON knowledge_items BEGIN
            INSERT INTO knowledge_fts(rowid, content) VALUES (new.id, new.content);
        END;
        INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild');
    """)
    conn.commit()
    conn.close()
    store = DomainStore(db)
    assert store.search("osmosis")
    store.add("Mitochondria produce energy.")
    assert store.search("mitochondria")
    store.close()


def test_associative_creates_missing_parent_directory(tmp_path):
    with AssociativeMemory(str(tmp_path / "sub" / "dir" / "x.db")) as mem:
        assert mem.remember("hello") > 0


def test_recall_embeds_query_once(tmp_path):
    calls = []

    def embed(text):
        calls.append(text)
        return [1.0, 0.0] if "cat" in text else [0.0, 1.0]

    with AssociativeMemory(str(tmp_path / "e.db"), embedding_fn=embed) as mem:
        for i in range(5):
            mem.remember(f"cat fact {i}")
        calls.clear()
        mem.recall("cat")
        assert calls == ["cat"]


def test_associate_unknown_memory_raises_keyerror(tmp_path):
    with AssociativeMemory(str(tmp_path / "a.db")) as mem:
        mem.upsert_entity("e1", "Entity")
        with pytest.raises(KeyError):
            mem.associate(999, "e1")


def test_explicit_rejected_status_survives_conflict(tmp_path):
    gate = KnowledgeGate(str(tmp_path / "g.db"))
    gate.admit("The sky is blue today", source="a", admission_score=0.9)
    out = gate.admit("The sky is not blue today", source="b", admission_score=0.9, verification_status="rejected")
    assert out["status"] == "rejected"
    gate.close()
    gate.close()  # idempotent


def test_pipeline_runs_extra_pass_after_spelling_change():
    seen = []

    def spelling(text):
        seen.append(text)
        return text.replace("teh", "the", 1)

    result = TextPipeline(spelling=spelling).process("teh teh", max_passes=3)
    assert result["corrected"] == "the the"
    assert len(seen) >= 2


def test_internet_probe_rejects_non_http_urls():
    with pytest.raises(ValueError):
        internet_available(probe_urls=("file:///etc/passwd",))


def test_cli_reports_sqlite_errors_instead_of_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.db"
    bad.write_text("this is not a sqlite database")
    assert main(["--db", str(bad), "stats"]) == 1
    assert "Error" in capsys.readouterr().err


def test_cli_federated_domains_are_trimmed(tmp_path, capsys):
    root = str(tmp_path / "k")
    assert main(["federated-remember", "Rivers flow.", "--root", root, "--domain", "geography"]) == 0
    capsys.readouterr()
    assert main(["federated-ask", "rivers", "--root", root, "--domains", " geography , qna "]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "SUCCESS"
