import sqlite3
from minimemory import FederatedBrain


def test_cross_domain_routing_and_framing(tmp_path):
    brain = FederatedBrain(tmp_path / "knowledge")
    brain.remember("Photosynthesis allows plants to convert light energy into chemical energy.", domain="science", source="science-book", confidence=.95)
    brain.remember("Forests are ecosystems containing trees, plants, animals and other organisms.", domain="nature", source="nature-book", confidence=.94)
    result = brain.ask("How do plants use sunlight in forests?", max_domains=2, top_k=6)
    assert result["status"] == "SUCCESS"
    assert "science" in result["domains"]
    assert "nature" in result["domains"]
    assert result["answer"]
    assert len(result["provenance"]) >= 2
    brain.close()


def test_qna_can_live_in_a_dedicated_domain(tmp_path):
    brain = FederatedBrain(tmp_path / "knowledge")
    mid = brain.remember_qna("What is Python?", "Python is a general-purpose programming language.", source="qna-source", confidence=.98)
    result = brain.ask("What is Python?", domains=["qna"], top_k=3)
    assert result["status"] == "SUCCESS"
    assert result["evidence"][0]["id"] == mid
    assert "Python" in result["answer"]
    brain.close()


def test_original_memory_is_not_rewritten_by_outgoing_flow(tmp_path):
    root = tmp_path / "knowledge"
    brain = FederatedBrain(root)
    mid = brain.remember("Trees store carbon in biomass and soils.", domain="environment", source="trusted-source", confidence=.91)
    before = sqlite3.connect(root / "environment.db").execute(
        "SELECT content, source, confidence, created_at FROM knowledge_items WHERE id=?", (mid,)
    ).fetchone()
    result = brain.ask("How do trees store carbon?", domains=["environment"], log_answer=True)
    assert result["status"] == "SUCCESS"
    after = sqlite3.connect(root / "environment.db").execute(
        "SELECT content, source, confidence, created_at FROM knowledge_items WHERE id=?", (mid,)
    ).fetchone()
    assert before == after
    answer_events = sqlite3.connect(root / "frame.db").execute("SELECT COUNT(*) FROM answer_events").fetchone()[0]
    assert answer_events == 1
    brain.close()


def test_append_only_duplicate_does_not_update_original(tmp_path):
    brain = FederatedBrain(tmp_path / "knowledge")
    first = brain.remember("A stable fact.", domain="science", source="book", confidence=.8)
    second = brain.remember("A stable fact.", domain="science", source="book", confidence=.2)
    assert first == second
    conn = sqlite3.connect(tmp_path / "knowledge" / "science.db")
    row = conn.execute("SELECT confidence FROM knowledge_items WHERE id=?", (first,)).fetchone()
    count = conn.execute("SELECT COUNT(*) FROM knowledge_items").fetchone()[0]
    assert row[0] == .8
    assert count == 1
    conn.close()
    brain.close()


def test_health_and_all_domain_files(tmp_path):
    brain = FederatedBrain(tmp_path / "knowledge")
    health = brain.health()
    assert health["ok"] is True
    assert health["frame"]["ok"] is True
    for name in ("memory", "qna", "coding", "science", "nature", "environment"):
        assert health["domains"][name]["ok"] is True
    brain.close()


def test_legacy_import_is_non_destructive(tmp_path):
    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy)
    conn.execute("CREATE TABLE knowledge_memory (id INTEGER PRIMARY KEY, question TEXT UNIQUE, answer TEXT, source TEXT, confidence REAL, metadata_json TEXT)")
    conn.execute("INSERT INTO knowledge_memory VALUES (1,?,?,?,?,?)", ("What is water?", "Water is a compound made of hydrogen and oxygen.", "book", .96, "{}"))
    conn.commit()
    original = conn.execute("SELECT * FROM knowledge_memory").fetchone()
    conn.close()

    brain = FederatedBrain(tmp_path / "knowledge")
    counts = brain.import_legacy_db(legacy, domain="science")
    assert counts["qna"] == 1
    migrated = brain.ask("What is water?", domains=["science"], top_k=2)
    assert migrated["status"] == "SUCCESS"

    conn = sqlite3.connect(legacy)
    assert conn.execute("SELECT * FROM knowledge_memory").fetchone() == original
    conn.close()
    brain.close()
