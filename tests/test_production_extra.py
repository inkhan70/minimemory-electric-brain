import json
import sqlite3
from pathlib import Path
import pytest
from minimemory import MemoryQA, KnowledgePack, __version__


def test_update_get_delete_stats_health_and_backup(tmp_path):
    db = tmp_path / "memory.db"
    ai = MemoryQA(db, semantic=False)
    ai.learn("Q", "A", source="user", confidence=0.8, metadata={"tag": "x"})
    item = ai.get(1)
    assert item and item["answer"] == "A" and item["metadata"] == {"tag": "x"}
    ai.learn("Q", "B", source="ai", confidence=0.9)
    assert ai.get(1)["answer"] == "B"
    assert ai.search("Q", source="ai")[0]["source"] == "ai"
    assert ai.stats()["count"] == 1
    assert ai.health()["ok"] is True
    backup = tmp_path / "backup.db"
    assert Path(ai.backup(str(backup))).is_file()
    assert MemoryQA(backup, semantic=False).get(1)["answer"] == "B"
    assert ai.delete(1) is True
    assert ai.count() == 0
    assert ai.delete(1) is False


def test_chat_dict_response_and_policy(tmp_path):
    ai = MemoryQA(tmp_path / "m.db", semantic=False, min_auto_learn_confidence=0.8)
    def model(question, context):
        return {"answer": "A", "confidence": 0.95, "metadata": {"model": "local"}}
    result = ai.chat("Q", model, threshold=1.0, auto_learn=True)
    assert result["learned"] is True
    assert ai.ask("Q", threshold=0.1)["answer"] == "A"


def test_unapproved_ai_is_not_saved(tmp_path):
    ai = MemoryQA(tmp_path / "m.db", semantic=False, min_auto_learn_confidence=0.9)
    result = ai.chat("Q", lambda q: "A", threshold=1.0, auto_learn=True, approve=False)
    assert result["learned"] is False
    assert ai.count() == 0


def test_pack_checksum_and_corruption(tmp_path):
    pack = KnowledgePack([("q", "a")])
    out = Path(pack.save(tmp_path / "pack"))
    loaded = KnowledgePack.from_file(str(out))
    assert loaded.checksum() == pack.checksum()
    data = json.loads((out / "minimemory.json").read_text())
    data["pairs"][0]["answer"] = "tampered"
    (out / "minimemory.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        KnowledgePack.from_file(str(out))


def test_import_old_pack_without_checksum(tmp_path):
    data = {"format": "minimemory-pack", "version": "1.1", "pairs": [{"question": "q", "answer": "a"}]}
    p = tmp_path / "old.json"
    p.write_text(json.dumps(data))
    assert KnowledgePack.from_file(str(p)).pairs == [("q", "a")]


def test_in_memory_db():
    ai = MemoryQA(":memory:", semantic=False)
    ai.learn("q", "a")
    assert ai.ask("q", threshold=0.1)["answer"] == "a"
    ai.close()
    with pytest.raises(RuntimeError):
        ai.count()


def test_generator_object_interfaces(tmp_path):
    ai = MemoryQA(tmp_path / "m.db", semantic=False)
    class G:
        def generate(self, question):
            return "generated"
    assert ai.chat("q", G(), threshold=1.0, auto_learn=False)["answer"] == "generated"


def test_version():
    assert __version__ == "0.8.0"
