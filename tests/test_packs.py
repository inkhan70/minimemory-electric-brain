import json
from pathlib import Path
import pytest
from minimemory.core import MemoryQA
from minimemory.packs import KnowledgePack


@pytest.fixture
def ai_instance(tmp_path):
    db_file = str(tmp_path / "test_memory.db")
    ai = MemoryQA(db_path=db_file)
    ai.learn("What is Python?", "Python is an interpreted programming language.")
    ai.learn("Who created Python?", "Guido van Rossum.")
    return ai


def test_pack_creation_and_export(ai_instance, tmp_path):
    pack = KnowledgePack.from_memory(ai_instance, description="Python Fundamentals", author="Dev")
    export_dir = tmp_path / "python_pack"
    out_path = pack.save(str(export_dir))

    assert Path(out_path).exists()
    assert (export_dir / "minimemory.json").exists()
    assert (export_dir / "README.md").exists()


def test_pack_file_load_and_apply(ai_instance, tmp_path):
    export_dir = str(tmp_path / "python_pack")
    ai_instance.save_pack(export_dir, description="Test Export")

    # Load into fresh instance
    new_db = str(tmp_path / "new_memory.db")
    new_ai = MemoryQA(db_path=new_db)
    count = new_ai.load_pack(export_dir)

    assert count == 2
    res = new_ai.ask("Who created Python?")
    assert res["status"] == "SUCCESS"
    assert "Guido van Rossum" in res["answer"]


def test_pack_merging(tmp_path):
    pack1 = KnowledgePack(pairs=[("q1", "a1"), ("q2", "a2")], description="Pack 1")
    pack2 = KnowledgePack(pairs=[("q2", "updated_a2"), ("q3", "a3")], description="Pack 2")

    merged = pack1.merge_with(pack2)
    assert len(merged.pairs) == 3
    
    merged_dict = dict(merged.pairs)
    assert merged_dict["q2"] == "updated_a2"  # Verifies update on duplicate keys
