import json
from minimemory.cli import main

def test_cli_learn_ask_search_stats_health_forget(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "learn", "What is Python?", "A language."]) == 0
    out = capsys.readouterr().out
    assert "Learned" in out
    assert main(["--db", db, "ask", "What is Python?", "--threshold", "0.1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "SUCCESS"
    assert main(["--db", db, "search", "Python", "--top-k", "1"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["question"] == "What is Python?"
    assert main(["--db", db, "stats"]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 1
    assert main(["--db", db, "health"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert main(["--db", db, "forget", "1"]) == 0
    assert "Deleted" in capsys.readouterr().out
