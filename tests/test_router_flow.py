from minimemory import LocalBrain, QueryRoute, classify_query


def test_query_classifier_distinguishes_text_and_code():
    assert classify_query("Where is Denver?")["kind"] == "text"
    assert classify_query("```python\nprint('hello')\n```")["kind"] == "code"


def test_full_routing_flow(tmp_path):
    with LocalBrain(tmp_path / "brain.db", behavior_tracking=True) as brain:
        brain.memory.learn("Where is Denver?", "Denver is a city in Colorado.", confidence=1.0)
        mid = brain.remember("The user is building an Android app with Termux.", confidence=0.95)
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

        status = brain.behavior_status()
        assert status["stored_events"] >= 3
        assert mid > 0


def test_sensitive_behavior_is_not_stored(tmp_path):
    with LocalBrain(tmp_path / "brain.db", behavior_tracking=True) as brain:
        result = brain.ask("my password is secret")
        assert result["behavior"]["stored"] is False
        assert brain.behavior_status()["stored_events"] == 0
