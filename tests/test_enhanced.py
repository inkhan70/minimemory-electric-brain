from minimemory import MemoryQA, TextPipeline, RefinementLoop


def test_pipeline_and_refinement():
    p = TextPipeline(alphabet_only=True)
    out = p.process("  Hello, world! ★ ")
    assert out["corrected"] == "Hello, world!"
    assert out["changed"]

    loop = RefinementLoop(max_iterations=3)
    state = {"n": 0}
    def verify(text):
        state["n"] += 1
        return {"passed": text == "good"}
    def revise(text, feedback):
        return "good"
    out = loop.run("bad", verify, revise)
    assert out["passed"] is True
    assert out["iterations"] == 2


def test_entity_graph():
    with MemoryQA(":memory:") as m:
        denver = m.upsert_entity("place:denver:co", "Denver", entity_type="city", aliases=["Denver, Colorado"])
        colorado = m.upsert_entity("region:colorado", "Colorado", entity_type="region")
        m.add_relation(denver, "located_in", "region:colorado", source="book.pdf", confidence=.99)
        found = m.find_entities("Denver, Colorado")
        assert found and found[0]["entity_key"] == "place:denver:co"
        g = m.graph("place:denver:co")
        assert g["relations"][0]["object"] == "region:colorado"
