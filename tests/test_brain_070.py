import tempfile
from minimemory import AssociativeMemory, LocalBrain, ProjectPlanner


def test_associative_recall_graph_and_programming():
    with tempfile.NamedTemporaryFile(suffix='.db') as f:
        m = AssociativeMemory(f.name)
        e1 = m.upsert_entity('place:mountain:1', 'Blue Mountain', entity_type='place', aliases=['mountain'])
        m.remember('A previous trip included a snowy mountain, pine forest and lake.', memory_type='episodic', confidence=.9)
        mid = m.remember('The trip location was northern region.', memory_type='semantic', confidence=.8)
        m.associate(mid, 'place:mountain:1')
        m.add_programming_term('HTML', category='language', meaning='markup language', simple_meaning='HTML structures web pages')
        results = m.recall('mountain trip')
        assert results
        assert m.search_programming('HTML web')
        m.close()


def test_embedding_association_and_bundle():
    def embed(x):
        return [1.0, 0.0] if 'mountain' in str(x).lower() else [0.0, 1.0]
    with tempfile.NamedTemporaryFile(suffix='.db') as f:
        m = AssociativeMemory(f.name, embedding_fn=embed)
        m.remember('snowy mountain lake', memory_type='episodic')
        result = m.recall('mountain')
        assert result and result[0]['embedding_score'] > 0.9
        bundle = m.memory_bundle('mountain')
        assert 'memories' in bundle and 'entities' in bundle
        m.close()


def test_seed_and_planner():
    with tempfile.NamedTemporaryFile(suffix='.db') as f:
        brain = LocalBrain(f.name)
        assert brain.seed_programming_knowledge() > 50
        plan = ProjectPlanner().analyze('build a calculator', platform='android', language='kotlin')
        assert plan['status'] == 'ready'
        brain.close()
