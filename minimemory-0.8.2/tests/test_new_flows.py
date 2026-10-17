from minimemory import LocalBrain, KnowledgeGate, ResourceRouter


def test_claim_gate_threshold_duplicate_and_polarity_conflict(tmp_path):
    gate = KnowledgeGate(str(tmp_path / 'claims.db'), admission_threshold=0.6)
    low = gate.admit('A low value candidate.', admission_score=0.2)
    assert low['stored'] is False
    a = gate.admit('Water can extinguish some fires.', source='book', admission_score=.9,
                   confidence=.8, evidence_polarity='supports', context={'condition':'some fires'})
    assert a['stored'] is True
    duplicate = gate.admit('Water can extinguish some fires.', source='book', admission_score=.9)
    assert duplicate['duplicate'] is True
    b = gate.admit('Water cannot extinguish some fires.', source='other', admission_score=.9,
                   evidence_polarity='opposes')
    assert b['status'] == 'contested'
    assert a['id'] in b['related_claims']
    assert gate.search('water fires')
    gate.close()


def test_resource_router_online_fallback_to_local():
    calls=[]
    def online(*args, **kwargs):
        calls.append('online'); raise TimeoutError()
    def local(*args, **kwargs):
        calls.append('local'); return 'local-result'
    router=ResourceRouter(online_available=lambda: True, online_provider=online, local_provider=local)
    result=router.run('hello')
    assert result['ok'] and result['mode']=='local' and result['result']=='local-result'
    assert calls == ['online','local']


def test_resource_router_offline_skips_online():
    calls=[]
    router=ResourceRouter(online_available=lambda: False,
                          online_provider=lambda: calls.append('online'),
                          local_provider=lambda: 'offline-result')
    result=router.run()
    assert result['mode']=='local' and result['result']=='offline-result'
    assert calls == []


def test_localbrain_claim_api_does_not_break_memory(tmp_path):
    brain=LocalBrain(str(tmp_path/'brain.db'))
    brain.remember('Python is a programming language.', source='test')
    result=brain.admit_claim('Python is a programming language.', source='manual', admission_score=.9, confidence=.9)
    assert result['stored']
    assert brain.search_claims('Python programming')
    assert brain.recall('Python programming')
    brain.close()


def test_federated_claim_ledger_integrates_without_replacing_domain_memory(tmp_path):
    from minimemory import FederatedBrain
    brain = FederatedBrain(tmp_path / 'federated', auto_domains=False)
    brain.register_domain('science', keywords=['water', 'fire'])
    brain.remember('Water can extinguish some ordinary fires.', domain='science', source='book', confidence=.8)
    admitted = brain.admit_claim('Water can extinguish some ordinary fires.', domain='science',
                                 source='book', admission_score=.9, confidence=.8,
                                 evidence_polarity='supports', context={'page': 12})
    assert admitted['stored']
    claims = brain.search_claims('water ordinary fires', domain='science')
    assert claims and claims[0]['context']['page'] == 12
    result = brain.ask('water fires', domains=['science'])
    assert result['status'] == 'SUCCESS'
    assert result['claim_evidence']
    assert brain.health()['ok']
    brain.close()


def test_optional_semantic_conflict_detector_is_used_and_failure_falls_back(tmp_path):
    gate = KnowledgeGate(str(tmp_path / 'semantic.db'), conflict_detector=lambda a, b: 'conflict-tag' in a and 'conflict-tag' in b)
    first = gate.admit('conflict-tag statement alpha', source='a', admission_score=.9)
    second = gate.admit('conflict-tag statement beta', source='b', admission_score=.9)
    assert first['stored'] and second['status'] == 'contested'
    gate.close()


def test_internet_probe_has_configurable_urls_and_timeout(monkeypatch):
    from minimemory.resource_router import internet_available
    class Response:
        status = 204
        def __enter__(self): return self
        def __exit__(self, *args): pass
    calls=[]
    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return Response()
    monkeypatch.setattr('minimemory.resource_router.urlopen', fake_urlopen)
    assert internet_available(probe_urls=('https://probe.invalid/test',), timeout=.25)
    assert calls == [('https://probe.invalid/test', .25)]
