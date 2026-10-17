# minimemory 0.8.2 — implementation notes

## Implemented in this patch

1. **Claim admission gate** (`minimemory/knowledge_gate.py`): separate SQLite claim ledger, configurable admission threshold, source/context/provenance fields, polarity, optional confidence, duplicate detection, searchable claims, and append-preserving potential-conflict links.
2. **Online/local provider routing** (`minimemory/resource_router.py`): injectable connectivity check and provider callables, online-first when connectivity reports available, local fallback on provider exceptions, and local-only behavior when offline.
3. **LocalBrain integration** (`brain.py`): `admit_claim` and `search_claims`; existing remember/recall/bundle APIs remain in place.
4. **Tests** (`tests/test_new_flows.py`): threshold, duplicate, conflict flagging, online-to-local fallback, offline routing, and backwards-compatible LocalBrain memory operations.

## Deliberate boundaries

- Connectivity is supplied by the host app; this package does not guess internet status from Wi-Fi alone.
- Provider functions must implement actual network/model calls and should set their own bounded timeouts.
- Lexical conflict detection is a review flag, not an NLI model and not a truth detector. It can miss contradictions and produce false positives.
- Scores are not truth probabilities. A claim below the admission threshold is not stored by default; callers may adjust the threshold or use `force_store=True` for review/archive workflows.
- Existing domain/federated memory is not automatically rewritten by these additions.
- Speech-to-text, translation, and text-to-speech engines are not bundled in this patch; `ResourceRouter` can route those operations once the application provides compatible providers.

## 0.8.2 implementation addendum
- `FederatedBrain` now owns a `KnowledgeGate` in separate tables inside `frame.db`, with `admit_claim()` and `search_claims()` APIs. Claim context includes a domain tag, and `ask()` returns related claim evidence separately from extracted domain-memory answer text.
- A host application may inject `conflict_detector(claim_a, claim_b) -> bool`. This is a plugin hook, not a bundled semantic model. If it raises, the lexical fallback is used.
- `internet_available()` probes actual HTTP reachability with bounded per-URL timeouts. The caller must still handle provider-level timeouts and API failures.
- Claim flags and scores are review signals, not proof of truth. The default answer framer remains extractive and does not automatically blend contested claims into the answer.
