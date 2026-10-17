# Changelog

## 0.7.0
- Added `AssociativeMemory` with semantic and episodic memory types.
- Added optional embedding-aware multimodal-ready recall.
- Added entities, aliases, relationships and memory links.
- Added conservative episodic-to-semantic consolidation.
- Added programming terminology storage and seed dictionary.
- Added reusable code-component storage with dependency/provenance/license metadata.
- Added deterministic project planning helper.
- Added `LocalBrain` facade for local applications.
- Added health/stats and expanded tests.

## 0.8.0 — Federated Domain Memory

- Added `FederatedBrain` for domain-aware routing across independent SQLite knowledge stores.
- Added `DomainStore` with append-only knowledge ingestion and deterministic lexical retrieval.
- Added SQLite FTS5 candidate indexing when available, with a standard-library fallback.
- Added `frame.db` for domain registry, routing metadata, answer events, and audit events.
- Added `AnswerFramer` for deterministic extractive multi-domain answer composition.
- Added provenance to every federated evidence result.
- Added protection against outgoing-answer contamination: framed answers are logged separately and are not inserted into trusted domain stores.
- Added legacy SQLite import without modifying the source database.
- Added federated CLI commands for remember, Q&A, ask, health, and legacy import.
- Added cross-domain, immutability, health, CLI, and migration tests.

## 0.8.2 — Federated evidence integration
- Integrated the claim/evidence ledger into `FederatedBrain` using separate tables in `frame.db`; existing domain memory APIs remain intact.
- Added `admit_claim()` and `search_claims()` to federated memory, domain context tagging, and claim evidence/warning metadata in `ask()` results.
- Added optional host-supplied conflict detection; the lexical heuristic remains the fallback and neither method proves truth.
- Added `internet_available()` HTTP reachability probes with configurable endpoints and bounded per-probe timeout; this checks internet reachability, not the health of any specific provider.
- Expanded regression tests for federated integration, conflict detector hooks, and connectivity probes.

## Unreleased — review fixes
- Fixed a file-descriptor leak: every `MemoryQA` / `SQLiteBackend` call opened a SQLite connection and never closed it (`with conn` only commits).
- Federated search: the FTS index now covers question/title/keyword metadata, so Q&A entries are found by their question text. Existing databases migrate automatically (old index and triggers are replaced and rebuilt on open).
- Retrieval no longer returns unrelated items: confidence/importance used to give every memory a baseline score above `min_score` (`DomainStore.search`, `AssociativeMemory.recall`).
- `FederatedBrain`: mixed-case domain names no longer raise `KeyError`; routing probes no longer inflate `access_count`.
- `DomainStore.search` committed nothing after updating `access_count`, leaving a transaction (and write lock) open.
- `import_legacy_db` now opens the source read-only and raises `FileNotFoundError` instead of silently creating an empty database.
- `AssociativeMemory.recall` embeds the query once (was once per stored memory) and loads links in one query; creates missing parent directories; reads/writes are lock-protected; `associate()` validates the memory id.
- `KnowledgeGate`: an explicit `rejected`/`superseded` status is no longer overwritten by conflict detection; `close()` is idempotent.
- `MemoryQA`: semantic-backend failures fall back to lexical search instead of returning empty results or failing to open.
- `TextPipeline`: extra passes now run when spelling changed the text.
- `internet_available` only accepts http(s) probe URLs.
- CLI: export/push/pull close their database, `--domains` is trimmed, and SQLite/Key/Type errors print a message instead of a traceback.
- Packaging: `sentence-transformers>=2.3.0` (needed for `local_files_only`), `wheel` added to build requirements, `run.sh` message corrected.
