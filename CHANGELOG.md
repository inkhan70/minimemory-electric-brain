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
