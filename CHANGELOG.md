# Changelog

## 0.8.0 - Electric Brain

- Added privacy-aware behavior classification and a privacy firewall.
- Behavior tracking is opt-in and sensitive behavior is not persisted or exported.
- Added deterministic memory evolution and bounded episodic-to-semantic consolidation.
- Added electric-brain `pulse()` diagnostics.
- Added local network/storage budget primitives for optional research workflows.
- Added optional web research/evidence pipeline with explicit approval before storage.
- Added portable `.mmpack` brain exchange with checksums and local embedding regeneration.
- Behavior history is excluded from portable knowledge packs.
- Added CLI commands under `minimemory brain` for observe, evolve, pulse, export, import, and pack inspection.
- Added production tests for privacy, evolution, research evidence, knowledge transfer, tamper detection, and CLI smoke flows.
- Updated package version to 0.8.0.
- Added deterministic unified query routing for Q&A, associative memory, programming knowledge, and reusable code components.
- Added code-block/text classification and `LocalBrain.route()` / `LocalBrain.ask()` orchestration with optional AI fallback.
- Added routing and privacy regression tests; full suite passes with 35 tests.

## 0.8.0-flow-tested additions

- Added controlled Markdown documentation updates with explicit approval.
- Added `documentation.md` architecture and evolution documentation.
- Added portable `tests/test_complete.py` for Android/Termux and GitHub.
- Added Android and GitHub testing guides.
- Documentation updates are section-scoped and do not occur implicitly during memory evolution.
