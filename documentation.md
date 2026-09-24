# minimemory Documentation

This document describes the local memory brain and its controlled evolution model.

## Architecture

`minimemory` is an offline-first SQLite memory layer. It does not train neural-model weights. It stores, indexes, retrieves, consolidates, and exchanges knowledge for an AI application.

### Query flow

```text
User input
   |
   v
Classifier
   |---- ordinary text ------> Q&A / associative memory
   |---- programming --------> programming knowledge / code components
   |---- code or mixed ------> code components first
   |
   v
Local evidence ranking
   |
   +---- strong result ------> return local answer
   |
   +---- insufficient ------> optional AI fallback
                              |
                              v
                       optional approved learning
```

The router is deterministic and explainable. It does not call an AI model merely to decide where a query belongs.

## Associative evolution

The evolution engine can learn aggregate, non-sensitive usage signals when behavior tracking is explicitly enabled. Repeated topics gain a bounded strength signal, which can influence associative recall.

Evolution is deliberately conservative:

- it does not invent facts;
- sensitive behavior is not persisted by the built-in privacy policy;
- behavior history is not exported in knowledge packs;
- consolidation uses existing stored content;
- AI-generated knowledge requires the application's approval policy.

## Controlled documentation evolution

A mature local brain can prepare documentation updates from verified knowledge or test results. `minimemory` intentionally does **not** modify repository files automatically during ordinary recall/evolution.

Use the explicit API when an application has reviewed and approved the content:

```python
from minimemory import update_documentation

update_documentation(
    "documentation.md",
    "project-notes",
    "The routing layer now checks local programming knowledge before AI fallback.",
    approved=True,
    heading="Project notes",
)
```

The update is section-scoped using stable markers, so future approved updates replace the same section rather than duplicating it.

## Testing

The repository contains both focused regression tests and a portable end-to-end test file:

```bash
python -m pytest -q
```

The complete test file is `tests/test_complete.py`. It is designed to run in standard Python/Termux without network access or optional ML packages.

## Android / Termux

Minimal setup:

```bash
pkg update
pkg install python
cd ~/minimemory-0.8.0-flow
python -m pip install -e .
python -m pytest -q
```

For Android, the minimal standard-library path is recommended first. Semantic-search and AI-server extras can be installed separately when the device has the required resources.

## GitHub

Commit source, tests, documentation, and configuration. Do not commit local SQLite databases, API keys, virtual environments, caches, or private knowledge packs.

The included `.gitignore` excludes common local artifacts including `*.db`, `*.sqlite`, `*.sqlite3`, `__pycache__/`, `*.py[cod]`, `.venv/`, `env/`, `venv/`, `.DS_Store`, and build/test caches.
