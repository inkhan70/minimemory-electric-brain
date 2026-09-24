# Test Report

## Current source

- Package: `minimemory`
- Version: `0.8.0`
- Test suite: `48 passed`
- Complete portable suite: `13 passed`
- Python compilation: passed
- Editable installation with `--no-build-isolation`: passed
- Wheel build with `--no-build-isolation --no-deps`: passed

## Portable complete test

`tests/test_complete.py` covers the core Q&A API, associative memory, entities and relations, query classification, routing, programming/code retrieval, AI fallback, behavior evolution/privacy, knowledge-pack round trips, documentation approval/update behavior, backup/health, invalid-input handling, and public exports.

## Environment note

A normal isolated build may try to download build requirements. The project itself remains standard Python/SQLite based. In offline environments where build requirements are already installed, `--no-build-isolation` can be used.

## Android / Termux

Run:

```bash
python -m pip install pytest
python -m pytest -q tests/test_complete.py
python -m pytest -q
```

## GitHub

The repository includes `.github/workflows/tests.yml` to run the portable test and the full suite on Python 3.9, 3.11, and 3.13.
