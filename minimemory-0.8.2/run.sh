#!/usr/bin/env bash
set -euo pipefail

# Offline-friendly production smoke test / install helper.
python -m pytest -q
python -m pip install --no-build-isolation .
python -m minimemory.cli --help >/dev/null
printf '%s\n' 'minimemory 0.8.2 smoke test passed.'
