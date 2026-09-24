# GitHub Testing

Run the same portable tests used for Android/Termux from a clean checkout.

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest -q tests/test_complete.py
python -m pytest -q
```

For CI, the complete test file is safe to run without external AI credentials or network services. Optional semantic/research/AI integrations should be tested in separate jobs when their dependencies and credentials are intentionally configured.
