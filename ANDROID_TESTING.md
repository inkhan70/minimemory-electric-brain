# Android / Termux Testing

`minimemory` is designed to have a standard-library baseline so the core test suite can run on Android/Termux without network access.

## Run the complete test file

From the project directory:

```bash
python -m pytest -q tests/test_complete.py
```

Then run the entire suite:

```bash
python -m pytest -q
```

## If pytest is not installed

```bash
python -m pip install pytest
python -m pytest -q tests/test_complete.py
```

The complete test intentionally avoids requiring Sentence Transformers, Torch, NumPy, Hugging Face, Ollama, or an internet connection.

## Expected result

The exact number can change as the project grows. A successful run ends with `passed` and no failures/errors.
