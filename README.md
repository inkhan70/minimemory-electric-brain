# minimemory

**minimemory** is a small, local-first memory brain for Python AI applications. It stores knowledge in SQLite and provides deterministic retrieval, associative memory, lightweight graph relationships, optional semantic search, text refinement, project planning, and portable knowledge packs.

It is designed to work without a cloud AI service. You can connect it to any local or remote AI model through your own application code, while keeping the memory layer independent from the model.

> **License:** Apache License 2.0

## What it does

- **Local Q&A memory:** learn, ask, search, update, delete, and inspect memories.
- **SQLite persistence:** normal database files survive restarts and work well on desktop, servers, and Termux/Android.
- **Offline fallback retrieval:** deterministic token/cosine-style lexical similarity works without third-party ML packages.
- **Optional semantic search:** use `sentence-transformers`, `torch`, and `numpy` when you want embedding-based retrieval.
- **Associative memory:** store episodic/semantic memories, entities, aliases, links, and relations.
- **Local brain facade:** `LocalBrain` combines Q&A and associative memory behind one API.
- **Knowledge packs:** export/import JSON packs, merge packs, and optionally publish/load them through Hugging Face Hub.
- **Text pipeline:** normalization, optional alphabet filtering, spelling/grammar callbacks, grounding callbacks, and bounded refinement loops.
- **AI-assisted learning:** `learn_from_ai()` can require approval or a confidence threshold before storing an AI answer.
- **Project planner:** create deterministic project specifications and dependency-aware task lists.
- **Backups and health checks:** SQLite backup support and integrity checks.
- **Image concepts:** store image-derived descriptions/embeddings as metadata without putting binary image files into SQLite.

## Installation

### Minimal, standard-library mode

```bash
python -m pip install .
```

This mode uses SQLite and the built-in deterministic retrieval path.

### Semantic search

```bash
python -m pip install '.[semantic]'
```

The semantic extra installs:

```text
sentence-transformers>=2.2.0
torch>=2.0.0
numpy
```

Semantic models may need to be downloaded once before fully offline use. With `local_files_only=True` (the default), minimemory will not try to download a missing model and will fall back to local retrieval.

### Hugging Face knowledge packs

```bash
python -m pip install '.[hub]'
```

### Everything

```bash
python -m pip install '.[all]'
```

## Quick start

```python
from minimemory import MemoryQA

with MemoryQA("memory.db") as brain:
    brain.learn("Where is Denver?", "Denver is a city in Colorado, USA.",
                source="user", confidence=0.95)

    result = brain.ask("Where is Denver?")
    print(result)

    print(brain.stats())
    print(brain.health())
```

The memory layer does **not** train model weights. It stores and retrieves knowledge that your AI application can provide or approve.

## Connect an AI model

minimemory can connect to an AI model server without making the memory database dependent on that provider. The included adapters use Python's standard library HTTP client.

### Option A: Ollama

Run an Ollama server on the same computer (or another reachable machine), then test the connection:

```bash
minimemory ai test --provider ollama --model llama3.2:3b
```

Ask a model while giving it retrieved local memories:

```bash
minimemory --db memory.db ai ask "What do you know about my project?" --provider ollama --model llama3.2:3b
```

The AI answer is **not automatically treated as trusted knowledge**. Use `--approve` when you explicitly want that answer eligible for learning:

```bash
minimemory --db memory.db ai ask "What do you know about my project?" --provider ollama --model llama3.2:3b --approve
```

### Option B: OpenAI-compatible servers

The same adapter works with services or local servers that expose a compatible `/v1/chat/completions` endpoint:

```bash
export OPENAI_API_KEY="your-key"
minimemory ai test --provider openai-compatible --model YOUR_MODEL --base-url https://api.openai.com/v1
```

For another compatible server, change `--base-url` and `--model`. Keep API keys in environment variables or another secret manager; never commit them to GitHub.

### GitHub tester privacy

Each person who clones this repository should use their **own database file and their own AI credentials**. minimemory does not require a shared account or shared memory database. The default `memory.db` is local to the tester's machine and is ignored by Git.

- A tester's memories stay in their local SQLite database unless they deliberately export/share a knowledge pack.
- API keys are supplied by the tester and should never be placed in source code, README files, or committed configuration.
- Do not commit `*.db`, `*.sqlite`, or `*.sqlite3` files; `.gitignore` already excludes them.
- Do not put private conversations or personal data into a public knowledge pack.
- GitHub users can test the project with their own model endpoint and their own local memory.

## Local AI connection

minimemory does not lock you to a specific model provider. A simple application flow is:

```python
from minimemory import MemoryQA

brain = MemoryQA("memory.db")

question = "What is my project database?"
context = brain.search(question, top_k=5)

# Send `context` to your local AI/LLM in your own code.
answer = local_model.generate(question, context=context)

# Store only when your policy allows it.
brain.learn_from_ai(
    question,
    answer,
    confidence=0.92,
    approved=True,
    source="local-ai",
)

brain.close()
```

Your local model can be an API server, an on-device model, or another Python callable. minimemory remains the storage/retrieval layer.

## Associative memory and local brain

```python
from minimemory import LocalBrain

with LocalBrain("brain.db") as brain:
    colorado = brain.associative.upsert_entity(
        "place:denver-co",
        "Denver",
        entity_type="city",
        aliases=["Denver, Colorado"],
    )
    country = brain.associative.upsert_entity(
        "country:usa",
        "United States",
        entity_type="country",
        aliases=["USA", "United States of America"],
    )

    brain.associative.add_relation(
        "place:denver-co",
        "located_in",
        "country:usa",
        confidence=0.95,
    )

    brain.remember(
        "Denver is the capital of Colorado.",
        memory_type="semantic",
        source="knowledge-base",
        confidence=0.95,
    )

    print(brain.recall("Where is Denver?"))
    print(brain.bundle("Denver"))
    print(brain.health())
```

Entities and aliases let an application keep multiple similarly named places or concepts separate by using stable keys and metadata.

## Knowledge packs

Export:

```bash
minimemory --db memory.db export ./pack --description "My local knowledge"
```

Import:

```bash
minimemory --db memory.db pull ./pack
```

Merge local packs:

```bash
minimemory merge ./pack_a ./pack_b --out ./merged_pack
```

Optional Hub operations:

```bash
minimemory --db memory.db push username/my-memory-pack
minimemory --db memory.db pull username/my-memory-pack
minimemory browse --search minimemory
```

Authentication can be supplied with `--token` or the `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` environment variable.

## CLI

```bash
minimemory --db memory.db learn "What is Python?" "Python is a programming language."
minimemory --db memory.db ask "What is Python?"
minimemory --db memory.db search "Python" --top-k 5
minimemory --db memory.db stats
minimemory --db memory.db health
minimemory --db memory.db forget 1
```

Run:

```bash
minimemory --help
```

for the complete command list.

## Text correction and refinement

```python
from minimemory import TextPipeline, RefinementLoop

pipeline = TextPipeline(alphabet_only=False)
result = pipeline.process("  Hello   world !  ")
print(result["corrected"])
```

Spelling, grammar, and grounding are callback-based so applications can plug in their own tools without making them required dependencies.

For answer verification, `RefinementLoop` provides a bounded loop with a maximum of 10 iterations, preventing an accidental infinite correction loop.

## Project planning

```python
from minimemory import ProjectPlanner

plan = ProjectPlanner().analyze(
    "Build a local AI assistant",
    platform="Android",
    backend="Python",
    database="SQLite",
    language="Python",
)

print(plan)
```

## Backup

```python
from minimemory import MemoryQA

with MemoryQA("memory.db") as brain:
    brain.backup("backups/memory-backup.db")
```

## Android / Termux

minimemory is pure Python at its core and uses SQLite from Python's standard library. For a lightweight Termux installation:

```bash
cd ~/storage/download/minimemory-0.7.0-associative/mm070
python -m pip install .
pytest -q
```

If you want semantic search on Android, install the semantic extra only if your device/Python environment supports the required PyTorch and Sentence Transformers wheels.

## Repository layout

```text
mm070/
├── minimemory/
│   ├── core.py
│   ├── associative.py
│   ├── brain.py
│   ├── pipeline.py
│   ├── planner.py
│   ├── packs.py
│   ├── programming_knowledge.py
│   ├── cli.py
│   └── backends/
├── tests/
├── README.md
├── LICENSE
├── CHANGELOG.md
├── pyproject.toml
├── .gitignore
└── run.sh
```

Generated build directories, wheels, SQLite databases, caches, and virtual environments should not be committed; `.gitignore` covers these artifacts.

## Testing

From the repository root:

```bash
python -m pytest -q
```

The test suite covers core memory behavior, production behavior, associative memory, the brain facade, text pipeline, packs, and CLI helpers.

## Design notes

- Memory storage and AI generation are deliberately separated.
- Retrieval has a deterministic fallback, so basic use does not require ML packages.
- Semantic retrieval is optional and loaded lazily.
- Knowledge packs use checksums to detect accidental corruption.
- SQLite is the persistent source of truth; in-memory caches are rebuilt from it.
- Binary assets such as images are not copied into the database by `remember_image()`; store a path, URI, hash, or object ID in metadata instead.
- Automatic learning should be governed by the host application's approval/confidence policy.

## License

Copyright 2026 minimemory contributors.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.

## Electric brain: privacy-aware evolution

`LocalBrain` is the high-level orchestration layer for the evolving local brain. It combines Q&A memory, associative memory, bounded consolidation, optional behavior evolution, resource budgets, web-research evidence, and portable `.mmpack` knowledge exchange.

### Privacy firewall

Behavior tracking is **off by default**. When enabled, minimemory records only aggregate non-sensitive signals such as a category/topic, query hash, count, and recency. Exact query text is not stored by the behavior engine. Sensitive observations are classified transiently and are never persisted, evolved, recommended from, or exported by the built-in policy.

```python
from minimemory import LocalBrain

with LocalBrain("brain.db", behavior_tracking=True) as brain:
    brain.observe("Python FastAPI SQLite")
    print(brain.behavior_status())
    print(brain.evolve())
```

The classifier has categories including programming, technology, science, writing, productivity, shopping, travel, entertainment, general research, temporary interest, sensitive, and unknown. Applications can add their own policy layer before enabling behavior storage.

### Electric-brain pulse

```python
with LocalBrain("brain.db", behavior_tracking=True) as brain:
    print(brain.pulse())
```

A pulse reports database health, memory statistics, evolution topics, and storage status. `evolve()` performs conservative consolidation: frequently recalled episodic memories can be copied into semantic memory using their original content and provenance; it does not invent facts.

### Web research and evidence

Web research is optional:

```bash
python -m pip install '.[research]'
```

Research produces **evidence, not automatic truth**. A result can be single-source or supported by multiple domains, but it is not marked verified merely because an AI model says so. Store researched knowledge only after explicit approval or an application-supplied validator.

```python
from minimemory import LocalBrain, WebResearcher

with LocalBrain("brain.db") as brain:
    result = brain.research("What is Python?", researcher=WebResearcher(max_results=3))
    accepted = brain.store_research(
        "What is Python?",
        "Python is a programming language.",
        result,
        approved=True,
    )
```

### Portable `.mmpack` brain exchange

Do not copy `memory.db` between unrelated installations as the primary exchange format. Use the portable knowledge pack:

```bash
minimemory --db brain.db brain export ./knowledge.mmpack
minimemory --db other.db brain inspect-pack ./knowledge.mmpack
minimemory --db other.db brain import ./knowledge.mmpack
```

The pack preserves Q&A knowledge, associative memories, entities, aliases, relationships, provenance metadata, and confidence. Machine-specific embeddings are excluded so the receiving installation can regenerate them locally. Behavior history is excluded. Checksums are verified before import.

### Resource budgets

Applications can set local network/storage budgets for optional research:

```python
from minimemory import LocalBrain

brain = LocalBrain("brain.db", network_budget_mb=30, storage_budget_mb=100)
```

These are application-level budgets, not Android permission dialogs. Android application permissions remain controlled by the Android application/host environment.

## GitHub tester privacy notice

If you clone this project, use your own database and your own AI credentials. Never commit API keys, private conversations, or private memory databases. `.gitignore` excludes SQLite database files, Python caches, virtual environments, and common local artifacts. Knowledge packs are deliberate exports: review them before publishing.

## Unified local query flow

`LocalBrain.ask()` now provides a deterministic routing layer before any AI fallback:

1. **Classify input** as ordinary text, code, or mixed text/code.
2. **Check local Q&A memory** for known question/answer pairs.
3. **Check associative memory** for contextual/episodic/entity-linked knowledge.
4. **Check programming knowledge and code components** for programming terms and reusable code.
5. **Return the strongest local result first**; an optional AI generator is only used as a fallback when local knowledge is insufficient.
6. **Observe behavior only when enabled**. Sensitive observations remain non-persistent.

Use `LocalBrain.route(query)` when an application wants the routing decision and all candidate local matches without generating an answer.

```python
from minimemory import LocalBrain

with LocalBrain("brain.db", behavior_tracking=True) as brain:
    result = brain.ask("Where is Denver?")
    print(result["route"], result["answer"])

    result = brain.ask("```python\nprint('hello')\n```")
    print(result["route"], result["answer"])
```

The code/programming knowledge currently lives in the same SQLite database as the associative brain by default, so the whole local knowledge base remains portable through the existing `.mmpack` workflow.

## Controlled documentation evolution

The brain can prepare updates to a Markdown documentation file, but it does not silently rewrite project files during normal learning or evolution. An application must explicitly approve a documentation update:

```python
from minimemory import update_documentation

update_documentation(
    "documentation.md",
    "routing-notes",
    "The local router now checks code components before AI fallback.",
    approved=True,
    heading="Routing notes",
)
```

Updates are section-scoped and repeatable. The same section is replaced on the next approved update instead of being duplicated. This makes documentation evolution suitable for GitHub repositories while keeping autonomous writes under application control.

## Complete portable tests

A consolidated regression suite is included at `tests/test_complete.py`. It is intentionally dependency-light and network-free so it can run both on GitHub and Android/Termux:

```bash
python -m pytest -q tests/test_complete.py
python -m pytest -q
```

See `ANDROID_TESTING.md` and `GITHUB_TESTING.md` for environment-specific commands.
