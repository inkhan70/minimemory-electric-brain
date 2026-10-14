# minimemory 0.7.0 — Associative Local Memory

`minimemory` is an offline-first SQLite memory layer for local AI applications. Version 0.7 adds an **associative memory layer** designed for multimodal applications, structured knowledge, programming knowledge, code components, entity relationships, and grounded retrieval.

## What it does

- Persistent Q&A memory (`MemoryQA`).
- Semantic/fallback text search.
- Episodic and semantic memories.
- Entity and relationship graph.
- Associative recall: lexical similarity + optional embeddings + graph links + confidence.
- Multimodal-ready storage: images/audio can be represented by captions/concepts plus application-provided embeddings.
- Programming dictionary with compact English meanings.
- Code-component storage with dependency/provenance/license metadata.
- Project planning helper that identifies missing requirements before code generation.
- Bounded text refinement and verification hooks.
- SQLite WAL, integrity checks, backups and no mandatory third-party dependencies.

## Important design rule

Similarity is evidence, not truth. Association is not proof. Every retrieved memory can carry source, confidence, timestamps and metadata. The library does not claim to reproduce a human brain; it implements useful computational patterns such as associative retrieval and persistent episodic/semantic storage.

## Quick start

```python
from minimemory import LocalBrain

with LocalBrain("memory.db") as brain:
    brain.seed_programming_knowledge()
    brain.remember(
        "A previous trip included a snowy mountain, pine forest and lake.",
        memory_type="episodic",
        source="user",
        confidence=0.9,
        metadata={"trip": "example"},
    )
    print(brain.recall("mountain lake"))
```

## Image/multimodal memory

The core does not force a particular vision model. Your Android/local AI application can calculate an image embedding and pass it to `remember_image`:

```python
brain.remember_image(
    "snowy mountain, pine forest, lake",
    embedding=image_vector,
    metadata={"asset_id": "photo-17", "location": "unknown"},
)
```

The binary image stays in your application's file storage. SQLite stores concepts, provenance and the optional vector. This keeps the core lightweight for Termux.

## Associative recall

A query can activate multiple memory paths:

```text
query
  ├─ lexical similarity
  ├─ optional embedding similarity
  ├─ entity matches
  ├─ graph relationships
  └─ confidence/importance
             ↓
        memory bundle
             ↓
          local LLM
```

A strong match should not be treated as proof of identity. If several entities share a name, keep them separate and require context for disambiguation.

## Programming knowledge

```python
from minimemory import LocalBrain

with LocalBrain("coding.db") as brain:
    count = brain.seed_programming_knowledge()
    print(count)
    print(brain.associative.search_programming("Python function"))
```

The seed contains concise original definitions for languages, concepts, keywords, web terms, databases, testing, and software-engineering concepts. Extend it with `add_programming_term` and preserve source/license information for imported material.

## Code components

Store reusable, user-owned or properly licensed components with language, platform, dependencies, tests, version and provenance. This lets a local LLM retrieve small verified building blocks instead of blindly copying large code corpora.

## Project planning

```python
from minimemory import ProjectPlanner

plan = ProjectPlanner().analyze(
    "build a calculator app",
    platform="android",
    language="kotlin",
)
print(plan)
```

The planner reports missing information instead of silently guessing it.

## Install

```bash
python -m pip install .
```

The core requires only Python 3.9+ and the standard library. Optional semantic dependencies from previous releases remain available.

## Tests

```bash
python -m pytest -q
```

## Scope

This is a memory/retrieval infrastructure layer, not an LLM and not a brain simulation. A local model such as a GGUF model can consume `memory_bundle()` as grounded context and produce the final response.

## 0.8 Federated domain memory and safe answer framing

Version 0.8 adds an optional `FederatedBrain` architecture for separating knowledge by domain:

```text
FederatedBrain
  -> router
  -> science.db / nature.db / environment.db / coding.db / qna.db / memory.db
  -> evidence bundle
  -> deterministic AnswerFramer
  -> answer + provenance
```

`frame.db` stores domain registry, routing metadata, audit events, and answer events. It is not treated as trusted knowledge.

Knowledge stores are append-only at the API level: duplicate ingestion returns the existing record instead of overwriting its content, source, confidence, or creation time. Retrieval and answer generation do not update trusted knowledge. Search access counters are operational metadata only.

```python
from minimemory import FederatedBrain

with FederatedBrain("knowledge") as brain:
    brain.remember("Photosynthesis converts light energy into chemical energy in plants.", domain="science", source="book", confidence=0.95)
    brain.remember("Forests are ecosystems containing trees and other organisms.", domain="nature", source="book", confidence=0.94)
    result = brain.ask("How do plants use sunlight in forests?", max_domains=2)
    print(result["answer"])
    print(result["provenance"])
```

Outgoing answers are logged separately in `frame.db`; they are not automatically inserted into science/nature/etc. This prevents an LLM response from silently becoming trusted knowledge.
