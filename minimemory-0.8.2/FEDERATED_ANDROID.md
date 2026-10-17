# Federated memory on Android / Termux

The 0.8 architecture is standard-library first and works without a model or network service.

## Install from the project root

```bash
cd ~/minimemory-0.8.0-federated/mm080
python -m pip install --no-build-isolation .
```

## Run the complete test suite

```bash
python -m pytest -q
```

Expected result for this release:

```text
29 passed
```

## Create domain databases

```bash
mkdir -p ~/minimemory_data
minimemory federated-health --root ~/minimemory_data
```

The first run creates:

```text
~/minimemory_data/
├── frame.db
├── memory.db
├── qna.db
├── coding.db
├── science.db
├── nature.db
├── environment.db
├── geography.db
├── history.db
└── technology.db
```

## Add knowledge

```bash
minimemory federated-remember "Photosynthesis converts light energy into chemical energy in plants." --root ~/minimemory_data --domain science --source book --confidence 0.95

minimemory federated-remember "Forests are ecosystems containing trees and other organisms." --root ~/minimemory_data --domain nature --source book --confidence 0.94
```

## Ask across domains

```bash
minimemory federated-ask "How do plants use sunlight in forests?" --root ~/minimemory_data --max-domains 2
```

The output includes the framed answer plus evidence and provenance.

## Important storage rule

`science.db`, `nature.db`, etc. are trusted knowledge stores. `frame.db` stores routing/audit/answer information. An outgoing answer is not automatically inserted into trusted knowledge.

## Import an older minimemory database

The source database is read and is not modified:

```bash
minimemory federated-import ~/old/minimemory.db --root ~/minimemory_data --domain science
```

The importer supports the older `knowledge_memory`, `assoc_memories`, `programming_terms`, and `code_components` tables.
