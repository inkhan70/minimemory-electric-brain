"""High-level local associative memory facade."""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Sequence
from .associative import AssociativeMemory
from .core import MemoryQA
from .programming_knowledge import seed as seed_programming
from .knowledge_gate import KnowledgeGate

class LocalBrain:
    """Combine Q&A memory and associative memory behind one small API."""
    def __init__(self, db_path: str = "memory.db", *, embedding_fn: Optional[Callable[[Any], Sequence[float]]] = None):
        self.memory = MemoryQA(db_path)
        # A separate SQLite connection to the same database is safe under WAL.
        self.associative = AssociativeMemory(db_path, embedding_fn=embedding_fn)
        # Evidence-aware claim ledger shares the DB file but uses separate tables.
        self.knowledge = KnowledgeGate(db_path)

    def remember(self, content: str, **kwargs: Any) -> int:
        return self.associative.remember(content, **kwargs)

    def remember_image(self, description: str, *, embedding: Optional[Sequence[float]] = None,
                       source: str = "image", confidence: Optional[float] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> int:
        """Store image-derived concepts plus an optional vision embedding.

        The image itself is intentionally not copied into SQLite. Store a path,
        content URI, hash, or object identifier in metadata and keep the binary
        asset in the application's file storage.
        """
        meta = dict(metadata or {})
        meta.setdefault("modality", "image")
        return self.associative.remember(description, memory_type="episodic", source=source,
                                         confidence=confidence, importance=0.7,
                                         metadata=meta, embedding=embedding)

    def recall(self, query: str, **kwargs: Any):
        return self.associative.recall(query, **kwargs)

    def bundle(self, query: str, **kwargs: Any):
        return self.associative.memory_bundle(query, **kwargs)

    def admit_claim(self, claim: str, **kwargs: Any):
        """Evaluate and store a claim without treating it as verified truth.

        Set evidence_polarity to supports/opposes/neutral. Scores are admission
        heuristics only; use source/context/confidence and review contested claims.
        """
        return self.knowledge.admit(claim, **kwargs)

    def search_claims(self, query: str, *, limit: int = 10):
        return self.knowledge.search(query, limit=limit)

    def seed_programming_knowledge(self) -> int:
        return seed_programming(self.associative)

    def health(self):
        return self.associative.health()

    def close(self) -> None:
        self.knowledge.close()
        self.associative.close()
        self.memory.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
