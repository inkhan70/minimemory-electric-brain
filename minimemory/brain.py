"""High-level local electric-brain facade.

The brain orchestrates memory, associations, optional behavior evolution,
privacy policy, resource budgets, AI connectors, and portable knowledge packs.
Behavior learning is opt-in and sensitive behavior is never persisted by the
built-in policy.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from .associative import AssociativeMemory
from .budget import ResourceBudget
from .core import MemoryQA
from .evolution import MemoryEvolution
from .privacy import classify_behavior
from .programming_knowledge import seed as seed_programming
from .brain_pack import export_brain, import_brain, inspect_brain_pack
from .research import WebResearcher
from .documentation import update_documentation
from .router import QueryRoute, classify_query, choose_route


class LocalBrain:
    """A local, associative, privacy-aware memory brain.

    ``behavior_tracking`` defaults to False. Enabling it stores only aggregate
    non-sensitive behavior signals; exact sensitive queries are never stored.
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        *,
        embedding_fn: Optional[Callable[[Any], Sequence[float]]] = None,
        behavior_tracking: bool = False,
        network_budget_mb: float = 0.0,
        storage_budget_mb: float = 0.0,
    ):
        if str(db_path) == ":memory:":
            raise ValueError("LocalBrain requires a file-backed SQLite database so its brain components share one store")
        self.db_path = str(db_path)
        self.memory = MemoryQA(self.db_path)
        self.associative = AssociativeMemory(self.db_path, embedding_fn=embedding_fn)
        self._brain_conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self._brain_conn.row_factory = sqlite3.Row
        self._brain_conn.execute("PRAGMA busy_timeout=30000")
        self._brain_conn.execute("PRAGMA journal_mode=WAL")
        self.evolution = MemoryEvolution(self._brain_conn, enabled=behavior_tracking)
        self.budget = ResourceBudget(network_mb=network_budget_mb, storage_mb=storage_budget_mb)
        self._closed = False

    def remember(self, content: str, **kwargs: Any) -> int:
        return self.associative.remember(content, **kwargs)

    def remember_image(self, description: str, *, embedding: Optional[Sequence[float]] = None,
                       source: str = "image", confidence: Optional[float] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> int:
        meta = dict(metadata or {})
        meta.setdefault("modality", "image")
        return self.associative.remember(description, memory_type="episodic", source=source,
                                         confidence=confidence, importance=0.7,
                                         metadata=meta, embedding=embedding)

    def route(self, query: str, *, top_k: int = 5) -> Dict[str, Any]:
        """Classify and route a query using local evidence only.

        The returned route is one of ``qna``, ``memory``, ``code``,
        ``programming``, or ``unknown``.  No AI call is made here.
        """
        features = classify_query(query)
        qna_hits = self.memory.search(query, top_k=top_k, min_score=0.05)
        memory_hits = self.associative.recall(query, top_k=top_k, min_score=0.05)
        code_hits = self.associative.search_code(query, limit=top_k)
        programming_hits = self.associative.search_programming(query, limit=top_k)
        route = choose_route(features, qna_hit=bool(qna_hits), memory_hit=bool(memory_hits),
                             code_hit=bool(code_hits), programming_hit=bool(programming_hits))
        return {
            "route": route.value,
            "features": features,
            "qna_hits": qna_hits,
            "memory_hits": memory_hits,
            "code_hits": code_hits,
            "programming_hits": programming_hits,
        }

    def ask(self, query: str, *, generator: Optional[Callable[..., Any]] = None,
            threshold: float = 0.55, top_k: int = 5, auto_learn: bool = False,
            approve: bool = False) -> Dict[str, Any]:
        """Run the complete local routing flow.

        Flow: classify input -> observe opt-in behavior -> inspect Q&A/memory/code
        stores -> select a local route -> optionally fall back to ``MemoryQA.chat``
        when a generator is supplied and no sufficiently strong local answer exists.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        routed = self.route(query, top_k=top_k)
        observation = self.observe(query)
        route = routed["route"]
        if route == QueryRoute.CODE.value and routed["code_hits"]:
            hit = routed["code_hits"][0]
            return {"answer": hit["code"], "source": "code", "route": route,
                    "confidence": float(hit.get("confidence") or hit.get("score") or 0.0),
                    "matches": routed["code_hits"], "learned": False, "behavior": observation}
        if route == QueryRoute.PROGRAMMING.value and routed["programming_hits"]:
            hit = routed["programming_hits"][0]
            return {"answer": hit["simple_meaning"], "source": "programming", "route": route,
                    "confidence": float(hit.get("score") or 0.0),
                    "matches": routed["programming_hits"], "learned": False, "behavior": observation}
        if route == QueryRoute.MEMORY.value and routed["memory_hits"]:
            hit = routed["memory_hits"][0]
            return {"answer": hit["content"], "source": "memory", "route": route,
                    "confidence": float(hit["score"]), "matches": routed["memory_hits"],
                    "learned": False, "behavior": observation}
        if route == QueryRoute.QNA.value and routed["qna_hits"]:
            hit = routed["qna_hits"][0]
            if float(hit.get("confidence") or 0.0) >= threshold:
                return {"answer": hit["answer"], "source": "qna", "route": route,
                        "confidence": float(hit.get("confidence") or 0.0),
                        "matches": routed["qna_hits"], "learned": False, "behavior": observation}
        if generator is not None:
            result = self.memory.chat(query, generator, threshold=threshold, context_items=top_k,
                                      auto_learn=auto_learn, approve=approve)
            result.update({"route": "ai_fallback", "behavior": observation})
            return result
        return {"answer": None, "source": "none", "route": "unknown", "confidence": 0.0,
                "matches": {"qna": routed["qna_hits"], "memory": routed["memory_hits"],
                            "code": routed["code_hits"], "programming": routed["programming_hits"]},
                "learned": False, "behavior": observation}

    def recall(self, query: str, **kwargs: Any):
        """Recall associative memories and expose the current evolution signal."""
        results = self.associative.recall(query, **kwargs)
        topics = self.evolution.top_topics(5)
        if topics:
            q = query.casefold()
            for item in results:
                boost = 0.0
                for topic in topics:
                    words = topic["topic"].casefold().split()
                    if words and any(word in item["content"].casefold() for word in words):
                        boost = max(boost, 0.05 * topic["strength"])
                if boost:
                    item["evolution_boost"] = round(boost, 6)
                    item["score"] = round(min(1.0, item["score"] + boost), 6)
            results.sort(key=lambda x: (x["score"], x.get("importance", 0), x.get("confidence", 0)), reverse=True)
        return results

    def bundle(self, query: str, **kwargs: Any):
        bundle = self.associative.memory_bundle(query, **kwargs)
        bundle["evolution"] = self.evolution.top_topics(5)
        return bundle

    def observe(self, query: str) -> Dict[str, Any]:
        """Observe a user query through the privacy firewall.

        When behavior tracking is disabled, the classifier runs transiently and
        no event is stored. Sensitive observations are always non-persistent.
        """
        observation = classify_behavior(query)
        return self.evolution.observe(observation)

    def set_behavior_tracking(self, enabled: bool) -> Dict[str, Any]:
        self.evolution.set_enabled(enabled)
        return self.evolution.status()

    def behavior_status(self) -> Dict[str, Any]:
        return self.evolution.status()

    def clear_behavior(self) -> Dict[str, Any]:
        self.evolution.clear()
        return self.evolution.status()

    def evolve(self, *, min_accesses: int = 2) -> Dict[str, Any]:
        """Run a bounded consolidation/evolution pulse.

        Consolidation copies only existing episodic content; it never invents
        facts. Behavior evolution only uses permitted non-sensitive aggregates.
        """
        consolidated = self.associative.consolidate(min_accesses=min_accesses)
        return {"consolidated": consolidated, "topics": self.evolution.top_topics(10)}

    def pulse(self, *, min_accesses: int = 2) -> Dict[str, Any]:
        """Return a production-friendly snapshot of the electric brain state."""
        return {
            "health": self.health(),
            "qna": self.memory.stats(),
            "associative": self.associative.stats(),
            "evolution": self.evolution.status(),
            "storage": self.budget.storage_status(self.db_path),
        }

    def seed_programming_knowledge(self) -> int:
        return seed_programming(self.associative)

    def research(self, question: str, *, researcher: Optional[WebResearcher] = None, max_results: Optional[int] = None) -> Dict[str, Any]:
        """Research the web without automatically writing facts into memory."""
        engine = researcher or WebResearcher(budget=self.budget)
        return engine.research(question, max_results=max_results)

    def store_research(self, question: str, answer: str, result: Dict[str, Any], *, validator: Optional[Callable[[Dict[str, Any]], bool]] = None, approved: bool = False) -> bool:
        """Store a researched answer only when explicitly approved/validated."""
        candidate = WebResearcher.candidate(question, answer, result)
        accepted = approved or WebResearcher.validator_accepts(candidate, validator)
        if not accepted:
            return False
        return self.memory.learn(question, answer, source="web-research", confidence=candidate["confidence"], metadata=candidate["metadata"])

    def update_documentation(self, path: str, section: str, content: str, *,
                             approved: bool = False, heading: Optional[str] = None) -> Dict[str, Any]:
        """Safely update one Markdown documentation section.

        Evolution never edits repository documentation implicitly. Callers must
        explicitly approve the generated/learned content before it is written.
        """
        return update_documentation(path, section, content, approved=approved, heading=heading)

    def export_knowledge(self, destination: str) -> str:
        """Export portable `.mmpack` knowledge; behavior is never exported."""
        return export_brain(self.db_path, destination)

    def inspect_knowledge_pack(self, source: str) -> Dict[str, Any]:
        return inspect_brain_pack(source)

    def import_knowledge(self, source: str) -> Dict[str, int]:
        result = import_brain(self.db_path, source)
        self.memory.reload_cache()
        return result

    def health(self) -> Dict[str, Any]:
        return self.associative.health()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.associative.close()
        finally:
            try:
                self.memory.close()
            finally:
                self._brain_conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
