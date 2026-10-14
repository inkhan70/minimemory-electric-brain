"""Domain-aware, append-only federated memory and answer framing.

This module sits above the existing MemoryQA/AssociativeMemory APIs.  It keeps
knowledge stores independent (science.db, nature.db, coding.db, ...), while a
small frame.db stores only routing metadata, audit events, and answer bundles.
Incoming knowledge is append-only and outgoing answers are never written back
as trusted knowledge automatically.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(text: str) -> Counter:
    return Counter(_TOKEN_RE.findall(str(text).casefold()))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _hash(*parts: str) -> str:
    payload = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class DomainStore:
    """Small append-only SQLite knowledge store for one domain."""

    def __init__(self, path: Union[str, Path]):
        self.path = str(path)
        Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._fts = False
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    normalized TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    memory_type TEXT NOT NULL DEFAULT 'semantic',
                    source TEXT NOT NULL DEFAULT 'user',
                    confidence REAL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_items(memory_type);
                CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge_items(source);
                CREATE INDEX IF NOT EXISTS idx_knowledge_created ON knowledge_items(created_at);
                CREATE TABLE IF NOT EXISTS knowledge_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL REFERENCES knowledge_items(id),
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )
            try:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(content, content='knowledge_items', content_rowid='id')"
                )
                self._conn.executescript(
                    """
                    CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge_items BEGIN
                        INSERT INTO knowledge_fts(rowid, content) VALUES (new.id, new.content);
                    END;
                    CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge_items BEGIN
                        INSERT INTO knowledge_fts(knowledge_fts, rowid, content) VALUES('delete', old.id, old.content);
                    END;
                    """
                )
                count = int(self._conn.execute("SELECT COUNT(*) FROM knowledge_fts").fetchone()[0])
                source_count = int(self._conn.execute("SELECT COUNT(*) FROM knowledge_items").fetchone()[0])
                if count != source_count:
                    self._conn.execute("INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild')")
                self._fts = True
            except sqlite3.OperationalError:
                self._fts = False

    @staticmethod
    def _normalize(content: str) -> str:
        return " ".join(str(content).casefold().split())

    def add(self, content: str, *, memory_type: str = "semantic", source: str = "user",
            confidence: Optional[float] = None, importance: float = 0.5,
            metadata: Optional[Dict[str, Any]] = None) -> int:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        if not 0 <= float(importance) <= 1:
            raise ValueError("importance must be between 0 and 1")
        if confidence is not None and not 0 <= float(confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        normalized = self._normalize(content)
        digest = _hash(memory_type, source, normalized)
        now = _now()
        with self._lock, self._conn:
            row = self._conn.execute("SELECT id FROM knowledge_items WHERE content_hash=?", (digest,)).fetchone()
            if row:
                return int(row[0])
            cur = self._conn.execute(
                "INSERT INTO knowledge_items(content,normalized,content_hash,memory_type,source,confidence,importance,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (content.strip(), normalized, digest, memory_type, source, confidence, float(importance), _json(metadata or {}), now),
            )
            memory_id = int(cur.lastrowid)
            self._conn.execute(
                "INSERT INTO knowledge_events(memory_id,event_type,created_at,metadata_json) VALUES(?,?,?,?)",
                (memory_id, "INGESTED", now, _json({"source": source, "memory_type": memory_type})),
            )
            return memory_id

    def add_qna(self, question: str, answer: str, *, source: str = "user", confidence: Optional[float] = None,
                metadata: Optional[Dict[str, Any]] = None) -> int:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be a non-empty string")
        meta = dict(metadata or {})
        meta["question"] = question.strip()
        return self.add(answer, memory_type="qna", source=source, confidence=confidence, metadata=meta)

    def search(self, query: str, *, top_k: int = 5, min_score: float = 0.05) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        q = _tokens(query)
        with self._lock:
            rows: List[sqlite3.Row]
            if self._fts:
                # FTS is used only to narrow candidates; scoring remains deterministic.
                terms = [t for t in _TOKEN_RE.findall(query.casefold()) if t]
                if terms:
                    match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms[:24])
                    try:
                        rows = self._conn.execute(
                            "SELECT k.* FROM knowledge_fts f JOIN knowledge_items k ON k.id=f.rowid WHERE knowledge_fts MATCH ? LIMIT 200",
                            (match,),
                        ).fetchall()
                    except sqlite3.OperationalError:
                        rows = self._conn.execute("SELECT * FROM knowledge_items ORDER BY id DESC LIMIT 200").fetchall()
                else:
                    rows = self._conn.execute("SELECT * FROM knowledge_items ORDER BY id DESC LIMIT 200").fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM knowledge_items ORDER BY id DESC").fetchall()

            results: List[Dict[str, Any]] = []
            for row in rows:
                meta = json.loads(row["metadata_json"] or "{}")
                searchable = " ".join([row["content"], str(meta.get("question", "")), str(meta.get("title", "")), str(meta.get("keywords", ""))])
                lexical = _cosine(q, _tokens(searchable))
                confidence = float(row["confidence"] if row["confidence"] is not None else 0.5)
                score = min(1.0, 0.78 * lexical + 0.12 * confidence + 0.10 * float(row["importance"]))
                if score < min_score:
                    continue
                results.append({
                    "id": int(row["id"]),
                    "content": row["content"],
                    "memory_type": row["memory_type"],
                    "source": row["source"],
                    "confidence": confidence,
                    "importance": float(row["importance"]),
                    "score": round(score, 6),
                    "lexical_score": round(lexical, 6),
                    "metadata": meta,
                    "created_at": row["created_at"],
                    "domain_db": Path(self.path).name,
                })
            results.sort(key=lambda x: (x["score"], x["confidence"], x["importance"]), reverse=True)
            selected = results[:max(1, int(top_k))]
            if selected:
                self._conn.executemany("UPDATE knowledge_items SET access_count=access_count+1 WHERE id=?", [(x["id"],) for x in selected])
            return selected

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM knowledge_items").fetchone()[0])

    def health(self) -> Dict[str, Any]:
        row = self._conn.execute("PRAGMA integrity_check").fetchone()
        return {"ok": bool(row and row[0] == "ok"), "integrity": row[0] if row else "unknown", "count": self.count(), "fts5": self._fts, "path": self.path}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


DEFAULT_DOMAINS: Dict[str, Dict[str, Any]] = {
    "memory": {"description": "General associative memory", "keywords": []},
    "qna": {"description": "Question and answer knowledge", "keywords": ["question", "answer", "what", "why", "how", "when", "where", "explain"]},
    "coding": {"description": "Programming and software engineering", "keywords": ["code", "coding", "programming", "python", "javascript", "java", "kotlin", "html", "css", "api", "database", "function", "class", "algorithm", "bug", "software"]},
    "science": {"description": "Scientific knowledge", "keywords": ["science", "scientific", "physics", "chemistry", "biology", "atom", "molecule", "energy", "experiment", "cell", "genetics", "photosynthesis"]},
    "nature": {"description": "Natural world and living systems", "keywords": ["nature", "natural", "forest", "mountain", "plant", "animal", "wildlife", "ecosystem", "river", "lake", "ocean", "tree", "bird"]},
    "environment": {"description": "Environment, climate and conservation", "keywords": ["environment", "climate", "pollution", "carbon", "global warming", "conservation", "emission", "recycling", "sustainability", "ecosystem"]},
    "geography": {"description": "Places, regions and Earth geography", "keywords": ["geography", "country", "city", "place", "location", "map", "continent", "mountain", "river", "ocean"]},
    "history": {"description": "Historical knowledge", "keywords": ["history", "historical", "war", "empire", "king", "civilization", "ancient", "century", "independence"]},
    "technology": {"description": "Technology and computing systems", "keywords": ["technology", "computer", "android", "linux", "hardware", "software", "internet", "network", "ai", "model"]},
}


class FrameEngine:
    """Stores routing metadata and answer/audit data separately from knowledge."""

    def __init__(self, path: Union[str, Path]):
        self.path = str(path)
        Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS domains (
                name TEXT PRIMARY KEY,
                db_file TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                keywords_json TEXT NOT NULL DEFAULT '[]',
                enabled INTEGER NOT NULL DEFAULT 1,
                priority REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS answer_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                domains_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS frame_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

    def ensure_domain(self, name: str, db_file: str, *, description: str = "", keywords: Iterable[str] = (), priority: float = 1.0) -> None:
        now = _now()
        self._conn.execute(
            "INSERT OR IGNORE INTO domains(name,db_file,description,keywords_json,priority,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
            (name, db_file, description, _json(list(keywords)), float(priority), now, now),
        )
        self._conn.commit()

    def domains(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM domains WHERE enabled=1 ORDER BY priority DESC,name").fetchall()
        return [{"name": r["name"], "db_file": r["db_file"], "description": r["description"], "keywords": json.loads(r["keywords_json"] or "[]"), "priority": float(r["priority"])} for r in rows]

    def log_frame(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._conn.execute("INSERT INTO frame_events(event_type,payload_json,created_at) VALUES(?,?,?)", (event_type, _json(payload), _now()))
        self._conn.commit()

    def log_answer(self, query: str, answer: str, domains: List[str], evidence: List[Dict[str, Any]]) -> int:
        cur = self._conn.execute(
            "INSERT INTO answer_events(query,answer,domains_json,evidence_json,created_at) VALUES(?,?,?,?,?)",
            (query, answer, _json(domains), _json(evidence), _now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def health(self) -> Dict[str, Any]:
        row = self._conn.execute("PRAGMA integrity_check").fetchone()
        return {"ok": bool(row and row[0] == "ok"), "integrity": row[0] if row else "unknown", "domains": len(self.domains()), "path": self.path}

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]


class AnswerFramer:
    """Deterministic, extractive framer: it only uses retrieved evidence."""

    def __init__(self, max_sentences: int = 5):
        self.max_sentences = max(1, int(max_sentences))

    def frame(self, query: str, evidence: Sequence[Dict[str, Any]], *, max_domains: int = 3) -> str:
        if not evidence:
            return "I could not find a relevant stored answer in the selected knowledge stores."
        selected_domains: List[str] = []
        sentences: List[str] = []
        seen: set[str] = set()
        for item in sorted(evidence, key=lambda x: x.get("score", 0.0), reverse=True):
            domain = str(item.get("domain", ""))
            if domain and domain not in selected_domains:
                if len(selected_domains) >= max_domains:
                    continue
                selected_domains.append(domain)
            for sentence in _sentences(item.get("content", "")):
                key = " ".join(_TOKEN_RE.findall(sentence.casefold()))
                if not key or key in seen:
                    continue
                seen.add(key)
                sentences.append(sentence)
                if len(sentences) >= self.max_sentences:
                    break
            if len(sentences) >= self.max_sentences:
                break
        return " ".join(sentences)


class FederatedBrain:
    """Domain-aware memory coordinator with immutable knowledge and isolated answers."""

    def __init__(self, root: Union[str, Path] = "minimemory_data", *, auto_domains: bool = True):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.frame = FrameEngine(self.root / "frame.db")
        self.stores: Dict[str, DomainStore] = {}
        self.framer = AnswerFramer()
        if auto_domains:
            for name, spec in DEFAULT_DOMAINS.items():
                self.register_domain(name, keywords=spec["keywords"], description=spec["description"])

    def register_domain(self, name: str, *, keywords: Iterable[str] = (), description: str = "", priority: float = 1.0) -> None:
        name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name).strip().lower()).strip("_")
        if not name:
            raise ValueError("domain name must be non-empty")
        db_file = f"{name}.db"
        self.frame.ensure_domain(name, db_file, description=description, keywords=keywords, priority=priority)
        if name not in self.stores:
            self.stores[name] = DomainStore(self.root / db_file)

    def _ensure_store(self, domain: str) -> DomainStore:
        if domain not in self.stores:
            self.register_domain(domain)
        return self.stores[domain]

    def route(self, query: str, *, max_domains: int = 3) -> List[Dict[str, Any]]:
        """Select domains using both registry keywords and stored-content evidence.

        Keyword-only routing can miss cross-domain questions (for example,
        ``sunlight + forests`` can require science and nature). A lightweight
        probe against each enabled store keeps routing content-aware while the
        full retrieval still happens only for the selected domains.
        """
        q = _tokens(query)
        ranked: List[Dict[str, Any]] = []
        for item in self.frame.domains():
            domain = item["name"]
            keywords = set(_tokens(" ".join(item["keywords"])))
            overlap = sum(q.get(k, 0) for k in keywords)
            keyword_score = min(1.0, overlap / max(1, len(q)))
            content_score = 0.0
            content_lexical = 0.0
            store = self._ensure_store(domain)
            try:
                probe = store.search(query, top_k=1, min_score=0.0)
                if probe:
                    content_lexical = float(probe[0].get("lexical_score", 0.0))
                    if content_lexical > 0:
                        content_score = float(probe[0].get("score", 0.0))
            except (sqlite3.Error, ValueError):
                content_score = 0.0
            # Q&A is a generic fallback; do not let words like "what/how"
            # outrank a domain containing actual evidence.
            generic_penalty = 0.35 if domain == "qna" else 1.0
            score = generic_penalty * (0.35 * keyword_score + 0.65 * content_score)
            if content_lexical > 0 or keyword_score > 0:
                ranked.append({
                    "domain": domain,
                    "score": round(score, 6),
                    "keyword_score": round(keyword_score, 6),
                    "content_score": round(content_score, 6),
                    "db_file": item["db_file"],
                })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        if not ranked:
            ranked = [x for x in (
                {"domain": item["name"], "score": 0.0, "keyword_score": 0.0, "content_score": 0.0, "db_file": item["db_file"]}
                for item in self.frame.domains() if item["name"] in ("qna", "memory")
            )]
        return ranked[:max(1, int(max_domains))]

    def import_legacy_db(self, path: Union[str, Path], *, domain: str = "memory") -> Dict[str, int]:
        """Import data from a pre-0.8 minimemory SQLite database without mutating it.

        The legacy file is opened read-only at the application level: this method
        only SELECTs from it and creates new append-only records in the target
        domain store. This makes migration safe to repeat because content hashes
        deduplicate identical imports.
        """
        source_path = str(Path(path).expanduser())
        target = self._ensure_store(domain)
        conn = sqlite3.connect(source_path)
        conn.row_factory = sqlite3.Row
        imported = {"qna": 0, "memories": 0, "programming_terms": 0, "code_components": 0}
        try:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "knowledge_memory" in tables:
                rows = conn.execute("SELECT question,answer,source,confidence,metadata_json FROM knowledge_memory").fetchall()
                for row in rows:
                    try:
                        meta = json.loads(row["metadata_json"] or "{}")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        meta = {}
                    target.add_qna(row["question"], row["answer"], source=row["source"] or "legacy",
                                   confidence=row["confidence"], metadata={**meta, "legacy_db": source_path})
                    imported["qna"] += 1
            if "assoc_memories" in tables:
                rows = conn.execute("SELECT content,memory_type,source,confidence,importance,metadata_json FROM assoc_memories").fetchall()
                for row in rows:
                    try:
                        meta = json.loads(row["metadata_json"] or "{}")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        meta = {}
                    target.add(row["content"], memory_type=row["memory_type"], source=row["source"] or "legacy",
                               confidence=row["confidence"], importance=row["importance"], metadata={**meta, "legacy_db": source_path})
                    imported["memories"] += 1
            if "programming_terms" in tables:
                rows = conn.execute("SELECT term,category,language,meaning,simple_meaning,aliases_json,related_json,source,license,confidence FROM programming_terms").fetchall()
                for row in rows:
                    meta = {"term": row["term"], "category": row["category"], "language": row["language"],
                            "aliases": json.loads(row["aliases_json"] or "[]"), "related": json.loads(row["related_json"] or "[]"),
                            "license": row["license"], "legacy_db": source_path}
                    target.add(f"{row['term']}: {row['simple_meaning']} {row['meaning']}", memory_type="programming",
                               source=row["source"] or "legacy", confidence=row["confidence"], metadata=meta)
                    imported["programming_terms"] += 1
            if "code_components" in tables:
                rows = conn.execute("SELECT component_key,name,language,framework,platform,purpose,code,source,license,version,metadata_json,confidence FROM code_components").fetchall()
                for row in rows:
                    try:
                        meta = json.loads(row["metadata_json"] or "{}")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        meta = {}
                    meta.update({"component_key": row["component_key"], "language": row["language"], "framework": row["framework"],
                                 "platform": row["platform"], "version": row["version"], "license": row["license"], "code": row["code"],
                                 "legacy_db": source_path})
                    target.add(f"{row['name']}: {row['purpose']}", memory_type="code_component", source=row["source"] or "legacy",
                               confidence=row["confidence"], metadata=meta)
                    imported["code_components"] += 1
        finally:
            conn.close()
        self.frame.log_frame("LEGACY_IMPORT", {"path": source_path, "domain": domain, "counts": imported})
        return imported

    def remember(self, content: str, *, domain: str = "memory", memory_type: str = "semantic",
                 source: str = "user", confidence: Optional[float] = None, importance: float = 0.5,
                 metadata: Optional[Dict[str, Any]] = None) -> int:
        store = self._ensure_store(domain)
        return store.add(content, memory_type=memory_type, source=source, confidence=confidence, importance=importance, metadata=metadata)

    def remember_qna(self, question: str, answer: str, *, domain: str = "qna", source: str = "user",
                     confidence: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
        return self._ensure_store(domain).add_qna(question, answer, source=source, confidence=confidence, metadata=metadata)

    def retrieve(self, query: str, *, domains: Optional[Iterable[str]] = None, max_domains: int = 3,
                 top_k: int = 8, min_score: float = 0.05) -> Dict[str, Any]:
        chosen = list(domains) if domains else [x["domain"] for x in self.route(query, max_domains=max_domains)]
        per_domain = max(1, math.ceil(int(top_k) / max(1, len(chosen))))
        evidence: List[Dict[str, Any]] = []
        domain_results: Dict[str, List[Dict[str, Any]]] = {}
        for domain in chosen:
            store = self._ensure_store(domain)
            results = store.search(query, top_k=per_domain, min_score=min_score)
            for item in results:
                item = dict(item)
                item["domain"] = domain
                evidence.append(item)
            domain_results[domain] = results
        evidence.sort(key=lambda x: (x["score"], x.get("confidence", 0.0)), reverse=True)
        return {"query": query, "domains": chosen, "results": evidence[:int(top_k)], "by_domain": domain_results}

    def ask(self, query: str, *, domains: Optional[Iterable[str]] = None, max_domains: int = 3,
            top_k: int = 8, min_score: float = 0.05, max_answer_sentences: int = 5,
            log_answer: bool = True) -> Dict[str, Any]:
        bundle = self.retrieve(query, domains=domains, max_domains=max_domains, top_k=top_k, min_score=min_score)
        self.framer.max_sentences = max(1, int(max_answer_sentences))
        answer = self.framer.frame(query, bundle["results"], max_domains=max_domains)
        sources = [
            {"domain": x["domain"], "memory_id": x["id"], "score": x["score"], "source": x["source"], "confidence": x["confidence"]}
            for x in bundle["results"]
        ]
        event_id = self.frame.log_answer(query, answer, bundle["domains"], sources) if log_answer else None
        return {
            "status": "SUCCESS" if bundle["results"] else "EMPTY_MEMORY",
            "query": query,
            "answer": answer,
            "domains": bundle["domains"],
            "evidence": bundle["results"],
            "provenance": sources,
            "answer_event_id": event_id,
        }

    def stats(self) -> Dict[str, Any]:
        return {"root": str(self.root), "frame": self.frame.health(), "domains": {name: store.health() for name, store in self.stores.items()}}

    def health(self) -> Dict[str, Any]:
        checks = self.stats()
        checks["ok"] = bool(checks["frame"]["ok"] and all(x["ok"] for x in checks["domains"].values()))
        return checks

    def close(self) -> None:
        for store in self.stores.values():
            store.close()
        self.frame.close()

    def __enter__(self) -> "FederatedBrain":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
