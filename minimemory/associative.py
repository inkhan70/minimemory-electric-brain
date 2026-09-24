"""Associative, episodic, semantic, and multimodal-friendly memory.

This module is deliberately offline-first. It stores structured memories and
relationships in SQLite and accepts optional application-provided embeddings
for images, audio, or other modalities. It does not require a model to work.
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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


def _vector_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) ** 2 for x in a))
    nb = math.sqrt(sum(float(y) ** 2 for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_confidence(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return value


class AssociativeMemory:
    """SQLite-backed associative memory for local AI applications.

    Memory is organized into semantic facts, episodes, entities, relations and
    reusable programming knowledge. Retrieval combines lexical similarity,
    optional embedding similarity, graph proximity, recency and confidence.

    ``embedding_fn`` is optional and should return a numeric vector for any
    supported input. For images, pass an application-specific vision encoder;
    the core library never assumes a particular model or network service.
    """

    def __init__(self, db_path: str = "memory.db", *, embedding_fn: Optional[Callable[[Any], Sequence[float]]] = None):
        self.db_path = str(db_path)
        self.embedding_fn = embedding_fn
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS assoc_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user',
                    confidence REAL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    embedding_json TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(memory_type, normalized, source)
                );
                CREATE INDEX IF NOT EXISTS idx_assoc_type ON assoc_memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_assoc_updated ON assoc_memories(updated_at);
                CREATE INDEX IF NOT EXISTS idx_assoc_source ON assoc_memories(source);

                CREATE TABLE IF NOT EXISTS assoc_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_key TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL DEFAULT 'unknown',
                    description TEXT NOT NULL DEFAULT '',
                    confidence REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_assoc_entity_name ON assoc_entities(name);

                CREATE TABLE IF NOT EXISTS assoc_aliases (
                    entity_id INTEGER NOT NULL REFERENCES assoc_entities(id) ON DELETE CASCADE,
                    alias TEXT NOT NULL,
                    UNIQUE(entity_id, alias)
                );
                CREATE INDEX IF NOT EXISTS idx_assoc_alias ON assoc_aliases(alias);

                CREATE TABLE IF NOT EXISTS assoc_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id INTEGER NOT NULL REFERENCES assoc_entities(id) ON DELETE CASCADE,
                    predicate TEXT NOT NULL,
                    object_id INTEGER REFERENCES assoc_entities(id) ON DELETE CASCADE,
                    object_text TEXT,
                    source TEXT NOT NULL DEFAULT 'user',
                    confidence REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_assoc_rel_subject ON assoc_relations(subject_id);
                CREATE INDEX IF NOT EXISTS idx_assoc_rel_object ON assoc_relations(object_id);

                CREATE TABLE IF NOT EXISTS assoc_links (
                    memory_id INTEGER NOT NULL REFERENCES assoc_memories(id) ON DELETE CASCADE,
                    entity_id INTEGER NOT NULL REFERENCES assoc_entities(id) ON DELETE CASCADE,
                    relation TEXT NOT NULL DEFAULT 'mentions',
                    UNIQUE(memory_id, entity_id, relation)
                );

                CREATE TABLE IF NOT EXISTS programming_terms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    language TEXT,
                    meaning TEXT NOT NULL,
                    simple_meaning TEXT NOT NULL,
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    related_json TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL DEFAULT 'built-in',
                    license TEXT,
                    confidence REAL NOT NULL DEFAULT 1.0
                );
                CREATE INDEX IF NOT EXISTS idx_prog_term ON programming_terms(term);
                CREATE INDEX IF NOT EXISTS idx_prog_lang ON programming_terms(language);

                CREATE TABLE IF NOT EXISTS code_components (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_key TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    language TEXT NOT NULL,
                    framework TEXT,
                    platform TEXT,
                    purpose TEXT NOT NULL,
                    inputs_json TEXT NOT NULL DEFAULT '[]',
                    outputs_json TEXT NOT NULL DEFAULT '[]',
                    dependencies_json TEXT NOT NULL DEFAULT '[]',
                    code TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user',
                    license TEXT,
                    version TEXT,
                    tests_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    confidence REAL NOT NULL DEFAULT 1.0
                );
                CREATE INDEX IF NOT EXISTS idx_component_language ON code_components(language);
                CREATE INDEX IF NOT EXISTS idx_component_framework ON code_components(framework);
                CREATE INDEX IF NOT EXISTS idx_component_platform ON code_components(platform);
                """
            )

    @staticmethod
    def _normalize(content: str) -> str:
        return " ".join(str(content).casefold().split())

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def remember(self, content: str, *, memory_type: str = "semantic", source: str = "user",
                 confidence: Optional[float] = None, importance: float = 0.5,
                 metadata: Optional[Dict[str, Any]] = None, embedding: Optional[Sequence[float]] = None) -> int:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        if not 0 <= float(importance) <= 1:
            raise ValueError("importance must be between 0 and 1")
        confidence = _validate_confidence(confidence)
        metadata = {} if metadata is None else dict(metadata)
        vector = list(embedding) if embedding is not None else None
        if vector is None and self.embedding_fn is not None:
            try:
                vector = list(self.embedding_fn(content))
            except Exception:
                vector = None
        now = _now()
        normalized = self._normalize(content)
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT id FROM assoc_memories WHERE memory_type=? AND normalized=? AND source=?",
                (memory_type, normalized, source),
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE assoc_memories SET confidence=COALESCE(?, confidence), importance=?, metadata_json=?, embedding_json=COALESCE(?, embedding_json), updated_at=? WHERE id=?",
                    (confidence, float(importance), _json(metadata), _json(vector) if vector is not None else None, now, int(row[0])),
                )
                return int(row[0])
            cur = self._conn.execute(
                "INSERT INTO assoc_memories(memory_type,content,normalized,source,confidence,importance,embedding_json,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (memory_type, content.strip(), normalized, source, confidence, float(importance), _json(vector) if vector is not None else None, _json(metadata), now, now),
            )
            return int(cur.lastrowid)

    def remember_episode(self, content: str, *, source: str = "user", confidence: Optional[float] = None,
                         importance: float = 0.7, metadata: Optional[Dict[str, Any]] = None,
                         embedding: Optional[Sequence[float]] = None) -> int:
        return self.remember(content, memory_type="episodic", source=source, confidence=confidence,
                             importance=importance, metadata=metadata, embedding=embedding)

    def associate(self, memory_id: int, entity_key: str, *, relation: str = "mentions") -> None:
        entity = self._conn.execute("SELECT id FROM assoc_entities WHERE entity_key=?", (entity_key,)).fetchone()
        if not entity:
            raise KeyError(f"unknown entity: {entity_key}")
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO assoc_links(memory_id,entity_id,relation) VALUES(?,?,?)", (memory_id, int(entity[0]), relation))

    def upsert_entity(self, entity_key: str, name: str, *, entity_type: str = "unknown", description: str = "",
                      aliases: Optional[Iterable[str]] = None, confidence: Optional[float] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> int:
        if not entity_key.strip() or not name.strip():
            raise ValueError("entity_key and name are required")
        confidence = _validate_confidence(confidence)
        aliases = [str(a).strip() for a in (aliases or []) if str(a).strip()]
        now = _now()
        with self._lock, self._conn:
            row = self._conn.execute("SELECT id FROM assoc_entities WHERE entity_key=?", (entity_key,)).fetchone()
            if row:
                entity_id = int(row[0])
                self._conn.execute("UPDATE assoc_entities SET name=?,entity_type=?,description=?,confidence=COALESCE(?,confidence),metadata_json=?,updated_at=? WHERE id=?",
                                   (name.strip(), entity_type, description, confidence, _json(metadata or {}), now, entity_id))
            else:
                cur = self._conn.execute("INSERT INTO assoc_entities(entity_key,name,entity_type,description,confidence,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                                         (entity_key, name.strip(), entity_type, description, confidence, _json(metadata or {}), now, now))
                entity_id = int(cur.lastrowid)
            for alias in aliases:
                self._conn.execute("INSERT OR IGNORE INTO assoc_aliases(entity_id,alias) VALUES(?,?)", (entity_id, alias))
            return entity_id

    def add_relation(self, subject: str, predicate: str, object_value: str, *, object_is_entity: bool = True,
                     source: str = "user", confidence: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> int:
        s = self._conn.execute("SELECT id FROM assoc_entities WHERE entity_key=?", (subject,)).fetchone()
        if not s:
            raise KeyError(f"unknown subject entity: {subject}")
        object_id = None
        object_text = object_value
        if object_is_entity:
            o = self._conn.execute("SELECT id FROM assoc_entities WHERE entity_key=?", (object_value,)).fetchone()
            if not o:
                raise KeyError(f"unknown object entity: {object_value}")
            object_id = int(o[0])
            object_text = None
        confidence = _validate_confidence(confidence)
        with self._lock, self._conn:
            cur = self._conn.execute("INSERT INTO assoc_relations(subject_id,predicate,object_id,object_text,source,confidence,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                                     (int(s[0]), predicate, object_id, object_text, source, confidence, _json(metadata or {}), _now()))
            return int(cur.lastrowid)

    def _entity_ids_for_query(self, query: str) -> List[int]:
        terms = set(_tokens(query))
        if not terms:
            return []
        rows = self._conn.execute("SELECT id,name FROM assoc_entities").fetchall()
        result = []
        for row in rows:
            if set(_tokens(row[1])) & terms:
                result.append(int(row[0]))
        alias_rows = self._conn.execute("SELECT entity_id,alias FROM assoc_aliases").fetchall()
        for row in alias_rows:
            if set(_tokens(row[1])) & terms and int(row[0]) not in result:
                result.append(int(row[0]))
        return result

    def recall(self, query: str, *, top_k: int = 8, memory_types: Optional[Iterable[str]] = None,
               min_score: float = 0.05) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        q_tokens = _tokens(query)
        allowed = set(memory_types or [])
        entity_ids = self._entity_ids_for_query(query)
        rows = self._conn.execute("SELECT * FROM assoc_memories ORDER BY updated_at DESC").fetchall()
        candidates: List[Dict[str, Any]] = []
        for row in rows:
            if allowed and row["memory_type"] not in allowed:
                continue
            lexical = _cosine(q_tokens, _tokens(row["content"]))
            embedding_score = 0.0
            if self.embedding_fn is not None and row["embedding_json"]:
                try:
                    qv = list(self.embedding_fn(query))
                    embedding_score = max(0.0, _vector_cosine(qv, json.loads(row["embedding_json"])))
                except Exception:
                    embedding_score = 0.0
            linked = self._conn.execute("SELECT entity_id FROM assoc_links WHERE memory_id=?", (int(row["id"]),)).fetchall()
            graph_bonus = 0.15 if any(int(x[0]) in entity_ids for x in linked) else 0.0
            confidence = float(row["confidence"] if row["confidence"] is not None else 0.5)
            score = min(1.0, 0.55 * lexical + 0.30 * embedding_score + graph_bonus + 0.15 * confidence)
            if score < min_score:
                continue
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            candidates.append({
                "id": int(row["id"]), "memory_type": row["memory_type"], "content": row["content"],
                "source": row["source"], "confidence": confidence, "score": round(score, 6),
                "lexical_score": round(lexical, 6), "embedding_score": round(embedding_score, 6),
                "graph_bonus": graph_bonus, "importance": float(row["importance"]),
                "metadata": metadata, "created_at": row["created_at"], "updated_at": row["updated_at"],
            })
        candidates.sort(key=lambda x: (x["score"], x["importance"], x["confidence"]), reverse=True)
        selected = candidates[: max(1, int(top_k))]
        with self._lock, self._conn:
            for item in selected:
                self._conn.execute("UPDATE assoc_memories SET access_count=access_count+1 WHERE id=?", (item["id"],))
        return selected

    def related(self, memory_id: int, *, depth: int = 2, limit: int = 30) -> List[Dict[str, Any]]:
        if depth < 1:
            return []
        seen_mem = {int(memory_id)}
        frontier = [int(memory_id)]
        result: List[Dict[str, Any]] = []
        for _ in range(depth):
            if not frontier or len(result) >= limit:
                break
            placeholders = ",".join("?" for _ in frontier)
            entity_rows = self._conn.execute(f"SELECT DISTINCT entity_id FROM assoc_links WHERE memory_id IN ({placeholders})", frontier).fetchall()
            entity_ids = [int(r[0]) for r in entity_rows]
            if not entity_ids:
                break
            placeholders = ",".join("?" for _ in entity_ids)
            mem_rows = self._conn.execute(
                f"SELECT DISTINCT m.* FROM assoc_memories m JOIN assoc_links l ON l.memory_id=m.id WHERE l.entity_id IN ({placeholders}) ORDER BY m.updated_at DESC LIMIT ?",
                entity_ids + [limit],
            ).fetchall()
            frontier = []
            for row in mem_rows:
                mid = int(row["id"])
                if mid in seen_mem:
                    continue
                seen_mem.add(mid)
                result.append({"id": mid, "memory_type": row["memory_type"], "content": row["content"], "source": row["source"]})
                frontier.append(mid)
                if len(result) >= limit:
                    break
        return result

    def consolidate(self, *, min_accesses: int = 2) -> int:
        """Promote frequently recalled episodic memories into semantic summaries.

        Consolidation is deliberately conservative: it copies only exact stored
        content and records the episode as its provenance; it does not invent facts.
        """
        rows = self._conn.execute("SELECT * FROM assoc_memories WHERE memory_type='episodic' AND access_count>=?", (int(min_accesses),)).fetchall()
        created = 0
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            metadata = dict(metadata)
            metadata["consolidated_from"] = int(row["id"])
            self.remember(row["content"], memory_type="semantic", source=f"consolidated:{row['id']}",
                          confidence=row["confidence"], importance=min(1.0, float(row["importance"]) + 0.1), metadata=metadata)
            created += 1
        return created

    def stats(self) -> Dict[str, Any]:
        rows = self._conn.execute("SELECT memory_type,COUNT(*) AS n FROM assoc_memories GROUP BY memory_type").fetchall()
        return {"memories": sum(int(r[1]) for r in rows), "by_type": {r[0]: int(r[1]) for r in rows},
                "entities": int(self._conn.execute("SELECT COUNT(*) FROM assoc_entities").fetchone()[0]),
                "relations": int(self._conn.execute("SELECT COUNT(*) FROM assoc_relations").fetchone()[0]),
                "programming_terms": int(self._conn.execute("SELECT COUNT(*) FROM programming_terms").fetchone()[0]),
                "code_components": int(self._conn.execute("SELECT COUNT(*) FROM code_components").fetchone()[0])}

    def add_programming_term(self, term: str, *, category: str, meaning: str, simple_meaning: str,
                             language: Optional[str] = None, aliases: Optional[Iterable[str]] = None,
                             related: Optional[Iterable[str]] = None, source: str = "user",
                             license: Optional[str] = None, confidence: float = 1.0) -> int:
        confidence = _validate_confidence(confidence) or 0.0
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO programming_terms(term,category,language,meaning,simple_meaning,aliases_json,related_json,source,license,confidence) VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(term) DO UPDATE SET category=excluded.category,language=excluded.language,meaning=excluded.meaning,simple_meaning=excluded.simple_meaning,aliases_json=excluded.aliases_json,related_json=excluded.related_json,source=excluded.source,license=excluded.license,confidence=excluded.confidence",
                (term.strip(), category, language, meaning.strip(), simple_meaning.strip(), _json(list(aliases or [])), _json(list(related or [])), source, license, confidence),
            )
            row = self._conn.execute("SELECT id FROM programming_terms WHERE term=?", (term.strip(),)).fetchone()
            return int(row[0])

    def search_programming(self, query: str, *, language: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        q = _tokens(query)
        rows = self._conn.execute("SELECT * FROM programming_terms" + (" WHERE language=?" if language else ""), ((language,) if language else ())).fetchall()
        out = []
        for r in rows:
            text = " ".join([r["term"], r["category"], r["language"] or "", r["meaning"], r["simple_meaning"]])
            score = _cosine(q, _tokens(text))
            if score > 0:
                out.append({"term": r["term"], "category": r["category"], "language": r["language"], "meaning": r["meaning"], "simple_meaning": r["simple_meaning"], "score": round(score, 6), "source": r["source"], "license": r["license"]})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:limit]

    def add_code_component(self, component_key: str, name: str, *, language: str, purpose: str, code: str,
                           framework: Optional[str] = None, platform: Optional[str] = None,
                           inputs: Optional[Iterable[str]] = None, outputs: Optional[Iterable[str]] = None,
                           dependencies: Optional[Iterable[str]] = None, source: str = "user",
                           license: Optional[str] = None, version: Optional[str] = None,
                           tests: Optional[Iterable[str]] = None, metadata: Optional[Dict[str, Any]] = None,
                           confidence: float = 1.0) -> int:
        confidence = _validate_confidence(confidence) or 0.0
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO code_components(component_key,name,language,framework,platform,purpose,inputs_json,outputs_json,dependencies_json,code,source,license,version,tests_json,metadata_json,confidence) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(component_key) DO UPDATE SET name=excluded.name,language=excluded.language,framework=excluded.framework,platform=excluded.platform,purpose=excluded.purpose,inputs_json=excluded.inputs_json,outputs_json=excluded.outputs_json,dependencies_json=excluded.dependencies_json,code=excluded.code,source=excluded.source,license=excluded.license,version=excluded.version,tests_json=excluded.tests_json,metadata_json=excluded.metadata_json,confidence=excluded.confidence",
                (component_key, name, language, framework, platform, purpose, _json(list(inputs or [])), _json(list(outputs or [])), _json(list(dependencies or [])), code, source, license, version, _json(list(tests or [])), _json(metadata or {}), confidence),
            )
            row = self._conn.execute("SELECT id FROM code_components WHERE component_key=?", (component_key,)).fetchone()
            return int(row[0])

    def search_code(self, query: str, *, language: Optional[str] = None, platform: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        q = _tokens(query)
        clauses, params = [], []
        if language:
            clauses.append("language=?"); params.append(language)
        if platform:
            clauses.append("platform=?"); params.append(platform)
        sql = "SELECT * FROM code_components" + ((" WHERE " + " AND ".join(clauses)) if clauses else "")
        rows = self._conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            text = " ".join([r["name"], r["language"], r["framework"] or "", r["platform"] or "", r["purpose"], r["component_key"], r["code"]])
            score = _cosine(q, _tokens(text))
            if score > 0:
                out.append({"component_key": r["component_key"], "name": r["name"], "language": r["language"], "framework": r["framework"], "platform": r["platform"], "purpose": r["purpose"], "dependencies": json.loads(r["dependencies_json"] or "[]"), "code": r["code"], "source": r["source"], "license": r["license"], "version": r["version"], "confidence": r["confidence"], "score": round(score, 6)})
        out.sort(key=lambda x: (x["score"], x["confidence"]), reverse=True)
        return out[:limit]

    def memory_bundle(self, query: str, *, top_k: int = 8) -> Dict[str, Any]:
        """Return a grounded context bundle suitable for a local LLM prompt."""
        memories = self.recall(query, top_k=top_k)
        entities = []
        for eid in self._entity_ids_for_query(query)[:top_k]:
            row = self._conn.execute("SELECT * FROM assoc_entities WHERE id=?", (eid,)).fetchone()
            if row:
                entities.append({"entity_key": row["entity_key"], "name": row["name"], "type": row["entity_type"], "description": row["description"], "confidence": row["confidence"]})
        return {"query": query, "memories": memories, "entities": entities,
                "programming": self.search_programming(query, limit=min(5, top_k)),
                "code_components": self.search_code(query, limit=min(5, top_k))}

    def health(self) -> Dict[str, Any]:
        row = self._conn.execute("PRAGMA integrity_check").fetchone()
        return {"ok": bool(row and row[0] == "ok"), "integrity": row[0] if row else "unknown", **self.stats()}

    def close(self) -> None:
        with self._lock:
            if not self._conn:
                return
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "AssociativeMemory":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
