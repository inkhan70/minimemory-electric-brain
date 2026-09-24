"""Offline-first persistent memory for local AI applications."""
from __future__ import annotations

import inspect
import json
import re
import shutil
import sqlite3
import threading
from collections import Counter
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_SCHEMA_VERSION = 3


def _tokens(value: str) -> Counter:
    return Counter(_TOKEN_RE.findall(value.casefold()))


def _cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[k] * right.get(k, 0) for k in left)
    nl = sqrt(sum(v * v for v in left.values()))
    nr = sqrt(sum(v * v for v in right.values()))
    return dot / (nl * nr) if nl and nr else 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryQA:
    """Persistent local Q&A memory with optional semantic retrieval.

    The core has no required third-party dependencies.  Semantic search is
    loaded lazily and is offline-first.  ``chat`` can connect any local model
    through a callable, ``generate`` method, or ``respond`` method.

    ``minimemory`` stores/retrieves knowledge; it does not train model weights.
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        model_name: str = _DEFAULT_MODEL,
        *,
        semantic: bool = True,
        local_files_only: bool = True,
        min_auto_learn_confidence: float = 0.0,
        max_context_items: int = 5,
    ):
        if not 0 <= min_auto_learn_confidence <= 1:
            raise ValueError("min_auto_learn_confidence must be between 0 and 1")
        if max_context_items < 1:
            raise ValueError("max_context_items must be >= 1")
        self.db_path = str(db_path)
        self.model_name = str(model_name)
        self.semantic_requested = bool(semantic)
        self.local_files_only = bool(local_files_only)
        self.min_auto_learn_confidence = float(min_auto_learn_confidence)
        self.max_context_items = int(max_context_items)
        self._lock = threading.RLock()
        self._closed = False
        self._memory_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", timeout=30, check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        self._model = None
        self._semantic_available = False
        self._semantic_attempted = False
        self._questions: List[str] = []
        self._answers: List[str] = []
        self._ids: List[int] = []
        self._metadata: List[Dict[str, Any]] = []
        self._sources: List[str] = []
        self._fallback_vectors: List[Counter] = []
        self._embeddings = None
        self._init_db()
        self.reload_cache()

    def _get_conn(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("MemoryQA is closed")
        if self._memory_conn is not None:
            conn = self._memory_conn
        else:
            Path(self.db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        if self.db_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT UNIQUE NOT NULL,
                    answer TEXT NOT NULL,
                    usage_count INTEGER NOT NULL DEFAULT 1,
                    source TEXT NOT NULL DEFAULT 'user',
                    confidence REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(knowledge_memory)")}
            migrations = {
                "source": "ALTER TABLE knowledge_memory ADD COLUMN source TEXT NOT NULL DEFAULT 'user'",
                "confidence": "ALTER TABLE knowledge_memory ADD COLUMN confidence REAL",
                "metadata_json": "ALTER TABLE knowledge_memory ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'",
                "updated_at": "ALTER TABLE knowledge_memory ADD COLUMN updated_at TIMESTAMP",
            }
            for col, sql in migrations.items():
                if col not in cols:
                    conn.execute(sql)
            conn.execute("UPDATE knowledge_memory SET updated_at=COALESCE(updated_at, created_at, CURRENT_TIMESTAMP)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_updated ON knowledge_memory(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_source ON knowledge_memory(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_confidence ON knowledge_memory(confidence)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_key TEXT UNIQUE NOT NULL,
                    canonical_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL DEFAULT 'unknown',
                    description TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    confidence REAL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(canonical_name)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_aliases (
                    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    alias TEXT NOT NULL,
                    UNIQUE(entity_id, alias)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_alias ON entity_aliases(alias)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    predicate TEXT NOT NULL,
                    object_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
                    object_text TEXT,
                    source TEXT NOT NULL DEFAULT 'user',
                    confidence REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_subject ON relations(subject_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_predicate ON relations(predicate)")
            conn.execute("CREATE TABLE IF NOT EXISTS minimemory_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("INSERT OR REPLACE INTO minimemory_meta(key,value) VALUES ('schema_version',?)", (str(_SCHEMA_VERSION),))

    @staticmethod
    def _validate_pair(question: str, answer: str) -> Tuple[str, str]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be a non-empty string")
        return question.strip(), answer.strip()

    @staticmethod
    def _validate_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if metadata is None:
            return {}
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")
        try:
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable") from exc
        return metadata

    @staticmethod
    def _validate_confidence(confidence: Optional[float]) -> Optional[float]:
        if confidence is None:
            return None
        try:
            value = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be a number between 0 and 1") from exc
        if not 0 <= value <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return value

    def _ensure_semantic_model(self) -> bool:
        if not self.semantic_requested or self._semantic_attempted:
            return self._semantic_available
        self._semantic_attempted = True
        try:
            from sentence_transformers import SentenceTransformer  # lazy optional import
            kwargs: Dict[str, Any] = {}
            # Different Sentence Transformers releases expose different loading args.
            if self.local_files_only:
                kwargs["local_files_only"] = True
            self._model = SentenceTransformer(self.model_name, **kwargs)
            self._semantic_available = True
        except Exception:
            self._model = None
            self._semantic_available = False
        return self._semantic_available

    @property
    def semantic_available(self) -> bool:
        return self._ensure_semantic_model()

    def reload_cache(self) -> None:
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, question, answer, source, metadata_json FROM knowledge_memory ORDER BY id"
            ).fetchall()
            self._ids = [int(row["id"]) for row in rows]
            self._questions = [row["question"] for row in rows]
            self._answers = [row["answer"] for row in rows]
            self._metadata = []
            self._sources = [str(row["source"]) for row in rows]
            for row in rows:
                try:
                    value = json.loads(row["metadata_json"] or "{}")
                    self._metadata.append(value if isinstance(value, dict) else {})
                except (TypeError, ValueError, json.JSONDecodeError):
                    self._metadata.append({})
            self._fallback_vectors = [_tokens(q) for q in self._questions]
            self._embeddings = None
            if self._ensure_semantic_model() and self._questions:
                self._embeddings = self._model.encode(
                    self._questions, convert_to_tensor=True,
                    normalize_embeddings=True, show_progress_bar=False,
                )

    def learn(self, question: str, answer: str, *, source: str = "user",
              confidence: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        return self.learn_batch([(question, answer)], source=source, confidence=confidence, metadata=metadata) == 1

    def learn_batch(self, pairs: Iterable[Tuple[str, str]], *, source: str = "user",
                    confidence: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
        normalized = [self._validate_pair(q, a) for q, a in pairs]
        if not normalized:
            return 0
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        confidence = self._validate_confidence(confidence)
        metadata = self._validate_metadata(metadata)
        metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        with self._lock, self._get_conn() as conn:
            for question, answer in normalized:
                conn.execute("""
                    INSERT INTO knowledge_memory
                      (question, answer, usage_count, source, confidence, metadata_json,
                       created_at, updated_at)
                    VALUES (?, ?, 1, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(question) DO UPDATE SET
                      answer=excluded.answer,
                      usage_count=knowledge_memory.usage_count + 1,
                      source=excluded.source,
                      confidence=excluded.confidence,
                      metadata_json=excluded.metadata_json,
                      updated_at=CURRENT_TIMESTAMP
                """, (question, answer, source.strip(), confidence, metadata_json))
        self.reload_cache()
        return len(normalized)

    def learn_from_ai(self, question: str, answer: str, *, confidence: Optional[float] = None,
                      approved: bool = False, source: str = "ai",
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store an AI answer only after approval or a configured confidence threshold."""
        confidence = self._validate_confidence(confidence)
        if not approved and (confidence is None or confidence < self.min_auto_learn_confidence):
            return False
        return self.learn(question, answer, source=source, confidence=confidence, metadata=metadata)

    def get(self, memory_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_conn() as conn:
            row = conn.execute("SELECT * FROM knowledge_memory WHERE id=?", (int(memory_id),)).fetchone()
            return self._row_to_dict(row) if row else None

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        return {"id": int(row["id"]), "question": row["question"], "answer": row["answer"],
                "usage_count": int(row["usage_count"]), "source": row["source"],
                "confidence": row["confidence"], "metadata": metadata,
                "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def get_all_pairs(self) -> List[Tuple[str, str]]:
        with self._lock, self._get_conn() as conn:
            rows = conn.execute("SELECT question, answer FROM knowledge_memory ORDER BY id").fetchall()
            return [(row["question"], row["answer"]) for row in rows]

    def count(self) -> int:
        with self._lock, self._get_conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM knowledge_memory").fetchone()[0])

    def stats(self) -> Dict[str, Any]:
        with self._lock, self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS count, COALESCE(SUM(usage_count),0) AS uses, MIN(created_at) AS first, MAX(updated_at) AS last FROM knowledge_memory").fetchone()
            return {"count": int(row["count"]), "total_usage": int(row["uses"]),
                    "first_created_at": row["first"], "last_updated_at": row["last"],
                    "semantic_available": self.semantic_available, "db_path": self.db_path}

    def clear(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute("DELETE FROM knowledge_memory")
        self.reload_cache()

    def delete(self, memory_id: int) -> bool:
        with self._lock, self._get_conn() as conn:
            cur = conn.execute("DELETE FROM knowledge_memory WHERE id=?", (int(memory_id),))
        if cur.rowcount:
            self.reload_cache()
            return True
        return False

    forget = delete

    def _scores(self, question: str) -> List[float]:
        if self._ensure_semantic_model() and self._questions:
            if self._embeddings is None:
                self._embeddings = self._model.encode(self._questions, convert_to_tensor=True,
                                                      normalize_embeddings=True, show_progress_bar=False)
            qv = self._model.encode(question, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            try:
                from sentence_transformers import util
                return [float(x) for x in util.cos_sim(qv, self._embeddings)[0]]
            except Exception:
                return []
        q = _tokens(question)
        return [_cosine(q, v) for v in self._fallback_vectors]

    def search(self, question: str, *, top_k: int = 5, min_score: float = 0.0,
               source: Optional[str] = None) -> List[Dict[str, Any]]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        with self._lock:
            if not self._questions:
                return []
            scores = self._scores(question.strip())
            if len(scores) != len(self._questions):
                return []
            indices = [i for i, score in enumerate(scores) if score >= min_score and (source is None or self._sources[i] == source)]
            indices.sort(key=lambda i: (-scores[i], self._ids[i]))
            selected = indices[:top_k]
            results = []
            for rank, i in enumerate(selected, 1):
                results.append({"id": self._ids[i], "question": self._questions[i], "answer": self._answers[i],
                                "confidence": round(float(max(0.0, min(1.0, scores[i]))), 4),
                                "rank": rank, "source": self._sources[i], "metadata": self._metadata[i]})
            return results

    def ask(self, question: str, threshold: float = 0.55, *, top_k: int = 5) -> Dict[str, Any]:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        hits = self.search(question, top_k=max(1, top_k))
        if not hits:
            return {"question": question, "answer": None, "confidence": 0.0,
                    "status": "EMPTY_MEMORY", "matches": []}
        best = hits[0]
        result = {"question": question, "answer": None, "confidence": best["confidence"],
                  "status": "LOW_CONFIDENCE", "matches": hits}
        if best["confidence"] >= threshold:
            result.update(answer=best["answer"], status="SUCCESS")
            self._increment_usage(best["id"])
        return result

    def _increment_usage(self, memory_id: int) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute("UPDATE knowledge_memory SET usage_count=usage_count+1 WHERE id=?", (int(memory_id),))

    def chat(self, question: str, generator: Callable[..., Any], *, threshold: float = 0.75,
             auto_learn: bool = True, approve: bool = False,
             confidence: Optional[float] = None, context_items: Optional[int] = None,
             source: str = "ai", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrieve memory, call a local AI if needed, and optionally learn its answer.

        The generator may return a string or ``{"answer": ..., "confidence": ...}``.
        Its confidence is treated as an application/model signal, not fact checking.
        """
        if not callable(generator) and not hasattr(generator, "generate") and not hasattr(generator, "respond"):
            raise TypeError("generator must be callable or expose generate()/respond()")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        limit = context_items or self.max_context_items
        hits = self.search(question, top_k=limit)
        best = hits[0] if hits else None
        if best and best["confidence"] >= threshold:
            self._increment_usage(best["id"])
            return {"answer": best["answer"], "source": "memory", "confidence": best["confidence"],
                    "matches": hits, "learned": False, "memory_id": best["id"]}
        answer_raw = self._call_generator(generator, question, hits)
        model_confidence = confidence
        if isinstance(answer_raw, dict):
            answer = answer_raw.get("answer")
            if model_confidence is None:
                model_confidence = answer_raw.get("confidence")
            response_meta = answer_raw.get("metadata")
        else:
            answer = answer_raw
            response_meta = None
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("AI generator returned an empty/non-string answer")
        model_confidence = self._validate_confidence(model_confidence)
        learned = False
        combined_meta: Dict[str, Any] = {"retrieved_context": hits}
        if metadata:
            combined_meta.update(self._validate_metadata(metadata))
        if isinstance(response_meta, dict):
            combined_meta.update(response_meta)
        if auto_learn:
            learned = self.learn_from_ai(question, answer, confidence=model_confidence,
                                         approved=approve, source=source, metadata=combined_meta)
        return {"answer": answer.strip(), "source": "ai", "confidence": model_confidence or 0.0,
                "matches": hits, "learned": learned, "memory_id": None}

    @staticmethod
    def _call_generator(generator: Any, question: str, context: List[Dict[str, Any]]) -> Any:
        fn = generator.generate if hasattr(generator, "generate") else generator.respond if hasattr(generator, "respond") else generator
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
            if accepts_kwargs or len(params) >= 2:
                return fn(question, context)
        except (TypeError, ValueError):
            pass
        return fn(question)

    def upsert_entity(self, entity_key: str, canonical_name: str, *, entity_type: str = "unknown",
                      aliases: Optional[Iterable[str]] = None, description: str = "",
                      confidence: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Create/update a canonical entity and its aliases."""
        if not entity_key.strip() or not canonical_name.strip():
            raise ValueError("entity_key and canonical_name must be non-empty")
        confidence = self._validate_confidence(confidence)
        metadata = self._validate_metadata(metadata)
        aliases = set(a.strip() for a in (aliases or []) if isinstance(a, str) and a.strip())
        aliases.add(canonical_name.strip())
        with self._lock, self._get_conn() as conn:
            conn.execute("""INSERT INTO entities(entity_key,canonical_name,entity_type,description,metadata_json,confidence,updated_at)
                           VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP)
                           ON CONFLICT(entity_key) DO UPDATE SET canonical_name=excluded.canonical_name,
                           entity_type=excluded.entity_type,description=excluded.description,
                           metadata_json=excluded.metadata_json,confidence=excluded.confidence,updated_at=CURRENT_TIMESTAMP""",
                         (entity_key.strip(), canonical_name.strip(), entity_type.strip() or "unknown",
                          description or "", json.dumps(metadata, ensure_ascii=False, sort_keys=True), confidence))
            row = conn.execute("SELECT id FROM entities WHERE entity_key=?", (entity_key.strip(),)).fetchone()
            entity_id = int(row[0])
            for alias in aliases:
                conn.execute("INSERT OR IGNORE INTO entity_aliases(entity_id,alias) VALUES(?,?)", (entity_id, alias))
            return entity_id

    def find_entities(self, name: str, *, entity_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        if not name.strip() or limit < 1:
            return []
        with self._lock, self._get_conn() as conn:
            params: List[Any] = [name.strip()]
            sql = """SELECT DISTINCT e.* FROM entities e LEFT JOIN entity_aliases a ON a.entity_id=e.id
                     WHERE (e.canonical_name=? OR a.alias=?)"""
            params.append(name.strip())
            if entity_type:
                sql += " AND e.entity_type=?"
                params.append(entity_type)
            sql += " ORDER BY e.id LIMIT ?"
            params.append(int(limit))
            rows = conn.execute(sql, params).fetchall()
            out=[]
            for r in rows:
                try: md=json.loads(r["metadata_json"] or "{}")
                except Exception: md={}
                aliases=[x[0] for x in conn.execute("SELECT alias FROM entity_aliases WHERE entity_id=? ORDER BY alias", (r["id"],)).fetchall()]
                out.append({"id":int(r["id"]),"entity_key":r["entity_key"],"canonical_name":r["canonical_name"],
                            "entity_type":r["entity_type"],"description":r["description"],"aliases":aliases,
                            "confidence":r["confidence"],"metadata":md})
            return out

    def add_relation(self, subject: Union[int, str], predicate: str, object_value: Union[int, str], *,
                     source: str = "user", confidence: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a graph edge. Object can be an entity id or free text."""
        confidence = self._validate_confidence(confidence)
        metadata = self._validate_metadata(metadata)
        with self._lock, self._get_conn() as conn:
            def resolve(value: Union[int, str]) -> Optional[int]:
                if isinstance(value, int): return int(value)
                row = conn.execute("SELECT id FROM entities WHERE entity_key=?", (value,)).fetchone()
                return int(row[0]) if row else None
            sid = resolve(subject)
            if sid is None: raise ValueError("subject entity not found")
            oid = resolve(object_value)
            object_text = None if oid is not None else str(object_value)
            cur = conn.execute("""INSERT INTO relations(subject_id,predicate,object_id,object_text,source,confidence,metadata_json)
                                 VALUES(?,?,?,?,?,?,?)""",
                               (sid, predicate.strip(), oid, object_text, source.strip(), confidence,
                                json.dumps(metadata, ensure_ascii=False, sort_keys=True)))
            return int(cur.lastrowid)

    def graph(self, entity: Union[int, str], *, depth: int = 1, limit: int = 50) -> Dict[str, Any]:
        """Return a bounded neighborhood around an entity."""
        if depth < 1 or depth > 5: raise ValueError("depth must be between 1 and 5")
        with self._lock, self._get_conn() as conn:
            row = conn.execute("SELECT id,entity_key,canonical_name,entity_type FROM entities WHERE id=? OR entity_key=?",
                               (entity if isinstance(entity,int) else -1, entity if isinstance(entity,str) else "")).fetchone()
            if not row: return {"entity": None, "relations": []}
            seen={int(row[0])}; frontier={int(row[0])}; edges=[]
            for _ in range(depth):
                if not frontier or len(edges)>=limit: break
                placeholders=','.join('?' for _ in frontier)
                rows=conn.execute(f"SELECT r.*, s.entity_key AS sk, s.canonical_name AS sn, o.entity_key AS ok, o.canonical_name AS oname FROM relations r JOIN entities s ON s.id=r.subject_id LEFT JOIN entities o ON o.id=r.object_id WHERE r.subject_id IN ({placeholders}) LIMIT ?", [*frontier, max(0,limit-len(edges))]).fetchall()
                nxt=set()
                for r in rows:
                    edges.append({"subject":r["sk"],"subject_name":r["sn"],"predicate":r["predicate"],
                                  "object":r["ok"] or r["object_text"],"object_name":r["oname"] or r["object_text"],
                                  "source":r["source"],"confidence":r["confidence"]})
                    if r["object_id"] and int(r["object_id"]) not in seen:
                        nxt.add(int(r["object_id"])); seen.add(int(r["object_id"]))
                frontier=nxt
            return {"entity":{"id":int(row[0]),"entity_key":row[1],"canonical_name":row[2],"entity_type":row[3]},"relations":edges}

    def refine_text(self, text: str, *, alphabet_only: bool = False, max_passes: int = 2) -> Dict[str, Any]:
        from .pipeline import TextPipeline
        return TextPipeline(alphabet_only=alphabet_only).process(text, max_passes=max_passes)

    def save_pack(self, folder: str, **kwargs: Any) -> str:
        from .packs import KnowledgePack
        return KnowledgePack.from_memory(self, **kwargs).save(folder)

    def load_pack(self, source: str, token: Optional[str] = None) -> int:
        from .packs import KnowledgePack
        path = Path(source).expanduser()
        pack = KnowledgePack.from_file(str(path)) if path.exists() else KnowledgePack.from_hub(source, token=token)
        return pack.apply_to(self)

    def push_pack(self, repo_id: str, private: bool = False, token: Optional[str] = None, **kwargs: Any) -> str:
        from .packs import KnowledgePack
        return KnowledgePack.from_memory(self, **kwargs).push_to_hub(repo_id, private=private, token=token)

    def backup(self, destination: str) -> str:
        """Create a consistent SQLite backup and return its path."""
        if self.db_path == ":memory:":
            raise ValueError("cannot file-backup an in-memory database")
        target = Path(destination).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with self._lock:
            src = self._get_conn()
            try:
                dst = sqlite3.connect(str(tmp))
                try:
                    src.backup(dst)
                finally:
                    dst.close()
            finally:
                src.close()
            tmp.replace(target)
        return str(target)

    def health(self) -> Dict[str, Any]:
        with self._lock, self._get_conn() as conn:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        return {"ok": integrity == "ok", "integrity": integrity, "count": self.count(),
                "semantic_available": self.semantic_available}

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._memory_conn is not None:
                self._memory_conn.close()
                self._memory_conn = None
            self._model = None
            self._embeddings = None
            self._questions.clear()
            self._answers.clear()
            self._ids.clear()
            self._metadata.clear()
            self._sources.clear()
            self._fallback_vectors.clear()

    def __enter__(self) -> "MemoryQA":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
