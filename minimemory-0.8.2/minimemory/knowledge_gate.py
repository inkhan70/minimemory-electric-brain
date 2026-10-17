"""Evidence-aware claim admission for minimemory.

Claims are append-only, source-attributed, and can remain contested. Scores are
heuristics for review/admission, not truth probabilities or fact verification.
"""
from __future__ import annotations
import hashlib, json, re, sqlite3, threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_NEGATIONS = {"not", "no", "never", "cannot", "can't", "doesn't", "isn't", "won't", "without"}
_WORDS = re.compile(r"[\w']+", re.UNICODE)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
def _norm(s: str) -> str:
    return " ".join(s.casefold().split())
def _hash(s: str) -> str:
    return hashlib.sha256(_norm(s).encode("utf-8")).hexdigest()

class KnowledgeGate:
    """SQLite claim ledger with admission threshold and basic conflict flagging."""
    def __init__(self, db_path: str, *, admission_threshold: float = 0.60, conflict_detector: Optional[Callable[[str, str], bool]] = None):
        if not 0 <= admission_threshold <= 1: raise ValueError("admission_threshold must be between 0 and 1")
        self.db_path = str(db_path); self.admission_threshold = float(admission_threshold)
        # Optional semantic detector supplied by the host app; no model is bundled.
        self.conflict_detector = conflict_detector
        Path(self.db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS claims (
          id INTEGER PRIMARY KEY AUTOINCREMENT, claim TEXT NOT NULL, normalized TEXT NOT NULL,
          claim_hash TEXT NOT NULL, claim_type TEXT NOT NULL, source TEXT NOT NULL,
          evidence_polarity TEXT NOT NULL, admission_score REAL NOT NULL,
          confidence REAL, verification_status TEXT NOT NULL, context_json TEXT NOT NULL,
          created_at TEXT NOT NULL, UNIQUE(claim_hash, source)
        );
        CREATE INDEX IF NOT EXISTS idx_claim_norm ON claims(normalized);
        CREATE INDEX IF NOT EXISTS idx_claim_status ON claims(verification_status);
        CREATE TABLE IF NOT EXISTS claim_relations (
          id INTEGER PRIMARY KEY AUTOINCREMENT, claim_id INTEGER NOT NULL,
          related_claim_id INTEGER NOT NULL, relation TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL, UNIQUE(claim_id, related_claim_id, relation)
        );
        """)
        self._conn.commit()

    @staticmethod
    def _heuristic_conflict(a: str, b: str) -> bool:
        """Only flag likely polarity mismatch; never claim semantic proof."""
        wa, wb = set(_WORDS.findall(a.casefold())), set(_WORDS.findall(b.casefold()))
        if not wa or not wb: return False
        overlap = len((wa - _NEGATIONS) & (wb - _NEGATIONS)) / max(1, len((wa - _NEGATIONS) | (wb - _NEGATIONS)))
        return overlap >= 0.55 and bool((wa & _NEGATIONS) != (wb & _NEGATIONS))

    def admit(self, claim: str, *, source: str = "user", claim_type: str = "claim",
              evidence_polarity: str = "neutral", admission_score: float = 0.5,
              confidence: Optional[float] = None, verification_status: Optional[str] = None,
              context: Optional[Dict[str, Any]] = None, force_store: bool = False) -> Dict[str, Any]:
        if not isinstance(claim, str) or not claim.strip(): raise ValueError("claim must be non-empty text")
        if not isinstance(source, str) or not source.strip(): raise ValueError("source must be a non-empty string")
        source = source.strip()
        if evidence_polarity not in {"supports", "opposes", "neutral"}: raise ValueError("invalid evidence_polarity")
        if not 0 <= float(admission_score) <= 1: raise ValueError("admission_score must be between 0 and 1")
        if confidence is not None and not 0 <= float(confidence) <= 1: raise ValueError("confidence must be between 0 and 1")
        normalized = _norm(claim); digest = _hash(claim); now = _now()
        status = verification_status or ("candidate" if confidence is None else "unverified")
        if status not in {"candidate", "unverified", "supported", "contested", "rejected", "superseded"}:
            raise ValueError("invalid verification_status")
        if not force_store and admission_score < self.admission_threshold:
            return {"stored": False, "status": "below_threshold", "admission_score": admission_score,
                    "threshold": self.admission_threshold, "claim": claim}
        with self._lock, self._conn:
            prior = self._conn.execute("SELECT * FROM claims WHERE claim_hash=? AND source=?", (digest, source)).fetchone()
            if prior:
                return {"stored": True, "duplicate": True, "id": int(prior["id"]), "status": prior["verification_status"], "related_claims": []}
            rows = self._conn.execute("SELECT id, claim FROM claims WHERE verification_status!='rejected' ORDER BY id DESC LIMIT 500").fetchall()
            conflicts = []
            for r in rows:
                try:
                    is_conflict = (bool(self.conflict_detector(claim, r["claim"]))
                                   if self.conflict_detector is not None
                                   else self._heuristic_conflict(claim, r["claim"]))
                except Exception:
                    # A failing optional model must not block safe memory admission.
                    is_conflict = self._heuristic_conflict(claim, r["claim"])
                if is_conflict:
                    conflicts.append(r)
            # Only auto-downgrade open statuses; never overwrite an explicit
            # "rejected"/"superseded" decision made by the caller.
            if conflicts and status in {"candidate", "unverified", "supported"}:
                status = "contested"
            cur = self._conn.execute("""INSERT INTO claims(claim,normalized,claim_hash,claim_type,source,evidence_polarity,
              admission_score,confidence,verification_status,context_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
              (claim.strip(), normalized, digest, claim_type, source, evidence_polarity, float(admission_score), confidence,
               status, json.dumps(context or {}, ensure_ascii=False, sort_keys=True), now))
            cid = int(cur.lastrowid)
            for r in conflicts:
                self._conn.execute("INSERT OR IGNORE INTO claim_relations(claim_id,related_claim_id,relation,note,created_at) VALUES(?,?,?,?,?)",
                  (cid, int(r["id"]), "POTENTIAL_CONTRADICTION", "Conflict detector flagged this pair; requires contextual review" if self.conflict_detector else "Lexical polarity heuristic; requires contextual review", now))
                self._conn.execute("UPDATE claims SET verification_status='contested' WHERE id=? AND verification_status IN ('candidate','unverified','supported')", (int(r["id"]),))
            return {"stored": True, "duplicate": False, "id": cid, "status": status,
                    "related_claims": [int(r["id"]) for r in conflicts],
                    "note": "Potential conflicts are flags for review, not proof of contradiction."}

    def search(self, query: str, *, limit: int = 10) -> list[Dict[str, Any]]:
        terms = set(_WORDS.findall(query.casefold()))
        if not terms: return []
        with self._lock:
            rows = self._conn.execute("SELECT * FROM claims ORDER BY id DESC LIMIT 1000").fetchall()
            out=[]
            for r in rows:
                words=set(_WORDS.findall(r["claim"].casefold()))
                score=len(terms & words)/max(1,len(terms | words))
                if score > 0:
                    item=dict(r)
                    try: item["context"] = json.loads(item.pop("context_json"))
                    except (TypeError, ValueError): item["context"] = {}
                    item["retrieval_score"] = round(score,4); out.append(item)
            return sorted(out,key=lambda x:x["retrieval_score"],reverse=True)[:max(1,limit)]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
