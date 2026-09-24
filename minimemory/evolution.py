"""Small, deterministic memory-evolution engine.

It learns aggregate topic strength only when behavior tracking is explicitly
enabled. It never stores sensitive query text and never promotes sensitive
behavior into memory.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryEvolution:
    def __init__(self, conn: sqlite3.Connection, *, enabled: bool = False) -> None:
        self.conn = conn
        self.enabled = bool(enabled)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS behavior_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            privacy_class TEXT NOT NULL,
            topic TEXT NOT NULL,
            query_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_behavior_topic ON behavior_events(topic);
        CREATE INDEX IF NOT EXISTS idx_behavior_created ON behavior_events(created_at);
        CREATE TABLE IF NOT EXISTS memory_evolution (
            topic TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            observation_count INTEGER NOT NULL DEFAULT 0,
            strength REAL NOT NULL DEFAULT 0,
            last_seen TEXT NOT NULL
        );
        """)
        self.conn.commit()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def observe(self, observation: Any) -> Dict[str, Any]:
        if not self.enabled or not observation.persist:
            return {"stored": False, "evolved": False, "category": observation.category, "privacy_class": observation.privacy_class}
        now = _now()
        with self.conn:
            self.conn.execute(
                "INSERT INTO behavior_events(category,privacy_class,topic,query_hash,created_at) VALUES(?,?,?,?,?)",
                (observation.category, observation.privacy_class, observation.topic, observation.query_hash, now),
            )
            row = self.conn.execute("SELECT observation_count FROM memory_evolution WHERE topic=?", (observation.topic,)).fetchone()
            count = int(row[0]) + 1 if row else 1
            strength = min(1.0, 1.0 - math.exp(-count / 5.0))
            self.conn.execute(
                "INSERT INTO memory_evolution(topic,category,observation_count,strength,last_seen) VALUES(?,?,?,?,?) "
                "ON CONFLICT(topic) DO UPDATE SET category=excluded.category, observation_count=excluded.observation_count, strength=excluded.strength, last_seen=excluded.last_seen",
                (observation.topic, observation.category, count, strength, now),
            )
        return {"stored": True, "evolved": True, "category": observation.category, "privacy_class": observation.privacy_class, "topic": observation.topic, "strength": strength}

    def top_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        rows = self.conn.execute("SELECT topic,category,observation_count,strength,last_seen FROM memory_evolution ORDER BY strength DESC, last_seen DESC LIMIT ?", (int(limit),)).fetchall()
        return [{"topic": r[0], "category": r[1], "observation_count": int(r[2]), "strength": float(r[3]), "last_seen": r[4]} for r in rows]

    def status(self) -> Dict[str, Any]:
        count = int(self.conn.execute("SELECT COUNT(*) FROM behavior_events").fetchone()[0])
        topics = int(self.conn.execute("SELECT COUNT(*) FROM memory_evolution").fetchone()[0])
        return {"enabled": self.enabled, "stored_events": count, "evolved_topics": topics, "top_topics": self.top_topics(10)}

    def clear(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM behavior_events")
            self.conn.execute("DELETE FROM memory_evolution")
