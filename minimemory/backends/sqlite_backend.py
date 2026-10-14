"""SQLite persistent backend."""

import sqlite3
import threading
from pathlib import Path
from typing import List, Tuple

from .minimemory_backends_base import BaseBackend


class SQLiteBackend(BaseBackend):
    def __init__(self, db_path: str = "memory.db") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._memory_conn = sqlite3.connect(":memory:", timeout=30, check_same_thread=False) if self.db_path == ":memory:" else None
        if self._memory_conn is not None:
            self._memory_conn.row_factory = sqlite3.Row
        self.init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        Path(self.db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS knowledge_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT UNIQUE NOT NULL,
                answer TEXT NOT NULL, usage_count INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)""")

    def learn_batch(self, pairs: List[Tuple[str, str]]) -> int:
        if not pairs:
            return 0
        with self._lock, self._get_conn() as conn:
            conn.executemany("""INSERT INTO knowledge_memory (question, answer) VALUES (?, ?)
                ON CONFLICT(question) DO UPDATE SET answer = excluded.answer""", pairs)
        return len(pairs)

    def get_all_pairs(self) -> List[Tuple[str, str]]:
        with self._lock, self._get_conn() as conn:
            rows = conn.execute("SELECT question, answer FROM knowledge_memory ORDER BY id").fetchall()
            return [(row["question"], row["answer"]) for row in rows]

    def count(self) -> int:
        with self._lock, self._get_conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM knowledge_memory").fetchone()[0])

    def clear(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute("DELETE FROM knowledge_memory")

    def close(self) -> None:
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None

    def __enter__(self) -> "SQLiteBackend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
