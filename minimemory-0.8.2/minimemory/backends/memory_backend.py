"""Fast in-memory backend."""

import threading
from typing import Dict, List, Tuple

from .minimemory_backends_base import BaseBackend


class MemoryBackend(BaseBackend):
    def __init__(self) -> None:
        self._data: Dict[str, str] = {}
        self._lock = threading.RLock()

    def init_db(self) -> None:
        return None

    def learn_batch(self, pairs: List[Tuple[str, str]]) -> int:
        with self._lock:
            for question, answer in pairs:
                self._data[question] = answer
            return len(pairs)

    def get_all_pairs(self) -> List[Tuple[str, str]]:
        with self._lock:
            return list(self._data.items())

    def count(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
