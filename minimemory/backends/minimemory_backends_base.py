"""Abstract storage backend contract."""

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseBackend(ABC):
    @abstractmethod
    def init_db(self) -> None:
        """Initialize database schema or storage structures."""

    @abstractmethod
    def learn_batch(self, pairs: List[Tuple[str, str]]) -> int:
        """Ingest or update pairs and return the number processed."""

    @abstractmethod
    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """Retrieve all stored pairs."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored pairs."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all stored pairs."""
