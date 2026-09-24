"""Privacy-aware behavior classification and storage policy.

Behavior observation is opt-in. Sensitive observations are classified in-memory
and are never persisted by the default policy. Exact sensitive queries are not
stored or exported.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class BehaviorCategory(str, Enum):
    LEARNING = "learning"
    PROGRAMMING = "programming"
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    WRITING = "writing"
    PRODUCTIVITY = "productivity"
    SHOPPING = "shopping"
    TRAVEL = "travel"
    ENTERTAINMENT = "entertainment"
    GENERAL_RESEARCH = "general_research"
    TEMPORARY_INTEREST = "temporary_interest"
    SENSITIVE = "sensitive"
    UNKNOWN = "unknown"


class PrivacyClass(str, Enum):
    NORMAL = "normal"
    TEMPORARY = "temporary"
    SENSITIVE = "sensitive"


@dataclass(frozen=True)
class BehaviorObservation:
    category: str
    privacy_class: str
    topic: str
    query_hash: str
    persist: bool
    evolve: bool


_DEFAULT_SENSITIVE_PATTERNS = (
    r"\bpassword\b", r"\bapi[ _-]?key\b", r"\bsecret\b", r"\btoken\b",
    r"\bcredit[ _-]?card\b", r"\bssn\b", r"\bmedical\b", r"\bdiagnos(?:is|tic)\b",
    r"\bnsfw\b", r"\badult[ _-]?content\b",
)
_CATEGORY_TERMS: Dict[str, tuple[str, ...]] = {
    BehaviorCategory.PROGRAMMING.value: ("python", "javascript", "typescript", "java", "code", "coding", "api", "sqlite", "git", "github", "fastapi", "django", "android", "termux"),
    BehaviorCategory.TECHNOLOGY.value: ("computer", "android", "linux", "windows", "database", "software", "hardware", "server", "network"),
    BehaviorCategory.SCIENCE.value: ("science", "physics", "chemistry", "biology", "astronomy", "research"),
    BehaviorCategory.WRITING.value: ("write", "writing", "grammar", "essay", "document", "rewrite", "translate"),
    BehaviorCategory.PRODUCTIVITY.value: ("task", "todo", "calendar", "schedule", "productivity", "plan"),
    BehaviorCategory.SHOPPING.value: ("buy", "price", "shop", "product", "laptop", "phone", "order"),
    BehaviorCategory.TRAVEL.value: ("travel", "hotel", "flight", "airport", "tourism", "trip"),
    BehaviorCategory.ENTERTAINMENT.value: ("movie", "music", "game", "gaming", "song", "show"),
    BehaviorCategory.LEARNING.value: ("learn", "tutorial", "course", "lesson", "how to", "explain"),
}
_STOP = {"the", "and", "for", "with", "what", "where", "when", "how", "why", "is", "are", "to", "of", "a", "an", "in", "on", "my", "your", "this", "that"}


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def classify_behavior(query: str, *, sensitive_patterns: Optional[tuple[str, ...]] = None) -> BehaviorObservation:
    """Classify a query without retaining its raw text."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    normalized = " ".join(query.casefold().split())
    patterns = sensitive_patterns or _DEFAULT_SENSITIVE_PATTERNS
    if any(re.search(pattern, normalized) for pattern in patterns):
        return BehaviorObservation(BehaviorCategory.SENSITIVE.value, PrivacyClass.SENSITIVE.value, "sensitive", _hash_query(normalized), False, False)
    scores = {category: sum(1 for term in terms if term in normalized) for category, terms in _CATEGORY_TERMS.items()}
    best = max(scores, key=scores.get) if scores else BehaviorCategory.UNKNOWN.value
    score = scores.get(best, 0)
    if score == 0:
        category = BehaviorCategory.GENERAL_RESEARCH.value
    else:
        category = best
    topic_tokens = [t for t in re.findall(r"[\w'-]+", normalized) if t not in _STOP and len(t) > 2]
    topic = " ".join(dict.fromkeys(topic_tokens[:3])) or category
    privacy = PrivacyClass.TEMPORARY.value if category in {BehaviorCategory.GENERAL_RESEARCH.value, BehaviorCategory.TEMPORARY_INTEREST.value} else PrivacyClass.NORMAL.value
    return BehaviorObservation(category, privacy, topic, _hash_query(normalized), privacy == PrivacyClass.NORMAL.value, privacy == PrivacyClass.NORMAL.value)
