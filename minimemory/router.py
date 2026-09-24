"""Deterministic query routing for the local brain.

The router does not call an LLM.  It classifies the input and chooses the
smallest useful local subsystem first: Q&A, associative memory, or the
programming/code knowledge store.  It also exposes whether the user supplied
ordinary text or an apparent code block so applications can adapt retrieval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class QueryKind(str, Enum):
    TEXT = "text"
    CODE = "code"
    MIXED = "mixed"


class QueryRoute(str, Enum):
    QNA = "qna"
    MEMORY = "memory"
    CODE = "code"
    PROGRAMMING = "programming"
    UNKNOWN = "unknown"


_CODE_FENCE = re.compile(r"```(?:[\w+#.-]+)?\s*.*?```", re.DOTALL)
_CODE_LINE = re.compile(r"(^|\n)\s*(?:def |class |import |from |const |let |var |function |public class |#include |SELECT |INSERT |UPDATE |<html|<script|#!/)" , re.I)
_CODE_SYMBOLS = re.compile(r"(?:=>|\b\w+\s*\([^\n]*\)\s*\{|\b\w+\s*=\s*[^\n]+;|[{};]{2,})")
_PROGRAMMING_TERMS = {
    "python", "javascript", "typescript", "java", "kotlin", "swift", "rust", "cpp", "c++",
    "html", "css", "sql", "sqlite", "api", "code", "coding", "program", "programming",
    "function", "class", "variable", "bug", "debug", "developer", "git", "github", "android",
    "termux", "frontend", "backend", "database", "framework", "algorithm", "script",
}
_MEMORY_TERMS = {
    "remember", "memory", "recall", "stored", "learned", "knowledge", "previous", "earlier",
    "forget", "associate", "related", "where did i", "what did i tell", "my project",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_+#.-]+", text.casefold()))


def classify_query(query: str) -> Dict[str, Any]:
    """Return deterministic, explainable query features without storing text."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    text = query.strip()
    tokens = _tokens(text)
    fenced = bool(_CODE_FENCE.search(text))
    line_code = bool(_CODE_LINE.search(text))
    symbol_code = bool(_CODE_SYMBOLS.search(text))
    code_score = int(fenced) * 5 + int(line_code) * 3 + int(symbol_code) * 2
    programming_hits = sorted(tokens & _PROGRAMMING_TERMS)
    memory_hits = sorted(tokens & _MEMORY_TERMS)
    if code_score >= 5:
        kind = QueryKind.CODE
    elif code_score >= 2 and programming_hits:
        kind = QueryKind.MIXED
    else:
        kind = QueryKind.TEXT
    return {
        "kind": kind.value,
        "code_score": code_score,
        "has_code_block": fenced,
        "programming_terms": programming_hits,
        "memory_terms": memory_hits,
        "looks_like_memory_request": bool(memory_hits),
        "looks_like_programming": bool(programming_hits),
    }


def choose_route(features: Dict[str, Any], *, qna_hit: bool, memory_hit: bool,
                 code_hit: bool, programming_hit: bool) -> QueryRoute:
    """Choose a route using input type first, then local retrieval evidence."""
    kind = features["kind"]
    if kind in {QueryKind.CODE.value, QueryKind.MIXED.value} and code_hit:
        return QueryRoute.CODE
    if features["looks_like_memory_request"] and memory_hit:
        return QueryRoute.MEMORY
    if features["looks_like_programming"] and (code_hit or programming_hit):
        return QueryRoute.CODE if code_hit else QueryRoute.PROGRAMMING
    if qna_hit:
        return QueryRoute.QNA
    if memory_hit:
        return QueryRoute.MEMORY
    if programming_hit:
        return QueryRoute.PROGRAMMING
    return QueryRoute.UNKNOWN
