"""Lightweight, offline-first text refinement and verification pipeline."""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, Iterable, List, Optional

_ALLOWED = set(string.ascii_letters + "0123456789 .,?!'\"\n\t-_:;()")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SPACE_RE = re.compile(r"[ \t]+")


@dataclass
class CheckResult:
    name: str
    passed: bool
    confidence: float = 1.0
    changed: bool = False
    text: str = ""
    issues: List[str] = field(default_factory=list)


class TextPipeline:
    """Deterministic text cleanup plus optional application-provided checks.

    This layer intentionally does not pretend to prove factual correctness. It
    handles safe normalization and lets callers provide dictionary/grammar/
    grounding callbacks for stronger checks.
    """

    def __init__(self, *, alphabet_only: bool = False,
                 spelling: Optional[Callable[[str], str]] = None,
                 grammar: Optional[Callable[[str], str]] = None,
                 grounding: Optional[Callable[[str, List[Dict[str, Any]]], Any]] = None):
        self.alphabet_only = bool(alphabet_only)
        self.spelling = spelling
        self.grammar = grammar
        self.grounding = grounding

    @staticmethod
    def normalize(text: str) -> str:
        text = text.strip().replace("\r\n", "\n").replace("\r", "\n")
        text = _SPACE_RE.sub(" ", text)
        text = re.sub(r"[ \t]+([,.!?;:])", r"\1", text)
        return text

    @staticmethod
    def alphabet_filter(text: str) -> str:
        return "".join(ch for ch in text if ch in _ALLOWED)

    @staticmethod
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.casefold(), b.casefold()).ratio()

    def process(self, text: str, *, evidence: Optional[List[Dict[str, Any]]] = None,
                max_passes: int = 2) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if max_passes < 1:
            raise ValueError("max_passes must be >= 1")
        original = text
        current = self.normalize(text)
        checks: List[CheckResult] = []

        if self.alphabet_only:
            filtered = self.alphabet_filter(current)
            checks.append(CheckResult("alphabet", filtered == current, 1.0, filtered != current, filtered,
                                      [] if filtered == current else ["removed unsupported characters"]))
            current = filtered
        else:
            checks.append(CheckResult("alphabet", True, 1.0, False, current))

        for _ in range(max_passes):
            before = current
            if self.spelling:
                current = self.spelling(current).strip()
            checks.append(CheckResult("spelling", True, 0.0 if not self.spelling else 1.0,
                                      current != before, current))
            before = current
            if self.grammar:
                current = self.grammar(current).strip()
            checks.append(CheckResult("grammar", True, 0.0 if not self.grammar else 1.0,
                                      current != before, current))
            current = self.normalize(current)
            if current == before and not self.spelling and not self.grammar:
                break
            if current == before and (not self.spelling or current == before):
                break

        grounded = True
        grounding_confidence = 1.0
        grounding_issues: List[str] = []
        if self.grounding:
            result = self.grounding(current, evidence or [])
            if isinstance(result, dict):
                grounded = bool(result.get("passed", False))
                grounding_confidence = float(result.get("confidence", 0.0))
                grounding_issues = [str(x) for x in result.get("issues", [])]
            else:
                grounded = bool(result)
                grounding_confidence = 1.0 if grounded else 0.0
            checks.append(CheckResult("grounding", grounded, grounding_confidence,
                                      False, current, grounding_issues))

        changed = current != original
        return {
            "original": original,
            "corrected": current,
            "changed": changed,
            "checks": [c.__dict__ for c in checks],
            "grounded": grounded,
            "confidence": min((c.confidence for c in checks if c.confidence > 0), default=1.0),
        }


class RefinementLoop:
    """Bounded answer refinement driven by explicit verification feedback."""

    def __init__(self, *, max_iterations: int = 3):
        if max_iterations < 1 or max_iterations > 10:
            raise ValueError("max_iterations must be between 1 and 10")
        self.max_iterations = int(max_iterations)

    def run(self, answer: str, verifier: Callable[[str], Dict[str, Any]],
            reviser: Callable[[str, Dict[str, Any]], str]) -> Dict[str, Any]:
        current = answer
        history: List[Dict[str, Any]] = []
        for iteration in range(1, self.max_iterations + 1):
            result = verifier(current)
            if not isinstance(result, dict):
                raise TypeError("verifier must return a dictionary")
            passed = bool(result.get("passed", False))
            history.append({"iteration": iteration, "answer": current, "verification": result})
            if passed:
                return {"answer": current, "passed": True, "iterations": iteration, "history": history}
            revised = reviser(current, result)
            if not isinstance(revised, str) or not revised.strip():
                break
            if revised.strip() == current.strip():
                break
            current = revised.strip()
        return {"answer": current, "passed": False, "iterations": len(history), "history": history}
