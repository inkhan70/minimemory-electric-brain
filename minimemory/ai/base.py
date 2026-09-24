"""Small provider-neutral AI adapter interface for minimemory."""
from __future__ import annotations
from typing import Any, Dict, List, Optional

class AIAdapter:
    """Minimal interface: generate an answer from a question and memory context."""
    provider = "unknown"

    def generate(self, question: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        raise NotImplementedError

    def test(self) -> Dict[str, Any]:
        answer = self.generate("Reply with exactly: minimemory connection OK", [])
        return {"ok": bool(answer.strip()), "provider": self.provider, "answer": answer}
