"""Ollama HTTP adapter; no Ollama Python package is required."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from .base import AIAdapter
from .http import post_json

class OllamaAdapter(AIAdapter):
    provider = "ollama"

    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _prompt(question: str, context: Optional[List[Dict[str, Any]]]) -> str:
        lines = ["You are an AI assistant with a local memory store.",
                 "Use the supplied memory only when it is relevant. If memory is insufficient, say so rather than inventing facts.", ""]
        if context:
            lines.append("Relevant local memories:")
            for item in context:
                lines.append(f"- Q: {item.get('question', '')}\n  A: {item.get('answer', '')}")
            lines.append("")
        lines.append(f"User question: {question}")
        return "\n".join(lines)

    def generate(self, question: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        result = post_json(f"{self.base_url}/api/chat", {
            "model": self.model,
            "messages": [{"role": "user", "content": self._prompt(question, context)}],
            "stream": False,
        }, timeout=self.timeout)
        try:
            return str(result["message"]["content"]).strip()
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Unexpected Ollama response format") from exc
