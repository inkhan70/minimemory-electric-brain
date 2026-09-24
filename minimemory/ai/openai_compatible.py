"""Adapter for OpenAI-compatible chat-completions servers."""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
from .base import AIAdapter
from .http import post_json

class OpenAICompatibleAdapter(AIAdapter):
    provider = "openai-compatible"

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: Optional[str] = None, timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout

    @staticmethod
    def _messages(question: str, context: Optional[List[Dict[str, Any]]]) -> list[dict[str, str]]:
        memory = "\n".join(f"- Q: {x.get('question','')}\n  A: {x.get('answer','')}" for x in (context or []))
        system = "You are an AI assistant with local memory. Use relevant memory as context. Do not invent facts when memory is insufficient."
        if memory:
            system += "\nRelevant local memories:\n" + memory
        return [{"role": "system", "content": system}, {"role": "user", "content": question}]

    def generate(self, question: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        if not self.api_key:
            raise RuntimeError("No API key. Set OPENAI_API_KEY or pass api_key explicitly.")
        result = post_json(f"{self.base_url}/chat/completions", {
            "model": self.model,
            "messages": self._messages(question, context),
        }, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout)
        try:
            return str(result["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected chat-completions response format") from exc
