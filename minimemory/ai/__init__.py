"""Optional AI adapters for connecting minimemory to model servers."""
from .base import AIAdapter
from .ollama import OllamaAdapter
from .openai_compatible import OpenAICompatibleAdapter

__all__ = ["AIAdapter", "OllamaAdapter", "OpenAICompatibleAdapter"]
