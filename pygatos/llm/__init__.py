"""LLM backend implementations."""

from pygatos.llm.base import BaseLLM
from pygatos.llm.ollama import OllamaBackend

# CerebrasBackend is imported lazily to avoid requiring cerebras-cloud-sdk
# unless it's actually used
def __getattr__(name):
    if name == "CerebrasBackend":
        from pygatos.llm.cerebras import CerebrasBackend
        return CerebrasBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["BaseLLM", "OllamaBackend", "CerebrasBackend"]
