import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lmi import (
        EmbeddingModel,
        HybridEmbeddingModel,
        LiteLLMEmbeddingModel,
        LiteLLMModel,
        LLMModel,
        LLMResult,
        SentenceTransformerEmbeddingModel,
        SparseEmbeddingModel,
        embedding_model_factory,
    )

    from paperqa.agents import ask
    from paperqa.agents.main import agent_query
    from paperqa.docs import Docs, PQASession
    from paperqa.llms import (
        NumpyVectorStore,
        QdrantVectorStore,
        VectorStore,
    )
    from paperqa.settings import Settings, get_settings
    from paperqa.types import Context, Doc, DocDetails, Text
    from paperqa.version import __version__

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "EmbeddingModel": ("lmi", "EmbeddingModel"),
    "HybridEmbeddingModel": ("lmi", "HybridEmbeddingModel"),
    "LiteLLMEmbeddingModel": ("lmi", "LiteLLMEmbeddingModel"),
    "LiteLLMModel": ("lmi", "LiteLLMModel"),
    "LLMModel": ("lmi", "LLMModel"),
    "LLMResult": ("lmi", "LLMResult"),
    "SentenceTransformerEmbeddingModel": ("lmi", "SentenceTransformerEmbeddingModel"),
    "SparseEmbeddingModel": ("lmi", "SparseEmbeddingModel"),
    "embedding_model_factory": ("lmi", "embedding_model_factory"),
    "ask": ("paperqa.agents", "ask"),
    "agent_query": ("paperqa.agents.main", "agent_query"),
    "Docs": ("paperqa.docs", "Docs"),
    "PQASession": ("paperqa.docs", "PQASession"),
    "NumpyVectorStore": ("paperqa.llms", "NumpyVectorStore"),
    "QdrantVectorStore": ("paperqa.llms", "QdrantVectorStore"),
    "VectorStore": ("paperqa.llms", "VectorStore"),
    "Settings": ("paperqa.settings", "Settings"),
    "get_settings": ("paperqa.settings", "get_settings"),
    "Context": ("paperqa.types", "Context"),
    "Doc": ("paperqa.types", "Doc"),
    "DocDetails": ("paperqa.types", "DocDetails"),
    "Text": ("paperqa.types", "Text"),
    "__version__": ("paperqa.version", "__version__"),
}

__all__ = [
    "Context",
    "Doc",
    "DocDetails",
    "Docs",
    "EmbeddingModel",
    "HybridEmbeddingModel",
    "LLMModel",
    "LLMResult",
    "LiteLLMEmbeddingModel",
    "LiteLLMModel",
    "NumpyVectorStore",
    "PQASession",
    "QdrantVectorStore",
    "SentenceTransformerEmbeddingModel",
    "Settings",
    "SparseEmbeddingModel",
    "Text",
    "VectorStore",
    "__version__",
    "agent_query",
    "ask",
    "embedding_model_factory",
    "get_settings",
]


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        val = getattr(module, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
