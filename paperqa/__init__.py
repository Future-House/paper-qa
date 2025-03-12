import warnings

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
from paperqa.docs import Docs, PQASession, print_callback
from paperqa.llms import (
    NumpyVectorStore,
    QdrantVectorStore,
    VectorStore,
)
from paperqa.settings import Settings, get_settings
from paperqa.types import Answer, Context, Doc, DocDetails, Text
from paperqa.version import __version__

# TODO: remove after refactoring all models to avoid using _* private vars
warnings.filterwarnings(
    "ignore", message="Valid config keys have changed in V2:", module="pydantic"
)


__all__ = [
    "Answer",
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
    "print_callback",
]
