import warnings

# TODO: remove after refactoring all models to avoid using _* private vars
warnings.filterwarnings(
    "ignore", message="Valid config keys have changed in V2:", module="pydantic"
)


from paperqa.agents import ask  # noqa: E402
from paperqa.agents.main import agent_query  # noqa: E402
from paperqa.agents.models import QueryRequest  # noqa: E402
from paperqa.docs import Docs, PQASession, print_callback  # noqa: E402
from paperqa.llms import (  # noqa: E402
    LiteLLMModel,
    LLMModel,
    LLMResult,
    NumpyVectorStore,
)
from paperqa.settings import Settings, get_settings  # noqa: E402
from paperqa.types import Answer, Context, Doc, DocDetails, Text  # noqa: E402
from paperqa.version import __version__  # noqa: E402

__all__ = [
    "Answer",
    "Context",
    "Doc",
    "DocDetails",
    "Docs",
    "LLMModel",
    "LLMResult",
    "LiteLLMModel",
    "NumpyVectorStore",
    "PQASession",
    "QueryRequest",
    "Settings",
    "Text",
    "__version__",
    "agent_query",
    "ask",
    "get_settings",
    "print_callback",
]
