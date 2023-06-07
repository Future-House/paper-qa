from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from pydantic import BaseModel

StrPath = Union[str, Path]
DocKey = Any
CBManager = Union[AsyncCallbackManagerForChainRun, CallbackManagerForChainRun]
CallbackFactory = Callable[[str], Union[None, List[BaseCallbackHandler]]]


class Doc(BaseModel):
    docname: str
    citation: str
    dockey: DocKey


class Text(BaseModel):
    text: str
    name: str
    doc: Doc
    embeddings: Optional[List[float]] = None


class Context(BaseModel):
    """A class to hold the context of a question."""

    context: str
    text: Text
    score: int = 5


def __str__(self) -> str:
    """Return the context as a string."""
    return self.context


class Answer(BaseModel):
    """A class to hold the answer to a question."""

    question: str
    answer: str = ""
    context: str = ""
    contexts: List[Context] = []
    references: str = ""
    formatted_answer: str = ""
    dockey_filter: Optional[Set[DocKey]] = None

    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer
