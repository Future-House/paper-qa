from typing import Union, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

StrPath = Union[str, Path]


@dataclass
class Answer:
    """A class to hold the answer to a question."""

    question: str
    answer: str = ""
    context: str = ""
    contexts: List[Any] = None
    references: str = ""
    formatted_answer: str = ""
    passages: Dict[str, str] = None
    tokens: int = 0
    cost: float = 0

    def __post_init__(self):
        """Initialize the answer."""
        if self.contexts is None:
            self.contexts = []
        if self.passages is None:
            self.passages = {}

    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer


@dataclass
class Context:
    """A class to hold the context of a question."""

    key: str
    citation: str
    context: str
    text: str

    def __str__(self) -> str:
        """Return the context as a string."""
        return self.context
