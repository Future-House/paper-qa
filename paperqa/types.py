from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, validator

from .prompts import (
    citation_prompt,
    qa_prompt,
    search_prompt,
    select_paper_prompt,
    summary_prompt,
)

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


class PromptCollection(BaseModel):
    summary: PromptTemplate = summary_prompt
    qa: PromptTemplate = qa_prompt
    select: PromptTemplate = select_paper_prompt
    search: PromptTemplate = search_prompt
    cite: PromptTemplate = citation_prompt
    pre: Optional[PromptTemplate] = None
    post: Optional[PromptTemplate] = None

    @validator("summary")
    def check_summary(cls, v: PromptTemplate) -> PromptTemplate:
        if set(v.input_variables) != set(summary_prompt.input_variables):
            raise ValueError(
                f"Summary prompt must have input variables: {summary_prompt.input_variables}"
            )
        return v

    @validator("qa")
    def check_qa(cls, v: PromptTemplate) -> PromptTemplate:
        if set(v.input_variables) != set(qa_prompt.input_variables):
            raise ValueError(
                f"QA prompt must have input variables: {qa_prompt.input_variables}"
            )
        return v

    @validator("select")
    def check_select(cls, v: PromptTemplate) -> PromptTemplate:
        if set(v.input_variables) != set(select_paper_prompt.input_variables):
            raise ValueError(
                f"Select prompt must have input variables: {select_paper_prompt.input_variables}"
            )
        return v

    @validator("search")
    def check_search(cls, v: PromptTemplate) -> PromptTemplate:
        if set(v.input_variables) != set(search_prompt.input_variables):
            raise ValueError(
                f"Search prompt must have input variables: {search_prompt.input_variables}"
            )
        return v

    @validator("pre")
    def check_pre(cls, v: Optional[PromptTemplate]) -> Optional[PromptTemplate]:
        if v is not None:
            if set(v.input_variables) != set(["answer"]):
                raise ValueError("Pre prompt must have input variables: question")
        return v

    @validator("post")
    def check_post(cls, v: Optional[PromptTemplate]) -> Optional[PromptTemplate]:
        if v is not None:
            if set(v.input_variables) != set(["answer", "answer"]):
                raise ValueError(
                    "Post prompt must have input variables: question, answer"
                )
        return v


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
    summary_length: str = "about 100 words"
    answer_length: str = "about 100 words"

    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer
