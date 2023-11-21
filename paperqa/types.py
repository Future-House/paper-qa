from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.prompts import PromptTemplate

try:
    from pydantic.v1 import BaseModel, validator
except ImportError:
    from pydantic import BaseModel, validator

from .prompts import (
    citation_prompt,
    default_system_prompt,
    qa_prompt,
    select_paper_prompt,
    summary_prompt,
)
from .utils import extract_doi, iter_citations

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
    cite: PromptTemplate = citation_prompt
    pre: Optional[PromptTemplate] = None
    post: Optional[PromptTemplate] = None
    system: str = default_system_prompt
    skip_summary: bool = False

    @validator("summary")
    def check_summary(cls, v: PromptTemplate) -> PromptTemplate:
        if not set(v.input_variables).issubset(set(summary_prompt.input_variables)):
            raise ValueError(
                f"Summary prompt can only have variables: {summary_prompt.input_variables}"
            )
        return v

    @validator("qa")
    def check_qa(cls, v: PromptTemplate) -> PromptTemplate:
        if not set(v.input_variables).issubset(set(qa_prompt.input_variables)):
            raise ValueError(
                f"QA prompt can only have variables: {qa_prompt.input_variables}"
            )
        return v

    @validator("select")
    def check_select(cls, v: PromptTemplate) -> PromptTemplate:
        if not set(v.input_variables).issubset(
            set(select_paper_prompt.input_variables)
        ):
            raise ValueError(
                f"Select prompt can only have variables: {select_paper_prompt.input_variables}"
            )
        return v

    @validator("pre")
    def check_pre(cls, v: Optional[PromptTemplate]) -> Optional[PromptTemplate]:
        if v is not None:
            if set(v.input_variables) != set(["question"]):
                raise ValueError("Pre prompt must have input variables: question")
        return v

    @validator("post")
    def check_post(cls, v: Optional[PromptTemplate]) -> Optional[PromptTemplate]:
        if v is not None:
            # kind of a hack to get list of attributes in answer
            attrs = [a.name for a in Answer.__fields__.values()]
            if not set(v.input_variables).issubset(attrs):
                raise ValueError(f"Post prompt must have input variables: {attrs}")
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
    memory: Optional[str] = None
    # these two below are for convenience
    # and are not set. But you can set them
    # if you want to use them.
    cost: Optional[float] = None
    token_counts: Optional[Dict[str, List[int]]] = None

    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer

    def get_citation(self, name: str) -> str:
        """Return the formatted citation for the gien docname."""
        try:
            doc = next(filter(lambda x: x.text.name == name, self.contexts)).text.doc
        except StopIteration:
            raise ValueError(f"Could not find docname {name} in contexts")
        return doc.citation

    def markdown(self) -> Tuple[str, str]:
        """Return the answer with footnote style citations."""
        # example: This is an answer.[^1]
        # [^1]: This the citation.
        output = self.answer
        refs: Dict[str, int] = dict()
        index = 1
        for citation in iter_citations(self.answer):
            compound = ""
            strip = True
            for c in citation.split(",;"):
                c = c.strip("() ")
                if c == "Extra background information":
                    continue
                if c in refs:
                    compound += f"[^{refs[c]}]"
                    continue
                # check if it is a citation
                try:
                    self.get_citation(c)
                except ValueError:
                    # not a citation
                    strip = False
                    continue
                refs[c] = index
                compound += f"[^{index}]"
                index += 1
            if strip:
                output = output.replace(citation, compound)
        formatted_refs = "\n".join(
            [
                f"[^{i}]: [{self.get_citation(r)}]({extract_doi(self.get_citation(r))})"
                for r, i in refs.items()
            ]
        )
        # quick fix of space before period
        output = output.replace(" .", ".")
        return output, formatted_refs

    def combine_with(self, other: "Answer") -> "Answer":
        """
        Combine this answer object with another, merging their context/answer.
        """
        combined = Answer(
            question=self.question + " / " + other.question,
            answer=self.answer + " " + other.answer,
            context=self.context + " " + other.context,
            contexts=self.contexts + other.contexts,
            references=self.references + " " + other.references,
            formatted_answer=self.formatted_answer + " " + other.formatted_answer,
            summary_length=self.summary_length,  # Assuming the same summary_length for both
            answer_length=self.answer_length,  # Assuming the same answer_length for both
            memory=self.memory if self.memory else other.memory,
            cost=self.cost if self.cost else other.cost,
            token_counts=self.merge_token_counts(self.token_counts, other.token_counts),
        )
        # Handling dockey_filter if present in either of the Answer objects
        if self.dockey_filter or other.dockey_filter:
            combined.dockey_filter = (
                self.dockey_filter if self.dockey_filter else set()
            ) | (other.dockey_filter if other.dockey_filter else set())
        return combined

    @staticmethod
    def merge_token_counts(
        counts1: Optional[Dict[str, List[int]]], counts2: Optional[Dict[str, List[int]]]
    ) -> Optional[Dict[str, List[int]]]:
        """
        Merge two dictionaries of token counts.
        """
        if counts1 is None and counts2 is None:
            return None
        if counts1 is None:
            return counts2
        if counts2 is None:
            return counts1
        merged_counts = counts1.copy()
        for key, values in counts2.items():
            if key in merged_counts:
                merged_counts[key][0] += values[0]
                merged_counts[key][1] += values[1]
            else:
                merged_counts[key] = values
        return merged_counts
