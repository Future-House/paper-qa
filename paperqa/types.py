from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field, field_validator

from .prompts import (
    citation_prompt,
    default_system_prompt,
    qa_prompt,
    select_paper_prompt,
    summary_prompt,
)

# Just for clarity
DocKey = Any

CallbackFactory = Callable[[str], Callable[[str], None]]


class Embeddable(BaseModel):
    embedding: list[float] | None = Field(default=None, repr=False)


class Doc(Embeddable):
    docname: str
    citation: str
    dockey: DocKey


class Text(Embeddable):
    text: str
    name: str
    doc: Doc


def cosine_similarity(a, b):
    dot_product = np.dot(a, b.T)
    norm_product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return dot_product / norm_product


class VectorStore(BaseModel, ABC):
    """Interface for vector store - very similar to LangChain's VectorStore to be compatible"""

    @abstractmethod
    def add_texts_and_embeddings(self, texts: list[Embeddable]) -> None:
        pass

    @abstractmethod
    def similarity_search(
        self, query: list[float], k: int
    ) -> list[tuple[Embeddable, float]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    def max_marginal_relevance_search(
        self, query: list[float], k: int, fetch_k: int, lambda_: float = 0.5
    ) -> list[tuple[Embeddable, float]]:
        """Vectorized implementation of Maximal Marginal Relevance (MMR) search.

        Args:
            query: Query vector.
            k: Number of results to return.
            lambda_: Weighting of relevance and diversity.

        Returns:
            List of tuples (doc, score) of length k.
        """
        if fetch_k < k:
            raise ValueError("fetch_k must be greater or equal to k")

        initial_results = self.similarity_search(query, fetch_k)
        if len(initial_results) <= k:
            return initial_results

        embeddings = np.array([t.embedding for t, _ in initial_results])
        scores = np.array([score for _, score in initial_results])
        similarity_matrix = cosine_similarity(embeddings, embeddings)

        selected_indices = [0]
        remaining_indices = list(range(1, len(initial_results)))

        while len(selected_indices) < k:
            selected_similarities = similarity_matrix[:, selected_indices]
            max_sim_to_selected = selected_similarities.max(axis=1)

            mmr_scores = lambda_ * scores - (1 - lambda_) * max_sim_to_selected
            mmr_scores[selected_indices] = -np.inf  # Exclude already selected documents

            max_mmr_index = mmr_scores.argmax()
            selected_indices.append(max_mmr_index)
            remaining_indices.remove(max_mmr_index)

        return [(initial_results[i][0], scores[i]) for i in selected_indices]


class NumpyVectorStore(VectorStore):
    texts: list[Embeddable] = []
    _embeddings_matrix: np.ndarray | None = None

    def clear(self) -> None:
        self.texts = []
        self._embeddings_matrix = None

    def add_texts_and_embeddings(
        self,
        texts: list[Embeddable],
    ) -> None:
        self.texts.extend(texts)
        self._embeddings_matrix = np.array([t.embedding for t in self.texts])

    def similarity_search(
        self, query: list[float], k: int
    ) -> list[tuple[Embeddable, float]]:
        if len(self.texts) == 0:
            return []
        query = np.array(query)
        similarity_scores = cosine_similarity(
            query.reshape(1, -1), self._embeddings_matrix
        )[0]
        similarity_scores = np.nan_to_num(similarity_scores, nan=-np.inf)
        sorted_indices = np.argsort(similarity_scores)[::-1]
        return [(self.texts[i], similarity_scores[i]) for i in sorted_indices[:k]]


class _FormatDict(dict):
    def __missing__(self, key: str) -> str:
        return key


def get_formatted_variables(s: str) -> set[str]:
    format_dict = _FormatDict()
    s.format_map(format_dict)
    return set(format_dict.keys())


class PromptCollection(BaseModel):
    summary: str = summary_prompt
    qa: str = qa_prompt
    select: str = select_paper_prompt
    cite: str = citation_prompt
    pre: str | None = None
    post: str | None = None
    system: str = default_system_prompt
    skip_summary: bool = False

    @field_validator("summary")
    @classmethod
    def check_summary(cls, v: str) -> str:
        if not set(get_formatted_variables(v)).issubset(
            set(get_formatted_variables(summary_prompt))
        ):
            raise ValueError(
                f"Summary prompt can only have variables: {get_formatted_variables(summary_prompt)}"
            )
        return v

    @field_validator("qa")
    @classmethod
    def check_qa(cls, v: str) -> str:
        if not set(get_formatted_variables(v)).issubset(
            set(get_formatted_variables(qa_prompt))
        ):
            raise ValueError(
                f"QA prompt can only have variables: {get_formatted_variables(qa_prompt)}"
            )
        return v

    @field_validator("select")
    @classmethod
    def check_select(cls, v: str) -> str:
        if not set(get_formatted_variables(v)).issubset(
            set(get_formatted_variables(select_paper_prompt))
        ):
            raise ValueError(
                f"Select prompt can only have variables: {get_formatted_variables(select_paper_prompt)}"
            )
        return v

    @field_validator("pre")
    @classmethod
    def check_pre(cls, v: str | None) -> str | None:
        if v is not None:
            if set(get_formatted_variables(v)) != set(["question"]):
                raise ValueError("Pre prompt must have input variables: question")
        return v

    @field_validator("post")
    @classmethod
    def check_post(cls, v: str | None) -> str | None:
        if v is not None:
            # kind of a hack to get list of attributes in answer
            attrs = [a.name for a in Answer.__fields__.values()]
            if not set(get_formatted_variables(v)).issubset(attrs):
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
    contexts: list[Context] = []
    references: str = ""
    formatted_answer: str = ""
    dockey_filter: set[DocKey] | None = None
    summary_length: str = "about 100 words"
    answer_length: str = "about 100 words"
    memory: str | None = None
    # these two below are for convenience
    # and are not set. But you can set them
    # if you want to use them.
    cost: float | None = None
    token_counts: dict[str, list[int]] | None = None

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
