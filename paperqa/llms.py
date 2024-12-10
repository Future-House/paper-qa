from __future__ import annotations

import itertools
import logging
from abc import ABC, abstractmethod
from collections.abc import (
    Awaitable,
    Callable,
    Iterable,
    Sequence,
)

import numpy as np
from llmclient import (
    Embeddable,
    EmbeddingModel,
    EmbeddingModes,
    LLMResult,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

PromptRunner = Callable[
    [dict, list[Callable[[str], None]] | None, str | None],
    Awaitable[LLMResult],
]

logger = logging.getLogger(__name__)


def cosine_similarity(a, b):
    norm_product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return a @ b.T / norm_product


class VectorStore(BaseModel, ABC):
    """Interface for vector store - very similar to LangChain's VectorStore to be compatible."""

    model_config = ConfigDict(extra="forbid")

    # can be tuned for different tasks
    mmr_lambda: float = Field(
        default=1.0,
        ge=0.0,
        description="MMR lambda value, a value above 1 disables MMR search.",
    )
    texts_hashes: set[int] = Field(default_factory=set)

    def __contains__(self, item) -> bool:
        return hash(item) in self.texts_hashes

    def __len__(self) -> int:
        return len(self.texts_hashes)

    @abstractmethod
    def add_texts_and_embeddings(self, texts: Iterable[Embeddable]) -> None:
        self.texts_hashes.update(hash(t) for t in texts)

    @abstractmethod
    async def similarity_search(
        self, query: str, k: int, embedding_model: EmbeddingModel
    ) -> tuple[Sequence[Embeddable], list[float]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        self.texts_hashes = set()

    async def partitioned_similarity_search(
        self,
        query: str,
        k: int,
        embedding_model: EmbeddingModel,
        partitioning_fn: Callable[[Embeddable], int],
    ) -> tuple[Sequence[Embeddable], list[float]]:
        """Partition the documents into different groups and perform similarity search.

        Args:
            query: query string
            k: Number of results to return
            embedding_model: model used to embed the query
            partitioning_fn: function to partition the documents into different groups.

        Returns:
            Tuple of lists of Embeddables and scores of length k.
        """
        raise NotImplementedError(
            "partitioned_similarity_search is not implemented for this VectorStore."
        )

    async def max_marginal_relevance_search(
        self,
        query: str,
        k: int,
        fetch_k: int,
        embedding_model: EmbeddingModel,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> tuple[Sequence[Embeddable], list[float]]:
        """Vectorized implementation of Maximal Marginal Relevance (MMR) search.

        Args:
            query: Query vector.
            k: Number of results to return.
            fetch_k: Number of results to fetch from the vector store.
            embedding_model: model used to embed the query
            partitioning_fn: optional function to partition the documents into
                different groups, performing MMR within each group.

        Returns:
            List of tuples (doc, score) of length k.
        """
        if fetch_k < k:
            raise ValueError("fetch_k must be greater or equal to k")

        if partitioning_fn is None:
            texts, scores = await self.similarity_search(
                query, fetch_k, embedding_model
            )
        else:
            texts, scores = await self.partitioned_similarity_search(
                query, fetch_k, embedding_model, partitioning_fn
            )

        if len(texts) <= k or self.mmr_lambda >= 1.0:
            return texts, scores

        embeddings = np.array([t.embedding for t in texts])
        np_scores = np.array(scores)
        similarity_matrix = cosine_similarity(embeddings, embeddings)

        selected_indices = [0]
        remaining_indices = list(range(1, len(texts)))

        while len(selected_indices) < k:
            selected_similarities = similarity_matrix[:, selected_indices]
            max_sim_to_selected = selected_similarities.max(axis=1)

            mmr_scores = (
                self.mmr_lambda * np_scores
                - (1 - self.mmr_lambda) * max_sim_to_selected
            )
            mmr_scores[selected_indices] = -np.inf  # Exclude already selected documents

            max_mmr_index = mmr_scores.argmax()
            selected_indices.append(max_mmr_index)
            remaining_indices.remove(max_mmr_index)

        return [texts[i] for i in selected_indices], [
            scores[i] for i in selected_indices
        ]


class NumpyVectorStore(VectorStore):
    texts: list[Embeddable] = Field(default_factory=list)
    _embeddings_matrix: np.ndarray | None = None
    _texts_filter: np.ndarray | None = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.texts == other.texts
            and self.texts_hashes == other.texts_hashes
            and self.mmr_lambda == other.mmr_lambda
            and (
                other._embeddings_matrix is None
                if self._embeddings_matrix is None
                else (
                    False
                    if other._embeddings_matrix is None
                    else np.allclose(self._embeddings_matrix, other._embeddings_matrix)
                )
            )
        )

    def clear(self) -> None:
        super().clear()
        self.texts = []
        self._embeddings_matrix = None
        self._texts_filter = None

    def add_texts_and_embeddings(self, texts: Iterable[Embeddable]) -> None:
        super().add_texts_and_embeddings(texts)
        self.texts.extend(texts)
        self._embeddings_matrix = np.array([t.embedding for t in self.texts])

    async def partitioned_similarity_search(
        self,
        query: str,
        k: int,
        embedding_model: EmbeddingModel,
        partitioning_fn: Callable[[Embeddable], int],
    ) -> tuple[Sequence[Embeddable], list[float]]:
        scores: list[list[float]] = []
        texts: list[Sequence[Embeddable]] = []

        text_partitions = np.array([partitioning_fn(t) for t in self.texts])
        # CPU bound so replacing w a gather wouldn't get us anything
        # plus we need to reset self._texts_filter each iteration
        for partition in np.unique(text_partitions):
            self._texts_filter = text_partitions == partition
            _texts, _scores = await self.similarity_search(query, k, embedding_model)
            texts.append(_texts)
            scores.append(_scores)
        # reset the filter after running
        self._texts_filter = None

        return (
            [
                t
                for t in itertools.chain.from_iterable(itertools.zip_longest(*texts))
                if t is not None
            ][:k],
            [
                s
                for s in itertools.chain.from_iterable(itertools.zip_longest(*scores))
                if s is not None
            ][:k],
        )

    async def similarity_search(
        self, query: str, k: int, embedding_model: EmbeddingModel
    ) -> tuple[Sequence[Embeddable], list[float]]:
        k = min(k, len(self.texts))
        if k == 0:
            return [], []

        # this will only affect models that embedding prompts
        embedding_model.set_mode(EmbeddingModes.QUERY)

        np_query = np.array((await embedding_model.embed_documents([query]))[0])

        embedding_model.set_mode(EmbeddingModes.DOCUMENT)

        embedding_matrix = self._embeddings_matrix

        if self._texts_filter is not None:
            original_indices = np.where(self._texts_filter)[0]
            embedding_matrix = embedding_matrix[self._texts_filter]  # type: ignore[index]
        else:
            original_indices = np.arange(len(self.texts))

        similarity_scores = cosine_similarity(
            np_query.reshape(1, -1), embedding_matrix
        )[0]
        similarity_scores = np.nan_to_num(similarity_scores, nan=-np.inf)
        # minus so descending
        # we could use arg-partition here
        # but a lot of algorithms expect a sorted list
        sorted_indices = np.argsort(-similarity_scores)
        return (
            [self.texts[i] for i in original_indices[sorted_indices][:k]],
            [similarity_scores[i] for i in sorted_indices[:k]],
        )
