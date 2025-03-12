import asyncio
import itertools
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Iterable,
    Sequence,
    Sized,
)
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from lmi import (
    Embeddable,
    EmbeddingModel,
    EmbeddingModes,
    HybridEmbeddingModel,
    LiteLLMEmbeddingModel,
    SentenceTransformerEmbeddingModel,
    SparseEmbeddingModel,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import override

from paperqa.types import Doc, Text

if TYPE_CHECKING:
    from qdrant_client.http.models import Record

    from paperqa.docs import Docs

try:
    from qdrant_client import AsyncQdrantClient, models

    qdrant_installed = True
except ImportError:
    qdrant_installed = False

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
    async def add_texts_and_embeddings(self, texts: Iterable[Embeddable]) -> None:
        """Add texts and their embeddings to the store."""
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

    async def add_texts_and_embeddings(self, texts: Iterable[Embeddable]) -> None:
        await super().add_texts_and_embeddings(texts)
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


class QdrantVectorStore(VectorStore):
    client: Any = Field(
        default=None,
        description=(
            "Instance of `qdrant_client.AsyncQdrantClient`. Defaults to an in-memory"
            " instance."
        ),
    )
    collection_name: str = Field(default_factory=lambda: f"paper-qa-{uuid.uuid4().hex}")
    vector_name: str | None = Field(default=None)
    _point_ids: set[str] | None = None

    def __del__(self):
        """Cleanup async client connection."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                _ = loop.create_task(self.aclose())  # noqa: RUF006
            else:
                loop.run_until_complete(self.aclose())
        except Exception as e:
            logger.warning(f"Error closing client connection: {e}")

    async def aclose(self):
        """Explicitly close async client."""
        await self.client.close()

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.texts_hashes == other.texts_hashes
            and self.mmr_lambda == other.mmr_lambda
            and self.collection_name == other.collection_name
            and self.vector_name == other.vector_name
            and self.client.init_options == other.client.init_options
            and self._point_ids == other._point_ids
        )

    @model_validator(mode="after")
    def validate_client(self):
        if not qdrant_installed:
            msg = (
                "`QdrantVectorStore` requires the `qdrant-client` package. "
                "Install it with `pip install paper-qa[qdrant]`"
            )
            raise ImportError(msg)

        if self.client and not isinstance(self.client, AsyncQdrantClient):
            raise TypeError(
                "'client' should be an instance of AsyncQdrantClient. Got"
                f" `{type(self.client)}`"
            )

        if not self.client:
            # Defaults to the Python based in-memory implementation.
            self.client = AsyncQdrantClient(location=":memory:")

        return self

    async def _collection_exists(self) -> bool:
        return await self.client.collection_exists(self.collection_name)

    @override
    def clear(self) -> None:
        """Synchronous clear method that matches parent class."""
        super().clear()  # Clear the base class attributes first

        # Create a new event loop in a new thread to avoid nested loop issues
        def run_async():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(self.aclear())
            finally:
                new_loop.close()

        thread = threading.Thread(target=run_async)
        thread.start()
        thread.join()

    async def aclear(self) -> None:
        """Asynchronous clear implementation."""
        if not await self._collection_exists():
            return

        await self.client.delete_collection(collection_name=self.collection_name)
        self._point_ids = None

    async def add_texts_and_embeddings(self, texts: Iterable[Embeddable]) -> None:
        await super().add_texts_and_embeddings(texts)

        texts_list = list(texts)

        if texts_list and not await self._collection_exists():
            params = models.VectorParams(
                size=len(cast("Sized", texts_list[0].embedding)),
                distance=models.Distance.COSINE,
            )

            await self.client.create_collection(
                self.collection_name,
                vectors_config=(
                    {self.vector_name: params} if self.vector_name else params
                ),
            )

        ids, payloads, vectors = [], [], []
        for text in texts_list:
            ids.append(uuid.uuid5(uuid.NAMESPACE_URL, str(text.embedding)).hex)
            payloads.append(text.model_dump(exclude={"embedding"}))
            vectors.append(
                {self.vector_name: text.embedding}
                if self.vector_name
                else text.embedding
            )

        await self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=some_id,
                    payload=some_payload,
                    vector=some_vector,
                )
                for some_id, some_payload, some_vector in zip(
                    ids, payloads, vectors, strict=True
                )
            ],
        )
        self._point_ids = set(ids)

    async def similarity_search(
        self, query: str, k: int, embedding_model: EmbeddingModel
    ) -> tuple[Sequence[Embeddable], list[float]]:
        if not await self._collection_exists():
            return ([], [])

        embedding_model.set_mode(EmbeddingModes.QUERY)
        np_query = np.array((await embedding_model.embed_documents([query]))[0])
        embedding_model.set_mode(EmbeddingModes.DOCUMENT)

        points = (
            await self.client.query_points(
                collection_name=self.collection_name,
                query=np_query,
                using=self.vector_name,
                limit=k,
                with_vectors=True,
                with_payload=True,
            )
        ).points

        return (
            [
                Text(
                    **p.payload,
                    embedding=(
                        p.vector[self.vector_name] if self.vector_name else p.vector
                    ),
                )
                for p in points
            ],
            [p.score for p in points],
        )

    @classmethod
    async def load_docs(
        cls,
        client: "AsyncQdrantClient",
        collection_name: str,
        vector_name: str | None = None,
        batch_size: int = 100,
        max_concurrent_requests: int = 5,
    ) -> "Docs":
        from paperqa.docs import Docs  # Avoid circular imports

        vectorstore = cls(
            client=client, collection_name=collection_name, vector_name=vector_name
        )
        docs = Docs(texts_index=vectorstore)

        collection_info = await client.get_collection(collection_name)
        total_points = collection_info.points_count or 0

        semaphore = asyncio.Semaphore(max_concurrent_requests)
        all_points: list[Record] = []

        async def fetch_batch_with_semaphore(offset: int) -> None:
            async with semaphore:
                points = await client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                all_points.extend(points[0])

        tasks = [
            fetch_batch_with_semaphore(offset)
            for offset in range(0, total_points, batch_size)
        ]
        await asyncio.gather(*tasks)

        for point in all_points:
            try:
                if point.payload is None:
                    continue

                payload = point.payload
                doc_data = payload.get("doc", {})
                if not isinstance(doc_data, dict):
                    continue

                if doc_data.get("dockey") not in docs.docs:
                    docs.docs[doc_data["dockey"]] = Doc(
                        docname=doc_data.get("docname", ""),
                        citation=doc_data.get("citation", ""),
                        dockey=doc_data["dockey"],
                    )
                    docs.docnames.add(doc_data.get("docname", ""))

                if point.vector is None:
                    continue

                vector_value = (
                    point.vector.get(vector_name)
                    if vector_name and isinstance(point.vector, dict)
                    else point.vector
                )

                text = Text(
                    text=payload.get("text", ""),
                    name=payload.get("name", ""),
                    doc=docs.docs[doc_data["dockey"]],
                    embedding=vector_value,
                )
                docs.texts.append(text)

            except KeyError as e:
                logger.warning(f"Skipping invalid point due to missing field: {e!s}")
                continue

        return docs


def embedding_model_factory(embedding: str, **kwargs) -> EmbeddingModel:
    """
    Factory function to create an appropriate EmbeddingModel based on the embedding string.

    Supports:
    - SentenceTransformer models prefixed with "st-" (e.g., "st-multi-qa-MiniLM-L6-cos-v1")
    - LiteLLM models (default if no prefix is provided)
    - Hybrid embeddings prefixed with "hybrid-", contains a sparse and a dense model

    Args:
        embedding: The embedding model identifier. Supports prefixes like "st-" for SentenceTransformer
                   and "hybrid-" for combining multiple embedding models.
        **kwargs: Additional keyword arguments for the embedding model.
    """
    embedding = embedding.strip()  # Remove any leading/trailing whitespace

    if embedding.startswith("hybrid-"):
        # Extract the component embedding identifiers after "hybrid-"
        dense_name = embedding[len("hybrid-") :]

        if not dense_name:
            raise ValueError(
                "Hybrid embedding must contain at least one component embedding."
            )

        # Recursively create each component embedding model
        dense_model = embedding_model_factory(dense_name, **kwargs)
        sparse_model = SparseEmbeddingModel(**kwargs)

        return HybridEmbeddingModel(models=[dense_model, sparse_model])

    if embedding.startswith("st-"):
        # Extract the SentenceTransformer model name after "st-"
        model_name = embedding[len("st-") :].strip()
        if not model_name:
            raise ValueError(
                "SentenceTransformer model name must be specified after 'st-'."
            )

        return SentenceTransformerEmbeddingModel(
            name=model_name,
            config=kwargs,
        )

    if embedding.startswith("litellm-"):
        # Extract the LiteLLM model name after "litellm-"
        model_name = embedding[len("litellm-") :].strip()
        if not model_name:
            raise ValueError("model name must be specified after 'litellm-'.")

        return LiteLLMEmbeddingModel(
            name=model_name,
            config=kwargs,
        )

    if embedding == "sparse":
        return SparseEmbeddingModel(**kwargs)

    # Default to LiteLLMEmbeddingModel if no special prefix is found
    return LiteLLMEmbeddingModel(name=embedding, config=kwargs)
