import asyncio
import itertools
import json
import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Iterable,
    Sequence,
    Sized,
)
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
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

from paperqa.types import AUTOPOPULATE_VALUE, Doc, Text

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


class NumpyVectorStore(VectorStore):  # noqa: PLW1641  # TODO: add __hash__
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


class QdrantVectorStore(VectorStore):  # noqa: PLW1641  # TODO: add __hash__
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
                    with_payload=True,  # noqa: FURB120
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
                        content_hash=doc_data.get("content_hash", AUTOPOPULATE_VALUE),
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


class MilvusVectorStore(VectorStore):  # noqa: PLW1641  # TODO: add __hash__
    """Milvus-backed vector store using the synchronous ``MilvusClient`` API."""

    client: Any = Field(
        default=None,
        exclude=True,
        description=(
            "Instance of `pymilvus.MilvusClient`. Defaults to a client configured"
            " from `uri`, `token`, and `db_name`."
        ),
    )
    uri: str = Field(
        default_factory=lambda: os.getenv("MILVUS_URI", "./milvus.db"),
        description="Milvus Lite database path or remote Milvus URI.",
    )
    token: str | None = Field(
        default_factory=lambda: os.getenv("MILVUS_TOKEN"),
        description="Token for a remote Milvus deployment.",
        repr=False,
    )
    db_name: str = Field(default="", description="Milvus database name.")
    collection_name: str = Field(default_factory=lambda: f"paper_qa_{uuid.uuid4().hex}")
    consistency_level: str = Field(
        default="Session",
        description="Milvus consistency level used when creating the collection.",
    )
    _entity_ids: set[str] | None = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.texts_hashes == other.texts_hashes
            and self.mmr_lambda == other.mmr_lambda
            and self.uri == other.uri
            and self.token == other.token
            and self.db_name == other.db_name
            and self.collection_name == other.collection_name
            and self.consistency_level == other.consistency_level
            and self._entity_ids == other._entity_ids
        )

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "MilvusVectorStore":
        copied = type(self)(client=self.client, **self.model_dump())
        if memo is not None:
            memo[id(self)] = copied
        copied._entity_ids = None if self._entity_ids is None else set(self._entity_ids)
        return copied

    @staticmethod
    def _pymilvus():
        try:
            return import_module("pymilvus")
        except ImportError as exc:
            msg = (
                "`MilvusVectorStore` requires the `pymilvus` package. "
                "Install it with `pip install 'paper-qa[milvus]'`"
            )
            raise ImportError(msg) from exc

    @model_validator(mode="after")
    def validate_client(self):
        pymilvus = self._pymilvus()
        if self.client and not isinstance(self.client, pymilvus.MilvusClient):
            raise TypeError(
                "'client' should be an instance of MilvusClient. Got"
                f" `{type(self.client)}`"
            )

        if not self.client:
            self.client = pymilvus.MilvusClient(
                uri=self.uri,
                token=self.token or "",
                db_name=self.db_name,
            )

        return self

    async def _collection_exists(self) -> bool:
        return await asyncio.to_thread(
            self.client.has_collection, collection_name=self.collection_name
        )

    def _ensure_collection(self, dimension: int) -> None:
        pymilvus = self._pymilvus()
        if self.client.has_collection(collection_name=self.collection_name):
            self._validate_collection(dimension, pymilvus.DataType)
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="id",
            datatype=pymilvus.DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        schema.add_field(
            field_name="vector",
            datatype=pymilvus.DataType.FLOAT_VECTOR,
            dim=dimension,
        )
        schema.add_field(field_name="payload", datatype=pymilvus.DataType.JSON)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level=self.consistency_level,
        )

    def _validate_collection(self, dimension: int, data_type: Any) -> None:
        description = self.client.describe_collection(
            collection_name=self.collection_name
        )
        fields = {field["name"]: field for field in description.get("fields", [])}
        id_field = fields.get("id")
        vector_field = fields.get("vector")
        payload_field = fields.get("payload")
        schema_flags_valid = all(
            value is False
            for value in (
                description.get("auto_id"),
                description.get("enable_dynamic_field"),
            )
        )
        vector_dimension = -1
        if vector_field is not None:
            try:
                vector_dimension = int(vector_field.get("params", {}).get("dim", -1))
            except (TypeError, ValueError):
                vector_dimension = -1

        valid_schema = (
            schema_flags_valid
            and id_field is not None
            and id_field.get("type") == data_type.VARCHAR
            and id_field.get("is_primary") is True
            and vector_field is not None
            and vector_field.get("type") == data_type.FLOAT_VECTOR
            and vector_dimension == dimension
            and payload_field is not None
            and payload_field.get("type") == data_type.JSON
        )
        if not valid_schema:
            raise ValueError(
                f"Milvus collection {self.collection_name!r} has an incompatible schema."
            )

        if "vector" not in self.client.list_indexes(
            collection_name=self.collection_name
        ):
            raise ValueError(
                f"Milvus collection {self.collection_name!r} requires a COSINE"
                " AUTOINDEX on the vector field."
            )

        index = self.client.describe_index(
            collection_name=self.collection_name, index_name="vector"
        )
        if (
            index.get("index_type") != "AUTOINDEX"
            or index.get("metric_type") != "COSINE"
        ):
            raise ValueError(
                f"Milvus collection {self.collection_name!r} requires a COSINE"
                " AUTOINDEX on the vector field."
            )

    @staticmethod
    def _prepare_embedding(embedding: Any) -> list[float]:
        if embedding is None:
            raise ValueError("Cannot add a text without an embedding to Milvus.")
        vector = np.asarray(embedding, dtype=float)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(
                "Milvus embeddings must be non-empty one-dimensional vectors."
            )
        return vector.tolist()

    @staticmethod
    def _entity_id(payload: dict[str, Any], vector: list[float]) -> str:
        serialized = json.dumps(
            {"payload": payload, "vector": vector},
            sort_keys=True,
            separators=(",", ":"),
        )
        return uuid.uuid5(uuid.NAMESPACE_URL, serialized).hex

    @staticmethod
    def _normalize_score(uri: str, score: float) -> float:
        if uri.startswith(("http://", "https://")):
            return score
        try:
            milvus_lite_version = version("milvus-lite").split("+", maxsplit=1)[0]
        except PackageNotFoundError:
            return score
        # Milvus Lite 3.0 reports COSINE distance instead of similarity.
        # https://github.com/milvus-io/milvus-lite/issues/343
        if milvus_lite_version in {"3.0", "3.0.0"}:
            return 1.0 - score
        return score

    @override
    def clear(self) -> None:
        super().clear()
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        self._entity_ids = None

    async def add_texts_and_embeddings(self, texts: Iterable[Embeddable]) -> None:
        texts_list = list(texts)
        if not texts_list:
            return

        vectors = [self._prepare_embedding(text.embedding) for text in texts_list]
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors[1:]):
            raise ValueError("All Milvus embeddings must have the same dimension.")

        await asyncio.to_thread(self._ensure_collection, dimension)
        entities: list[dict[str, Any]] = []
        for text, vector in zip(texts_list, vectors, strict=True):
            payload = text.model_dump(mode="json", exclude={"embedding"})
            entity_id = self._entity_id(payload, vector)
            entities.append({"id": entity_id, "vector": vector, "payload": payload})

        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.collection_name,
            data=entities,
        )
        await super().add_texts_and_embeddings(texts_list)
        if self._entity_ids is None:
            self._entity_ids = set()
        self._entity_ids.update(str(entity["id"]) for entity in entities)

    async def similarity_search(
        self, query: str, k: int, embedding_model: EmbeddingModel
    ) -> tuple[Sequence[Embeddable], list[float]]:
        if k <= 0 or not await self._collection_exists():
            return ([], [])

        embedding_model.set_mode(EmbeddingModes.QUERY)
        query_vector = self._prepare_embedding(
            (await embedding_model.embed_documents([query]))[0]
        )
        embedding_model.set_mode(EmbeddingModes.DOCUMENT)

        results = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection_name,
            data=[query_vector],
            limit=k,
            output_fields=["payload", "vector"],
            search_params={"metric_type": "COSINE"},
        )
        hits = results[0] if results else []
        return (
            [
                Text(
                    **hit["entity"]["payload"],
                    embedding=hit["entity"]["vector"],
                )
                for hit in hits
            ],
            [self._normalize_score(self.uri, float(hit["distance"])) for hit in hits],
        )


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
