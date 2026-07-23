from collections.abc import AsyncIterator
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import pytest_asyncio
from lmi import EmbeddingModel

from paperqa import (
    Doc,
    Docs,
    MilvusVectorStore,
    NumpyVectorStore,
    QdrantVectorStore,
    Text,
    VectorStore,
)


class StaticEmbeddingModel(EmbeddingModel):
    name: str = "static"
    vector: tuple[float, float] = (1.0, 0.0)

    async def embed_documents(self, texts):
        return [list(self.vector) for _ in texts]


def make_texts() -> list[Text]:
    doc = Doc(docname="vectors", citation="Vector test", dockey="vectors")
    return [
        Text(
            text="alpha",
            name="alpha",
            doc=doc,
            embedding=[1.0, 0.0],
            section="first",
        ),
        Text(
            text="beta",
            name="beta",
            doc=doc,
            embedding=[0.8, 0.2],
            section="second",
        ),
        Text(
            text="gamma",
            name="gamma",
            doc=doc,
            embedding=[0.0, 1.0],
            section="third",
        ),
    ]


@pytest_asyncio.fixture(name="vector_store", params=["numpy", "qdrant", "milvus"])
async def vector_store_fixture(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[VectorStore]:
    if request.param == "numpy":
        store: VectorStore = NumpyVectorStore(mmr_lambda=0.5)
    elif request.param == "qdrant":
        store = QdrantVectorStore(mmr_lambda=0.5)
    else:
        store = MilvusVectorStore(uri=str(tmp_path / "milvus.db"), mmr_lambda=0.5)

    yield store

    if isinstance(store, QdrantVectorStore):
        await store.aclear()
        await store.aclose()
    elif isinstance(store, MilvusVectorStore):
        store.clear()
        store.client.close()


@pytest.mark.asyncio
async def test_vector_store_conformance(vector_store: VectorStore) -> None:
    texts = make_texts()
    embedding_model = StaticEmbeddingModel()

    await vector_store.add_texts_and_embeddings(texts)

    assert len(vector_store) == len(texts)
    assert all(text in vector_store for text in texts)

    matches, scores = await vector_store.similarity_search(
        "alpha", k=10, embedding_model=embedding_model
    )
    assert [cast("Text", match).text for match in matches] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert scores == sorted(scores, reverse=True)
    assert matches[0].model_dump(exclude={"embedding"}) == texts[0].model_dump(
        exclude={"embedding"}
    )
    assert matches[0].embedding == texts[0].embedding

    mmr_matches, mmr_scores = await vector_store.max_marginal_relevance_search(
        "alpha", k=2, fetch_k=3, embedding_model=embedding_model
    )
    assert len(mmr_matches) == len(mmr_scores) == 2
    assert cast("Text", mmr_matches[0]).text == "alpha"

    vector_store.clear()
    assert len(vector_store) == 0
    assert not vector_store.texts_hashes
    assert await vector_store.similarity_search(
        "alpha", k=3, embedding_model=embedding_model
    ) == ([], [])


@pytest.mark.asyncio
async def test_milvus_schema_and_copy_semantics(tmp_path: Path) -> None:
    store = MilvusVectorStore(uri=str(tmp_path / "milvus.db"))
    empty_copy = type(store)(**store.model_dump())
    assert store == empty_copy
    empty_copy.client.close()

    await store.add_texts_and_embeddings(make_texts())

    description = store.client.describe_collection(store.collection_name)
    fields = {field["name"]: field for field in description["fields"]}
    assert fields["id"]["is_primary"] is True
    assert fields["vector"]["params"]["dim"] == 2
    index = store.client.describe_index(store.collection_name, "vector")
    assert index["index_type"] == "AUTOINDEX"
    assert index["metric_type"] == "COSINE"
    assert index["field_name"] == "vector"
    assert index["state"] == "Finished"

    shallow_copy = type(store)(**store.model_dump())
    deep_copy = deepcopy(store)
    assert store != shallow_copy
    assert store == deep_copy
    shallow_copy.client.close()

    store.clear()
    assert not store.client.has_collection(store.collection_name)
    store.client.close()


@pytest.mark.asyncio
async def test_milvus_rejects_incompatible_collection(tmp_path: Path) -> None:
    store = MilvusVectorStore(uri=str(tmp_path / "milvus.db"))
    await store.add_texts_and_embeddings(make_texts())
    incompatible_store = MilvusVectorStore(
        client=store.client,
        uri=store.uri,
        collection_name=store.collection_name,
    )
    incompatible_text = Text(
        text="wrong dimension",
        name="wrong",
        doc=Doc(docname="wrong", citation="Wrong", dockey="wrong"),
        embedding=[1.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError, match="incompatible schema"):
        await incompatible_store.add_texts_and_embeddings([incompatible_text])

    store.clear()
    store.client.close()


def test_milvus_missing_dependency_error() -> None:
    with (
        patch("paperqa.llms.import_module", side_effect=ModuleNotFoundError),
        pytest.raises(ImportError, match=r"paper-qa\[milvus\]"),
    ):
        MilvusVectorStore()


def test_milvus_lite_3_score_normalization() -> None:
    with patch("paperqa.llms.version", return_value="3.0.0"):
        assert MilvusVectorStore._normalize_score("./milvus.db", 0.25) == 0.75
        assert MilvusVectorStore._normalize_score("https://example.com", 0.25) == 0.25


@pytest.mark.asyncio
async def test_docs_with_milvus_vector_store(tmp_path: Path) -> None:
    store = MilvusVectorStore(uri=str(tmp_path / "milvus.db"))
    docs = Docs(texts_index=store)
    texts = make_texts()
    doc = texts[0].doc

    assert await docs.aadd_texts(texts=texts, doc=doc)
    matches = await docs.retrieve_texts(
        "alpha", k=2, embedding_model=StaticEmbeddingModel()
    )

    assert [match.text for match in matches] == ["alpha", "beta"]
    assert len(store.texts_hashes) == len(texts)

    docs.clear_docs()
    assert not store.client.has_collection(store.collection_name)
    store.client.close()
