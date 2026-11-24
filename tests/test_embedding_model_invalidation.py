from __future__ import annotations

import pytest
from lmi import EmbeddingModel, HybridEmbeddingModel
from pydantic import PrivateAttr

from paperqa import (
    Doc,
    Docs,
    NumpyVectorStore,
    QdrantVectorStore,
    Settings,
    Text,
    VectorStore,
)
from paperqa.llms import embedding_model_fingerprint


class RecordingEmbeddingModel(EmbeddingModel):
    """Small deterministic embedding model that records document batches."""

    vector: list[float]
    _document_batches: list[list[str]] = PrivateAttr(default_factory=list)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._document_batches.append(list(texts))
        return [list(self.vector) for _ in texts]

    @property
    def document_batches(self) -> list[list[str]]:
        return self._document_batches


def _make_docs() -> tuple[Docs, Doc, Text]:
    doc = Doc(docname="stub", citation="stub", dockey="stub")
    text = Text(text="The quick brown fox.", name="stub chunk 1", doc=doc)
    return Docs(), doc, text


@pytest.mark.offline
def test_embedding_model_fingerprint_tracks_model_identity(subtests) -> None:
    model = RecordingEmbeddingModel(
        name="shared-name",
        ndim=2,
        vector=[1.0, 0.0],
        config={"api_base": "https://example-a.test"},
    )

    with subtests.test(msg="stable-for-equal-model-configuration"):
        assert embedding_model_fingerprint(model) == embedding_model_fingerprint(
            model.model_copy()
        )
        assert embedding_model_fingerprint(
            Settings.from_name("fast").get_embedding_model()
        ) == embedding_model_fingerprint(
            Settings.from_name("fast").get_embedding_model()
        )

    with subtests.test(msg="changes-with-configuration"):
        other_config = model.model_copy(
            update={"config": {"api_base": "https://example-b.test"}}
        )
        assert embedding_model_fingerprint(model) != embedding_model_fingerprint(
            other_config
        )

    with subtests.test(msg="changes-with-output-shape"):
        other_shape = RecordingEmbeddingModel(
            name="shared-name",
            ndim=3,
            vector=[1.0, 0.0, 0.0],
            config=model.config,
        )
        assert embedding_model_fingerprint(model) != embedding_model_fingerprint(
            other_shape
        )


@pytest.mark.offline
def test_embedding_model_fingerprint_tracks_nested_model_identity() -> None:
    def make_hybrid(dense_name: str, dense_vector: list[float]) -> HybridEmbeddingModel:
        return HybridEmbeddingModel(
            name="hybrid",
            models=[
                RecordingEmbeddingModel(name=dense_name, ndim=2, vector=dense_vector),
                RecordingEmbeddingModel(name="sparse", ndim=1, vector=[1.0]),
            ],
        )

    model = make_hybrid("dense-a", [1.0, 0.0])

    assert embedding_model_fingerprint(model) == embedding_model_fingerprint(
        model.model_copy(deep=True)
    )
    assert embedding_model_fingerprint(model) != embedding_model_fingerprint(
        make_hybrid("dense-b", [1.0, 0.0])
    )
    assert embedding_model_fingerprint(model) != embedding_model_fingerprint(
        make_hybrid("dense-a", [0.0, 1.0])
    )


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector_store",
    [NumpyVectorStore, QdrantVectorStore],
)
async def test_changing_embedding_model_reembeds_same_shape_texts(
    vector_store: type[VectorStore],
) -> None:
    docs, doc, text = _make_docs()
    docs.texts_index = vector_store()
    first_model = RecordingEmbeddingModel(
        name="same-shape-a", ndim=2, vector=[1.0, 0.0]
    )
    second_model = RecordingEmbeddingModel(
        name="same-shape-b", ndim=2, vector=[0.0, 1.0]
    )

    await docs.aadd_texts([text], doc, embedding_model=first_model)
    assert text.embedding == first_model.vector
    assert text.embedding_model_fingerprint == embedding_model_fingerprint(first_model)

    matches = await docs.retrieve_texts("query", k=1, embedding_model=second_model)

    assert matches == [text]
    assert text.embedding == second_model.vector
    assert text.embedding_model_fingerprint == embedding_model_fingerprint(second_model)
    assert first_model.document_batches == [[text.text]]
    assert second_model.document_batches == [[text.text], ["query"]]

    await docs.retrieve_texts("another query", k=1, embedding_model=second_model)
    assert len(docs.texts_index) == 1
    assert second_model.document_batches == [
        [text.text],
        ["query"],
        ["another query"],
    ]


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector_store",
    [NumpyVectorStore, QdrantVectorStore],
)
async def test_changing_embedding_model_reembeds_different_shape_texts(
    vector_store: type[VectorStore],
) -> None:
    docs, doc, text = _make_docs()
    docs.texts_index = vector_store()
    first_model = RecordingEmbeddingModel(
        name="two-dimensional", ndim=2, vector=[1.0, 0.0]
    )
    second_model = RecordingEmbeddingModel(
        name="three-dimensional", ndim=3, vector=[0.0, 0.0, 1.0]
    )

    await docs.aadd_texts([text], doc, embedding_model=first_model)
    matches = await docs.retrieve_texts("query", k=1, embedding_model=second_model)

    assert matches == [text]
    assert len(text.embedding) == 3
    assert text.embedding == second_model.vector
    assert text.embedding_model_fingerprint == embedding_model_fingerprint(second_model)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_legacy_embedding_without_fingerprint_is_reembedded() -> None:
    docs, doc, text = _make_docs()
    model = RecordingEmbeddingModel(name="legacy-model", ndim=2, vector=[1.0, 0.0])

    await docs.aadd_texts([text], doc, embedding_model=model)
    text.embedding_model_fingerprint = None

    matches = await docs.retrieve_texts("query", k=1, embedding_model=model)

    assert matches == [text]
    assert text.embedding_model_fingerprint == embedding_model_fingerprint(model)
    assert model.document_batches == [[text.text], [text.text], ["query"]]
