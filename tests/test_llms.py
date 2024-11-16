import pathlib
import pickle
from typing import Any
from unittest.mock import patch

import litellm
import pytest

from paperqa import (
    HybridEmbeddingModel,
    LiteLLMEmbeddingModel,
    LiteLLMModel,
    SentenceTransformerEmbeddingModel,
    SparseEmbeddingModel,
    embedding_model_factory,
)
from paperqa.llms import Chunk
from tests.conftest import VCR_DEFAULT_MATCH_ON


class TestLiteLLMModel:
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "model_list": [
                        {
                            "model_name": "gpt-4o-mini",
                            "litellm_params": {
                                "model": "gpt-4o-mini",
                                "temperature": 0,
                                "max_tokens": 56,
                            },
                        }
                    ]
                },
                id="with-router",
            ),
            pytest.param(
                {
                    "pass_through_router": True,
                    "router_kwargs": {"temperature": 0, "max_tokens": 56},
                },
                id="without-router",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_run_prompt(self, config: dict[str, Any]) -> None:
        llm = LiteLLMModel(name="gpt-4o-mini", config=config)

        outputs = []

        def accum(x) -> None:
            outputs.append(x)

        completion = await llm.run_prompt(
            prompt="The {animal} says",
            data={"animal": "duck"},
            system_prompt=None,
            callbacks=[accum],
        )
        assert completion.model == "gpt-4o-mini"
        assert completion.seconds_to_first_token > 0
        assert completion.prompt_count > 0
        assert completion.completion_count > 0
        assert str(completion) == "".join(outputs)
        assert completion.cost > 0

        completion = await llm.run_prompt(
            prompt="The {animal} says",
            data={"animal": "duck"},
            system_prompt=None,
        )
        assert completion.seconds_to_first_token == 0
        assert completion.seconds_to_last_token > 0
        assert completion.cost > 0

        # check with mixed callbacks
        async def ac(x) -> None:
            pass

        completion = await llm.run_prompt(
            prompt="The {animal} says",
            data={"animal": "duck"},
            system_prompt=None,
            callbacks=[accum, ac],
        )
        assert completion.cost > 0

    @pytest.mark.vcr
    @pytest.mark.parametrize(
        ("config", "bypassed_router"),
        [
            pytest.param(
                {
                    "model_list": [
                        {
                            "model_name": "gpt-4o-mini",
                            "litellm_params": {"model": "gpt-4o-mini", "max_tokens": 3},
                        }
                    ]
                },
                False,
                id="with-router",
            ),
            pytest.param(
                {"pass_through_router": True, "router_kwargs": {"max_tokens": 3}},
                True,
                id="without-router",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_max_token_truncation(
        self, config: dict[str, Any], bypassed_router: bool
    ) -> None:
        llm = LiteLLMModel(name="gpt-4o-mini", config=config)
        with patch(
            "litellm.Router.atext_completion",
            side_effect=litellm.Router.atext_completion,
            autospec=True,
        ) as mock_atext_completion:
            chunk = await llm.acomplete("Please tell me a story")  # type: ignore[call-arg]
        if bypassed_router:
            mock_atext_completion.assert_not_awaited()
        else:
            mock_atext_completion.assert_awaited_once()
        assert isinstance(chunk, Chunk)
        assert chunk.completion_tokens == 3
        assert chunk.text
        assert len(chunk.text) < 20

    def test_pickling(self, tmp_path: pathlib.Path) -> None:
        pickle_path = tmp_path / "llm_model.pickle"
        llm = LiteLLMModel(
            name="gpt-4o-mini",
            config={
                "model_list": [
                    {
                        "model_name": "gpt-4o-mini",
                        "litellm_params": {
                            "model": "gpt-4o-mini",
                            "temperature": 0,
                            "max_tokens": 56,
                        },
                    }
                ]
            },
        )
        with pickle_path.open("wb") as f:
            pickle.dump(llm, f)
        with pickle_path.open("rb") as f:
            rehydrated_llm = pickle.load(f)
        assert llm.name == rehydrated_llm.name
        assert llm.config == rehydrated_llm.config
        assert llm.router.deployment_names == rehydrated_llm.router.deployment_names


@pytest.mark.asyncio
async def test_embedding_model_factory_sentence_transformer() -> None:
    """Test that the factory creates a SentenceTransformerEmbeddingModel when given an 'st-' prefix."""
    embedding = "st-multi-qa-MiniLM-L6-cos-v1"
    model = embedding_model_factory(embedding)
    assert isinstance(
        model, SentenceTransformerEmbeddingModel
    ), "Factory did not create SentenceTransformerEmbeddingModel"
    assert model.name == "multi-qa-MiniLM-L6-cos-v1", "Incorrect model name assigned"

    # Test embedding functionality
    texts = ["Hello world", "Test sentence"]
    embeddings = await model.embed_documents(texts)
    assert len(embeddings) == 2, "Incorrect number of embeddings returned"
    assert all(
        isinstance(embed, list) for embed in embeddings
    ), "Embeddings are not in list format"
    assert all(len(embed) > 0 for embed in embeddings), "Embeddings should not be empty"


@pytest.mark.asyncio
async def test_embedding_model_factory_hybrid_with_sentence_transformer() -> None:
    """Test that the factory creates a HybridEmbeddingModel containing a SentenceTransformerEmbeddingModel."""
    embedding = "hybrid-st-multi-qa-MiniLM-L6-cos-v1"
    model = embedding_model_factory(embedding)
    assert isinstance(
        model, HybridEmbeddingModel
    ), "Factory did not create HybridEmbeddingModel"
    assert len(model.models) == 2, "Hybrid model should contain two component models"
    assert isinstance(
        model.models[0], SentenceTransformerEmbeddingModel
    ), "First component should be SentenceTransformerEmbeddingModel"
    assert isinstance(
        model.models[1], SparseEmbeddingModel
    ), "Second component should be SparseEmbeddingModel"

    # Test embedding functionality
    texts = ["Hello world", "Test sentence"]
    embeddings = await model.embed_documents(texts)
    assert len(embeddings) == 2, "Incorrect number of embeddings returned"
    expected_length = len((await model.models[0].embed_documents(texts))[0]) + len(
        (await model.models[1].embed_documents(texts))[0]
    )
    assert all(
        len(embed) == expected_length for embed in embeddings
    ), "Embeddings do not match expected combined length"


@pytest.mark.asyncio
async def test_embedding_model_factory_invalid_st_prefix() -> None:
    """Test that the factory raises a ValueError when 'st-' prefix is provided without a model name."""
    embedding = "st-"
    with pytest.raises(
        ValueError,
        match="SentenceTransformer model name must be specified after 'st-'.",
    ):
        embedding_model_factory(embedding)


@pytest.mark.asyncio
async def test_embedding_model_factory_unknown_prefix() -> None:
    """Test that the factory defaults to LiteLLMEmbeddingModel when an unknown prefix is provided."""
    embedding = "unknown-prefix-model"
    model = embedding_model_factory(embedding)
    assert isinstance(
        model, LiteLLMEmbeddingModel
    ), "Factory did not default to LiteLLMEmbeddingModel for unknown prefix"
    assert model.name == "unknown-prefix-model", "Incorrect model name assigned"


@pytest.mark.asyncio
async def test_embedding_model_factory_sparse() -> None:
    """Test that the factory creates a SparseEmbeddingModel when 'sparse' is provided."""
    embedding = "sparse"
    model = embedding_model_factory(embedding)
    assert isinstance(
        model, SparseEmbeddingModel
    ), "Factory did not create SparseEmbeddingModel"
    assert model.name == "sparse", "Incorrect model name assigned"


@pytest.mark.asyncio
async def test_embedding_model_factory_litellm() -> None:
    """Test that the factory creates a LiteLLMEmbeddingModel when 'litellm-' prefix is provided."""
    embedding = "litellm-text-embedding-3-small"
    model = embedding_model_factory(embedding)
    assert isinstance(
        model, LiteLLMEmbeddingModel
    ), "Factory did not create LiteLLMEmbeddingModel"
    assert model.name == "text-embedding-3-small", "Incorrect model name assigned"


@pytest.mark.asyncio
async def test_embedding_model_factory_default() -> None:
    """Test that the factory defaults to LiteLLMEmbeddingModel when no known prefix is provided."""
    embedding = "default-model"
    model = embedding_model_factory(embedding)
    assert isinstance(
        model, LiteLLMEmbeddingModel
    ), "Factory did not default to LiteLLMEmbeddingModel"
    assert model.name == "default-model", "Incorrect model name assigned"
