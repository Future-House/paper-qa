import json
import pathlib
import pickle
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import litellm
import openai
import pytest

from paperqa import (
    AnthropicBatchLLMModel,
    HybridEmbeddingModel,
    LiteLLMEmbeddingModel,
    LiteLLMModel,
    OpenAIBatchLLMModel,
    SentenceTransformerEmbeddingModel,
    SparseEmbeddingModel,
    embedding_model_factory,
)
from paperqa.llms import (
    BatchStatus,
    Chunk,
)
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


class TestOpenAIBatchLLMModel:
    @pytest.fixture(scope="class")
    def config(self, request) -> dict[str, Any]:
        model_name = request.param
        return {
            "model": model_name,
            "temperature": 0.0,
            "max_tokens": 64,
            "batch_summary_time_limit": 24 * 60 * 60,
            "batch_polling_interval": 5,
        }

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param("gpt-4o-mini", id="chat-model"),
            pytest.param("gpt-3.5-turbo-instruct", id="completion-model"),
        ],
        indirect=True,
    )
    @pytest.mark.asyncio
    async def test_run_prompt(self, config: dict[str, Any], request) -> None:

        mock_client = AsyncMock(spec_set=openai.AsyncOpenAI())

        mock_file_id = "file-123"
        mock_client.files.create = AsyncMock(return_value=MagicMock(id=mock_file_id))

        mock_batch_id = "batch_123"
        mock_client.batches.create = AsyncMock(
            return_value=MagicMock(
                id=mock_batch_id, status=BatchStatus.PROGRESS.from_openai()
            )
        )

        if request.node.name == "test_run_prompt[completion-model]":
            batch_retrieve_calls = [
                MagicMock(
                    id=mock_batch_id,
                    status=BatchStatus.FAILURE.from_openai(),
                    errors=MagicMock(
                        data=[
                            MagicMock(
                                message=(
                                    "Batch failed: The model gpt-3.5-turbo-instruct "
                                    "is not supported for batch completions."
                                )
                            )
                        ]
                    ),
                ),
            ]
        elif request.node.name == "test_run_prompt[chat-model]":
            batch_retrieve_calls = [
                MagicMock(id=mock_batch_id, status=BatchStatus.PROGRESS.from_openai()),
                MagicMock(
                    id=mock_batch_id,
                    status=BatchStatus.COMPLETE.from_openai(),
                    output_file_id="file-789",
                ),
            ]
        mock_client.batches.retrieve = AsyncMock(side_effect=batch_retrieve_calls)

        sample_responses = [
            {
                "id": "file-789",
                "custom_id": "0",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": (
                                        'The duck says "quack." This vocalization is characteristic of the species '
                                        "Anas platyrhynchos, commonly known as the mallard duck, which is often used "
                                        "as a representative example for the duck family, Anatidae."
                                    ),
                                    "refusal": None,
                                },
                                "logprobs": None,
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 46,
                            "completion_tokens": 47,
                            "total_tokens": 93,
                            "prompt_tokens_details": {
                                "cached_tokens": 0,
                                "audio_tokens": 0,
                            },
                            "completion_tokens_details": {
                                "reasoning_tokens": 0,
                                "audio_tokens": 0,
                                "accepted_prediction_tokens": 0,
                                "rejected_prediction_tokens": 0,
                            },
                        },
                    }
                },
            },
            {
                "id": "file-789",
                "custom_id": "1",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": (
                                        'The dog says "bark." This is a vocalization '
                                        "commonly associated with canines, used for "
                                        "communication purposes such as alerting, expressing "
                                        "excitement, or seeking attention."
                                    ),
                                    "refusal": None,
                                },
                                "logprobs": None,
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 46,
                            "completion_tokens": 34,
                            "total_tokens": 80,
                            "prompt_tokens_details": {
                                "cached_tokens": 0,
                                "audio_tokens": 0,
                            },
                            "completion_tokens_details": {
                                "reasoning_tokens": 0,
                                "audio_tokens": 0,
                                "accepted_prediction_tokens": 0,
                                "rejected_prediction_tokens": 0,
                            },
                        },
                    }
                },
            },
            {
                "id": "file-789",
                "custom_id": "2",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": (
                                        'It seems you\'re quoting or referencing "the cat says." '
                                        "If you're looking for a specific context, such as a phrase, a song, "
                                        "or a scientific observation (like feline vocalizations), please provide "
                                        "more details for a precise response."
                                    ),
                                    "refusal": None,
                                },
                                "logprobs": None,
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 46,
                            "completion_tokens": 46,
                            "total_tokens": 92,
                            "prompt_tokens_details": {
                                "cached_tokens": 0,
                                "audio_tokens": 0,
                            },
                            "completion_tokens_details": {
                                "reasoning_tokens": 0,
                                "audio_tokens": 0,
                                "accepted_prediction_tokens": 0,
                                "rejected_prediction_tokens": 0,
                            },
                        },
                    }
                },
            },
        ]

        response_data = "\n".join(json.dumps(resp) for resp in sample_responses)
        mock_response_content = MagicMock()
        mock_response_content.read.return_value = response_data.encode()
        mock_client.files.content = AsyncMock(return_value=mock_response_content)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            llm = OpenAIBatchLLMModel(name=config["model"], config=config)

            outputs = []

            def accum(x) -> None:
                outputs.append(x)

            async def ac(x) -> None:
                pass

            data = [{"animal": "duck"}, {"animal": "dog"}, {"animal": "cat"}]

            if request.node.name == "test_run_prompt[completion-model]":
                with pytest.raises(RuntimeError) as e_info:
                    completion = await llm.run_prompt(
                        prompt="The {animal} says",
                        data=data,
                    )
                assert "Batch failed" in str(e_info.value)
                assert "not supported" in str(e_info.value)

            if request.node.name == "test_run_prompt[chat-model]":
                completion = await llm.run_prompt(
                    prompt="The {animal} says",
                    data=data,
                    callbacks=[accum, ac],
                )

                assert all(
                    completion[k].model == config["model"] for k in range(len(data))
                )
                assert all(
                    completion[k].seconds_to_first_token > 0 for k in range(len(data))
                )
                assert all(completion[k].prompt_count > 0 for k in range(len(data)))
                assert all(completion[k].completion_count > 0 for k in range(len(data)))
                assert all(
                    completion[k].completion_count <= config["max_tokens"]
                    for k in range(len(data))
                )
                assert sum(comp.cost for comp in completion) > 0
                assert all(str(completion[k]) == outputs[k] for k in range(len(data)))

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param("gpt-4o-mini"),
        ],
        indirect=True,
    )
    def test_pickling(self, tmp_path: pathlib.Path, config: dict[str, Any]) -> None:
        pickle_path = tmp_path / "llm_model.pickle"
        llm = OpenAIBatchLLMModel(
            name="gpt-4o-mini",
            config=config,
        )
        with pickle_path.open("wb") as f:
            pickle.dump(llm, f)
        with pickle_path.open("rb") as f:
            rehydrated_llm = pickle.load(f)
        assert llm.name == rehydrated_llm.name
        assert llm.config == rehydrated_llm.config


class TestAnthropicBatchLLMModel:
    @pytest.fixture(scope="class")
    def config(self, request) -> dict[str, Any]:
        model_name = request.param
        return {
            "model": model_name,
            "temperature": 0.0,
            "max_tokens": 64,
            "batch_summary_time_limit": 24 * 60 * 60,
            "batch_polling_interval": 5,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param("claude-3-haiku-20240307", id="chat-model"),
        ],
        indirect=True,
    )
    async def test_run_prompt(self, config: dict[str, Any]) -> None:

        mock_client = AsyncMock(spec_set=anthropic.AsyncAnthropic())

        mock_client = MagicMock()
        mock_batches = MagicMock()
        mock_client.beta.messages.batches = mock_batches

        mock_batch_id = "msgbatch_123"
        mock_batches.create = AsyncMock(
            return_value=MagicMock(
                id=mock_batch_id,
                processing_status=BatchStatus.PROGRESS.from_anthropic(),
            ),
        )

        batch_retrieve_call = [
            MagicMock(
                id=mock_batch_id,
                processing_status=BatchStatus.PROGRESS.from_anthropic(),
            ),
            MagicMock(
                id=mock_batch_id,
                processing_status=BatchStatus.COMPLETE.from_anthropic(),
            ),
        ]
        mock_batches.retrieve = AsyncMock(side_effect=batch_retrieve_call)

        mock_responses = [
            MagicMock(
                custom_id="0",
                result=MagicMock(
                    message=MagicMock(
                        id="msg_0143L9rPswgaUyENkHkPJLcn",
                        content=[
                            MagicMock(
                                text=(
                                    "I don't actually hear any ducks saying anything. "
                                    "As an AI assistant, I don't have the ability to hear or interpret "
                                    "sounds from the physical world. I can only respond based on the text "
                                    "you provide to me through this chat interface. "
                                    "If you'd like, you can tell me what you think the duck is"
                                ),
                            )
                        ],
                        model="claude-3-haiku-20240307",
                        role="assistant",
                        stop_reason="max_tokens",
                        stop_sequence=None,
                        type="message",
                        usage=MagicMock(input_tokens=10, output_tokens=64),
                    ),
                    type="succeeded",
                ),
            ),
            MagicMock(
                custom_id="1",
                result=MagicMock(
                    message=MagicMock(
                        id="msg_01KujiHEB5S8pfRUCmrbabu4",
                        content=[
                            MagicMock(
                                text=(
                                    "Unfortunately, I don't actually hear a dog speaking. "
                                    "As an AI assistant without physical senses, I"
                                    "can't directly perceive animals making sounds. "
                                    "Could you please provide more context about what the "
                                    "dog is saying, or what you would like me to respond to "
                                    "regarding the dog? I'd be happy to try to assist"
                                ),
                            )
                        ],
                        model="claude-3-haiku-20240307",
                        role="assistant",
                        stop_reason="max_tokens",
                        stop_sequence=None,
                        type="message",
                        usage=MagicMock(input_tokens=10, output_tokens=64),
                    ),
                    type="succeeded",
                ),
            ),
            MagicMock(
                custom_id="2",
                result=MagicMock(
                    message=MagicMock(
                        id="msg_01Pf2LqV7wjnwqerkZubbofA",
                        content=[
                            MagicMock(
                                text=(
                                    "I'm afraid I don't actually hear a cat speaking. "
                                    "As an AI assistant, I don't have the ability to hear "
                                    "or communicate with animals directly. I can only respond "
                                    "based on the text you provide to me. If you'd "
                                    "like, you can tell me what you imagine the cat is saying, and I'll"
                                ),
                            )
                        ],
                        model="claude-3-haiku-20240307",
                        role="assistant",
                        stop_reason="max_tokens",
                        stop_sequence=None,
                        type="message",
                        usage=MagicMock(input_tokens=10, output_tokens=64),
                    ),
                    type="succeeded",
                ),
            ),
        ]

        # Create a generator function
        def mock_results_generator(_batch_id):

            yield from mock_responses

        mock_batches.results = AsyncMock(
            return_value=mock_results_generator(mock_batch_id)
        )

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            llm = AnthropicBatchLLMModel(name=config["model"], config=config)

            data = [{"animal": "duck"}, {"animal": "dog"}, {"animal": "cat"}]

            completions = await llm.run_prompt(
                prompt="The {animal} says",
                data=data,
            )

            assert all(comp.model == config["model"] for comp in completions)
            assert all(comp.seconds_to_first_token > 0 for comp in completions)
            assert all(comp.prompt_count > 0 for comp in completions)
            assert all(comp.completion_count > 0 for comp in completions)
            assert all(
                comp.completion_count <= config["max_tokens"] for comp in completions
            )
            assert sum(comp.cost for comp in completions) > 0


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
