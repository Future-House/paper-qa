from __future__ import annotations

import asyncio
import datetime
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from inspect import signature
from typing import Any, AsyncGenerator, Callable, Coroutine, Sequence, cast

import numpy as np
import tiktoken
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .prompts import default_system_prompt
from .types import Doc, Embeddable, LLMResult, Text
from .utils import batch_iter, flatten, gather_with_concurrency, is_coroutine_callable

# only works for python 3.11
# def guess_model_type(model_name: str) -> str:
#     """Guess the model type from the model name for OpenAI models"""
#     import openai

#     model_type = get_type_hints(
#         openai.types.chat.completion_create_params.CompletionCreateParamsBase
#     )["model"]
#     model_union = get_args(get_args(model_type)[1])
#     model_arr = list(model_union)
#     if model_name in model_arr:
#         return "chat"
#     return "completion"

# def is_openai_model(model_name):
#     import openai

#     model_type = get_type_hints(
#         openai.types.chat.completion_create_params.CompletionCreateParamsBase
#     )["model"]
#     model_union = get_args(get_args(model_type)[1])
#     model_arr = list(model_union)

#     complete_model_types = get_type_hints(
#         openai.types.completion_create_params.CompletionCreateParamsBase
#     )["model"]
#     complete_model_union = get_args(get_args(complete_model_types)[1])
#     complete_model_arr = list(complete_model_union)

#     return model_name in model_arr or model_name in complete_model_arr

ANYSCALE_MODEL_PREFIXES: tuple[str, ...] = (
    "meta-llama/Meta-Llama-3-",
    "mistralai/Mistral-",
    "mistralai/Mixtral-",
)


def guess_model_type(model_name: str) -> str:  # noqa: PLR0911
    if model_name.startswith("babbage"):
        return "completion"
    if model_name.startswith("davinci"):
        return "completion"
    if (
        os.environ.get("ANYSCALE_API_KEY")
        and os.environ.get("ANYSCALE_BASE_URL")
        and (model_name.startswith(ANYSCALE_MODEL_PREFIXES))
    ):
        return "chat"
    if "instruct" in model_name:
        return "completion"
    if model_name.startswith("gpt-4"):
        if "base" in model_name:
            return "completion"
        return "chat"
    if model_name.startswith("gpt-3.5"):
        return "chat"
    return "completion"


def is_anyscale_model(model_name: str) -> bool:
    # compares prefixes with anyscale models
    # https://docs.anyscale.com/endpoints/text-generation/query-a-model/
    if (
        os.environ.get("ANYSCALE_API_KEY")
        and os.environ.get("ANYSCALE_BASE_URL")
        and model_name.startswith(ANYSCALE_MODEL_PREFIXES)
    ):
        return True
    return False


def is_openai_model(model_name: str) -> bool:
    return is_anyscale_model(model_name) or model_name.startswith(
        ("gpt-", "babbage", "davinci", "ft:gpt-")
    )


def process_llm_config(
    llm_config: dict, max_token_name: str = "max_tokens"  # noqa: S107
) -> dict:
    """Remove model_type and try to set max_tokens."""
    result = {k: v for k, v in llm_config.items() if k != "model_type"}
    if max_token_name not in result or result[max_token_name] == -1:
        model = llm_config["model"]
        # now we guess - we could use tiktoken to count,
        # but do have the initiative right now
        if model.startswith("gpt-4") or (
            model.startswith("gpt-3.5") and "1106" in model
        ):
            result[max_token_name] = 3000
        else:
            result[max_token_name] = 1500
    return result


async def embed_documents(
    client: AsyncOpenAI, texts: list[str], embedding_model: str, batch_size: int = 16
) -> list[list[float]]:
    """Embed a list of documents with batching."""
    if client is None:
        raise ValueError(
            "Your client is None - did you forget to set it after pickling?"
        )
    N = len(texts)
    embeddings = []
    for i in range(0, N, batch_size):
        response = await client.embeddings.create(
            model=embedding_model,
            input=texts[i : i + batch_size],
            encoding_format="float",
        )
        embeddings.extend([e.embedding for e in response.data])
    return embeddings


class EmbeddingModes(str, Enum):
    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingModel(ABC, BaseModel):
    name: str

    def set_mode(self, mode: EmbeddingModes) -> None:
        """Several embedding models have a 'mode' or prompt which affects output."""

    @abstractmethod
    async def embed_documents(self, client: Any, texts: list[str]) -> list[list[float]]:
        pass


class OpenAIEmbeddingModel(EmbeddingModel):
    name: str = Field(default="text-embedding-ada-002")

    async def embed_documents(self, client: Any, texts: list[str]) -> list[list[float]]:
        return await embed_documents(cast(AsyncOpenAI, client), texts, self.name)


class SparseEmbeddingModel(EmbeddingModel):
    """This is a very simple keyword search model - probably best to be mixed with others."""

    name: str = "sparse"
    ndim: int = 256
    enc: Any = Field(default_factory=lambda: tiktoken.get_encoding("cl100k_base"))

    async def embed_documents(self, client, texts) -> list[list[float]]:  # noqa: ARG002
        enc_batch = self.enc.encode_ordinary_batch(texts)
        # now get frequency of each token rel to length
        return [
            np.bincount([xi % self.ndim for xi in x], minlength=self.ndim).astype(float)
            / len(x)
            for x in enc_batch
        ]


class HybridEmbeddingModel(EmbeddingModel):
    name: str = "hybrid-embed"
    models: list[EmbeddingModel]

    async def embed_documents(self, client, texts):
        all_embeds = await asyncio.gather(
            *[m.embed_documents(client, texts) for m in self.models]
        )
        return np.concatenate(all_embeds, axis=1)


class VoyageAIEmbeddingModel(EmbeddingModel):
    """A wrapper around Voyage AI's client lib."""

    name: str = Field(default="voyage-large-2")
    embedding_type: EmbeddingModes = Field(default=EmbeddingModes.DOCUMENT)
    batch_size: int = 10

    def set_mode(self, mode: EmbeddingModes):
        self.embedding_type = mode

    async def embed_documents(self, client: Any, texts: list[str]) -> list[list[float]]:
        if client is None:
            raise ValueError(
                "Your client is None - did you forget to set it after pickling?"
            )
        N = len(texts)
        embeddings = []
        for i in range(0, N, self.batch_size):
            response = await client.embed(
                texts[i : i + self.batch_size],
                model=self.name,
                input_type=self.embedding_type.value,
            )
            embeddings.extend(response.embeddings)
        return embeddings


class LLMModel(ABC, BaseModel):
    llm_type: str | None = None
    name: str
    model_config = ConfigDict(extra="forbid")

    async def acomplete(self, client: Any, prompt: str) -> str:
        raise NotImplementedError

    async def acomplete_iter(self, client: Any, prompt: str) -> Any:
        """Return an async generator that yields chunks of the completion.

        I cannot get mypy to understand the override, so marked as Any
        """
        raise NotImplementedError

    async def achat(self, client: Any, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    async def achat_iter(self, client: Any, messages: list[dict[str, str]]) -> Any:
        """Return an async generator that yields chunks of the completion.

        I cannot get mypy to understand the override, so marked as Any
        """
        raise NotImplementedError

    def infer_llm_type(self, client: Any) -> str:  # noqa: ARG002
        return "completion"

    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # gross approximation

    def make_chain(  # noqa: C901, PLR0915
        self,
        client: Any,
        prompt: str,
        skip_system: bool = False,
        system_prompt: str = default_system_prompt,
    ) -> Callable[
        [dict, list[Callable[[str], None]] | None], Coroutine[Any, Any, LLMResult]
    ]:
        """Create a function to execute a batch of prompts.

        This replaces the previous use of langchain for combining prompts and LLMs.

        Args:
            client: a ephemeral client to use
            prompt: The prompt to use
            skip_system: Whether to skip the system prompt
            system_prompt: The system prompt to use

        Returns:
            A function to execute a prompt. Its signature is:
            execute(data: dict, callbacks: list[Callable[[str], None]]] | None = None) -> LLMResult
            where data is a dict with keys for the input variables that will be formatted into prompt
            and callbacks is a list of functions to call with each chunk of the completion.
        """
        # check if it needs to be set
        if self.llm_type is None:
            self.llm_type = self.infer_llm_type(client)
        if self.llm_type == "chat":
            system_message_prompt = {"role": "system", "content": system_prompt}
            human_message_prompt = {"role": "user", "content": prompt}
            chat_prompt = (
                [human_message_prompt]
                if skip_system
                else [system_message_prompt, human_message_prompt]
            )

            async def execute(
                data: dict,
                callbacks: list[Callable] | None = None,
            ) -> LLMResult:
                start_clock = asyncio.get_running_loop().time()
                result = LLMResult(
                    model=self.name,
                    date=datetime.datetime.now().isoformat(),
                )
                messages = []
                for m in chat_prompt:
                    messages.append(  # noqa: PERF401
                        {"role": m["role"], "content": m["content"].format(**data)}
                    )
                result.prompt = messages
                result.prompt_count = sum(
                    [self.count_tokens(m["content"]) for m in messages]
                ) + sum([self.count_tokens(m["role"]) for m in messages])

                if callbacks is None:
                    output = await self.achat(client, messages)
                else:
                    sync_callbacks = [
                        f for f in callbacks if not is_coroutine_callable(f)
                    ]
                    async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]
                    completion = self.achat_iter(client, messages)
                    text_result = []
                    async for chunk in completion:  # type: ignore[attr-defined]
                        if chunk:
                            if result.seconds_to_first_token == 0:
                                result.seconds_to_first_token = (
                                    asyncio.get_running_loop().time() - start_clock
                                )
                            text_result.append(chunk)
                            [await f(chunk) for f in async_callbacks]
                            [f(chunk) for f in sync_callbacks]
                    output = "".join(text_result)
                result.completion_count = self.count_tokens(output)
                result.text = output
                result.seconds_to_last_token = (
                    asyncio.get_running_loop().time() - start_clock
                )
                return result

            return execute
        elif self.llm_type == "completion":  # noqa: RET505
            completion_prompt = (
                prompt if skip_system else system_prompt + "\n\n" + prompt
            )

            async def execute(
                data: dict, callbacks: list[Callable] | None = None
            ) -> LLMResult:
                start_clock = asyncio.get_running_loop().time()
                result = LLMResult(
                    model=self.name,
                    date=datetime.datetime.now().isoformat(),
                )
                formatted_prompt = completion_prompt.format(**data)
                result.prompt_count = self.count_tokens(formatted_prompt)
                result.prompt = formatted_prompt
                if callbacks is None:
                    output = await self.acomplete(client, formatted_prompt)
                else:
                    sync_callbacks = [
                        f for f in callbacks if not is_coroutine_callable(f)
                    ]
                    async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]

                    completion = self.acomplete_iter(
                        client,
                        formatted_prompt,
                    )
                    text_result = []
                    async for chunk in completion:  # type: ignore[attr-defined]
                        if chunk:
                            if result.seconds_to_first_token == 0:
                                result.seconds_to_first_token = (
                                    asyncio.get_running_loop().time() - start_clock
                                )
                            text_result.append(chunk)
                            [await f(chunk) for f in async_callbacks]
                            [f(chunk) for f in sync_callbacks]
                    output = "".join(text_result)
                result.completion_count = self.count_tokens(output)
                result.text = output
                result.seconds_to_last_token = (
                    asyncio.get_running_loop().time() - start_clock
                )
                return result

            return execute
        raise ValueError(f"Unknown llm_type: {self.llm_type}")


class OpenAILLMModel(LLMModel):
    config: dict = Field(default={"model": "gpt-3.5-turbo", "temperature": 0.1})
    name: str = "gpt-3.5-turbo"

    def _check_client(self, client: Any) -> AsyncOpenAI:
        if client is None:
            raise ValueError(
                "Your client is None - did you forget to set it after pickling?"
            )
        if not isinstance(client, AsyncOpenAI):
            raise ValueError(  # noqa: TRY004
                f"Your client is not a required AsyncOpenAI client. It is a {type(client)}"
            )
        return client

    @model_validator(mode="after")
    @classmethod
    def guess_llm_type(cls, data: Any) -> Any:
        m = cast(OpenAILLMModel, data)
        m.llm_type = guess_model_type(m.config["model"])
        return m

    @model_validator(mode="after")
    @classmethod
    def set_model_name(cls, data: Any) -> Any:
        m = cast(OpenAILLMModel, data)
        m.name = m.config["model"]
        return m

    async def acomplete(self, client: Any, prompt: str) -> str:
        aclient = self._check_client(client)
        completion = await aclient.completions.create(
            prompt=prompt, **process_llm_config(self.config)
        )
        return completion.choices[0].text

    async def acomplete_iter(self, client: Any, prompt: str) -> Any:
        aclient = self._check_client(client)
        completion = await aclient.completions.create(
            prompt=prompt, **process_llm_config(self.config), stream=True
        )
        async for chunk in completion:
            yield chunk.choices[0].text

    async def achat(self, client: Any, messages: list[dict[str, str]]) -> str:
        aclient = self._check_client(client)
        completion = await aclient.chat.completions.create(
            messages=messages, **process_llm_config(self.config)
        )
        return completion.choices[0].message.content or ""

    async def achat_iter(self, client: Any, messages: list[dict[str, str]]) -> Any:
        aclient = self._check_client(client)
        completion = await aclient.chat.completions.create(
            messages=messages, **process_llm_config(self.config), stream=True
        )
        async for chunk in cast(AsyncGenerator, completion):
            yield chunk.choices[0].delta.content


try:
    from anthropic import AsyncAnthropic
    from anthropic.types import ContentBlockDeltaEvent
except ImportError:
    AsyncAnthropic = Any
    ContentBlockDeltaEvent = Any


class AnthropicLLMModel(LLMModel):
    config: dict = Field(
        default={"model": "claude-3-sonnet-20240229", "temperature": 0.1}
    )
    name: str = "claude-3-sonnet-20240229"

    def __init__(self, *args, **kwargs):
        if AsyncAnthropic is Any:
            raise ImportError("Please install anthropic to use this model")
        super().__init__(*args, **kwargs)

    def _check_client(self, client: Any) -> AsyncAnthropic:
        if client is None:
            raise ValueError(
                "Your client is None - did you forget to set it after pickling?"
            )
        if not isinstance(client, AsyncAnthropic):
            raise ValueError(  # noqa: TRY004
                f"Your client is not a required AsyncAnthropic client. It is a {type(client)}"
            )
        return client

    @model_validator(mode="after")
    @classmethod
    def set_llm_type(cls, data: Any) -> Any:
        m = cast(AnthropicLLMModel, data)
        m.llm_type = "chat"
        return m

    @model_validator(mode="after")
    @classmethod
    def set_model_name(cls, data: Any) -> Any:
        m = cast(AnthropicLLMModel, data)
        m.name = m.config["model"]
        return m

    async def achat(self, client: Any, messages: list[dict[str, str]]) -> str:
        aclient = self._check_client(client)
        # filter out system
        sys_message = next(
            (m["content"] for m in messages if m["role"] == "system"), None
        )
        # BECAUSE THEY DO NOT USE NONE TO INDICATE SENTINEL
        # LIKE ANY SANE PERSON
        if sys_message:
            completion = await aclient.messages.create(
                system=sys_message,
                messages=[m for m in messages if m["role"] != "system"],
                **process_llm_config(self.config, "max_tokens"),
            )
        else:
            completion = await aclient.messages.create(
                messages=[m for m in messages if m["role"] != "system"],
                **process_llm_config(self.config, "max_tokens"),
            )
        return str(completion.content) or ""

    async def achat_iter(self, client: Any, messages: list[dict[str, str]]) -> Any:
        aclient = self._check_client(client)
        sys_message = next(
            (m["content"] for m in messages if m["role"] == "system"), None
        )
        if sys_message:
            completion = await aclient.messages.create(
                stream=True,
                system=sys_message,
                messages=[m for m in messages if m["role"] != "system"],
                **process_llm_config(self.config, "max_tokens"),
            )
        else:
            completion = await aclient.messages.create(
                stream=True,
                messages=[m for m in messages if m["role"] != "system"],
                **process_llm_config(self.config, "max_tokens"),
            )
        async for event in completion:
            if isinstance(event, ContentBlockDeltaEvent):
                yield event.delta.text
            # yield event.message.content


class LlamaEmbeddingModel(EmbeddingModel):
    embedding_model: str = Field(default="llama")

    batch_size: int = Field(default=4)
    concurrency: int = Field(default=1)

    async def embed_documents(self, client: Any, texts: list[str]) -> list[list[float]]:
        cast(AsyncOpenAI, client)

        async def process(texts: list[str]) -> list[float]:
            for i in range(3):  # noqa: B007
                # access httpx client directly to avoid type casting
                response = await client._client.post(
                    client.base_url.join("../embedding"), json={"content": texts}
                )
                body = response.json()
                if len(texts) == 1:
                    if (
                        type(body) != dict  # noqa: E721
                        or body.get("embedding") is None
                    ):
                        continue
                    return [body["embedding"]]
                else:  # noqa: RET505
                    if type(body) != list or body[0] != "results":  # noqa: E721
                        continue
                    return [e["embedding"] for e in body[1]]
            raise ValueError("Failed to embed documents - response was ", body)

        return flatten(
            await gather_with_concurrency(
                self.concurrency,
                [process(b) for b in batch_iter(texts, self.batch_size)],
            )
        )


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    name: str = Field(default="multi-qa-MiniLM-L6-cos-v1")
    _model: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Please install sentence-transformers to use this model"
            ) from exc

        self._model = SentenceTransformer(self.name)

    async def embed_documents(
        self, client: Any, texts: list[str]  # noqa: ARG002
    ) -> list[list[float]]:
        from sentence_transformers import SentenceTransformer

        return cast(SentenceTransformer, self._model).encode(texts)


def cosine_similarity(a, b):
    norm_product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return a @ b.T / norm_product


class VectorStore(BaseModel, ABC):
    """Interface for vector store - very similar to LangChain's VectorStore to be compatible."""

    embedding_model: EmbeddingModel = Field(default=OpenAIEmbeddingModel())
    # can be tuned for different tasks
    mmr_lambda: float = Field(default=0.9)
    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def add_texts_and_embeddings(self, texts: Sequence[Embeddable]) -> None:
        pass

    @abstractmethod
    async def similarity_search(
        self, client: Any, query: str, k: int
    ) -> tuple[Sequence[Embeddable], list[float]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    async def max_marginal_relevance_search(  # noqa: D417
        self, client: Any, query: str, k: int, fetch_k: int
    ) -> tuple[Sequence[Embeddable], list[float]]:
        """Vectorized implementation of Maximal Marginal Relevance (MMR) search.

        Args:
            query: Query vector.
            k: Number of results to return.

        Returns:
            List of tuples (doc, score) of length k.
        """
        if fetch_k < k:
            raise ValueError("fetch_k must be greater or equal to k")

        texts, scores = await self.similarity_search(client, query, fetch_k)
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
    texts: list[Embeddable] = []
    _embeddings_matrix: np.ndarray | None = None

    def clear(self) -> None:
        self.texts = []
        self._embeddings_matrix = None

    def add_texts_and_embeddings(
        self,
        texts: Sequence[Embeddable],
    ) -> None:
        self.texts.extend(texts)
        self._embeddings_matrix = np.array([t.embedding for t in self.texts])

    async def similarity_search(
        self, client: Any, query: str, k: int
    ) -> tuple[Sequence[Embeddable], list[float]]:
        k = min(k, len(self.texts))
        if k == 0:
            return [], []

        # this will only affect models that embedding prompts
        self.embedding_model.set_mode(EmbeddingModes.QUERY)

        np_query = np.array(
            (await self.embedding_model.embed_documents(client, [query]))[0]
        )

        self.embedding_model.set_mode(EmbeddingModes.DOCUMENT)

        similarity_scores = cosine_similarity(
            np_query.reshape(1, -1), self._embeddings_matrix
        )[0]
        similarity_scores = np.nan_to_num(similarity_scores, nan=-np.inf)
        # minus so descending
        # we could use arg-partition here
        # but a lot of algorithms expect a sorted list
        sorted_indices = np.argsort(-similarity_scores)
        return (
            [self.texts[i] for i in sorted_indices[:k]],
            [similarity_scores[i] for i in sorted_indices[:k]],
        )


class LangchainLLMModel(LLMModel):
    """A wrapper around the wrapper langchain."""

    config: dict = Field(default={"temperature": 0.1})
    name: str = "langchain"

    def infer_llm_type(self, client: Any) -> str:
        from langchain_core.language_models.chat_models import BaseChatModel

        self.name = client.model_name
        if isinstance(client, BaseChatModel):
            return "chat"
        return "completion"

    async def acomplete(self, client: Any, prompt: str) -> str:
        return await client.ainvoke(prompt, **self.config)

    async def acomplete_iter(self, client: Any, prompt: str) -> Any:
        async for chunk in cast(AsyncGenerator, client.astream(prompt, **self.config)):
            yield chunk

    async def achat(self, client: Any, messages: list[dict[str, str]]) -> str:
        from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

        lc_messages: list[BaseMessage] = []
        for m in messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                raise ValueError(f"Unknown role: {m['role']}")
        return (await client.ainvoke(lc_messages, **self.config)).content

    async def achat_iter(self, client: Any, messages: list[dict[str, str]]) -> Any:
        from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

        lc_messages: list[BaseMessage] = []
        for m in messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                raise ValueError(f"Unknown role: {m['role']}")
        async for chunk in client.astream(lc_messages, **self.config):
            yield chunk.content


class LangchainEmbeddingModel(EmbeddingModel):
    """A wrapper around the wrapper langchain."""

    name: str = "langchain"

    async def embed_documents(self, client: Any, texts: list[str]) -> list[list[float]]:
        return await client.aembed_documents(texts)


class LangchainVectorStore(VectorStore):
    """A wrapper around the wrapper langchain.

    Note that if this is cleared (e.g., by `Docs` having `jit_texts_index` set to True),
    this will calls the `from_texts` class method on the `store`. This means that any non-default
    constructor arguments will be lost. You can override the clear method on this class.
    """

    _store_builder: Any | None = None
    _store: Any | None = None
    # JIT Generics - store the class type (Doc or Text)
    class_type: type[Embeddable] = Field(default=Embeddable)
    model_config = ConfigDict(extra="forbid")

    def __init__(self, **data):
        # we have to separate out store from the rest of the data
        # because langchain objects are not serializable
        store_builder = None
        if "store_builder" in data:
            store_builder = LangchainVectorStore.check_store_builder(
                data.pop("store_builder")
            )
        if "cls" in data and "embedding_model" in data:
            # make a little closure
            cls = data.pop("cls")
            embedding_model = data.pop("embedding_model")

            def candidate(x, y):
                return cls.from_embeddings(x, embedding_model, y)

            store_builder = LangchainVectorStore.check_store_builder(candidate)
        super().__init__(**data)
        self._store_builder = store_builder

    @classmethod
    def check_store_builder(cls, builder: Any) -> Any:
        # check it is a callable
        if not callable(builder):
            raise ValueError("store_builder must be callable")  # noqa: TRY004
        # check it takes two arguments
        # we don't use type hints because it could be
        # a partial
        sig = signature(builder)
        if len(sig.parameters) != 2:  # noqa: PLR2004
            raise ValueError("store_builder must take two arguments")
        return builder

    def __getstate__(self):
        state = super().__getstate__()
        # remove non-serializable private attributes
        del state["__pydantic_private__"]["_store"]
        del state["__pydantic_private__"]["_store_builder"]
        return state

    def __setstate__(self, state):
        # restore non-serializable private attributes
        state["__pydantic_private__"]["_store"] = None
        state["__pydantic_private__"]["_store_builder"] = None
        super().__setstate__(state)

    def add_texts_and_embeddings(self, texts: Sequence[Embeddable]) -> None:
        if self._store_builder is None:
            raise ValueError("You must set store_builder before adding texts")
        self.class_type = type(texts[0])
        if self.class_type == Text:
            vec_store_text_and_embeddings = [
                (x.text, x.embedding) for x in cast(list[Text], texts)
            ]
        elif self.class_type == Doc:
            vec_store_text_and_embeddings = [
                (x.citation, x.embedding) for x in cast(list[Doc], texts)
            ]
        else:
            raise ValueError("Only embeddings of type Text are supported")
        if self._store is None:
            self._store = self._store_builder(
                vec_store_text_and_embeddings,
                texts,
            )
            if self._store is None or not hasattr(self._store, "add_embeddings"):
                raise ValueError("store_builder did not return a valid vectorstore")
        self._store.add_embeddings(
            vec_store_text_and_embeddings,
            metadatas=texts,
        )

    async def similarity_search(
        self, client: Any, query: str, k: int  # noqa: ARG002
    ) -> tuple[Sequence[Embeddable], list[float]]:
        if self._store is None:
            return [], []
        results = await self._store.asimilarity_search_with_relevance_scores(query, k=k)
        texts, scores = [self.class_type(**r[0].metadata) for r in results], [
            r[1] for r in results
        ]
        return texts, scores

    def clear(self) -> None:
        del self._store  # be explicit, because it could be large
        self._store = None


def get_score(text: str) -> int:
    # check for N/A
    last_line = text.split("\n")[-1]
    if "N/A" in last_line or "n/a" in last_line or "NA" in last_line:
        return 0
    score = re.search(r"[sS]core[:is\s]+([0-9]+)", text)
    if not score:
        score = re.search(r"\(([0-9])\w*\/", text)
    if not score:
        score = re.search(r"([0-9]+)\w*\/", text)
    if score:
        s = int(score.group(1))
        if s > 10:  # noqa: PLR2004
            s = int(s / 10)  # sometimes becomes out of 100
        return s
    last_few = text[-15:]
    scores = re.findall(r"([0-9]+)", last_few)
    if scores:
        s = int(scores[-1])
        if s > 10:  # noqa: PLR2004
            s = int(s / 10)  # sometimes becomes out of 100
        return s
    if len(text) < 100:  # noqa: PLR2004
        return 1
    return 5


def llm_model_factory(llm: str) -> LLMModel:
    if llm != "default":
        if is_openai_model(llm):
            return OpenAILLMModel(config={"model": llm})
        elif llm.startswith("langchain"):  # noqa: RET505
            return LangchainLLMModel()
        elif "claude" in llm:
            return AnthropicLLMModel(config={"model": llm})
        else:
            raise ValueError(f"Could not guess model type for {llm}. ")
    return OpenAILLMModel()


def embedding_model_factory(embedding: str, **kwargs) -> EmbeddingModel:
    if embedding == "langchain":
        return LangchainEmbeddingModel(**kwargs)
    if embedding == "sentence-transformers":
        return SentenceTransformerEmbeddingModel(**kwargs)
    if embedding.startswith("voyage"):
        return VoyageAIEmbeddingModel(name=embedding, **kwargs)
    if embedding.startswith("hybrid"):
        embedding_model_name = "-".join(embedding.split("-")[1:])
        dense_model = (
            OpenAIEmbeddingModel(name=embedding_model_name)
            if not embedding_model_name.startswith("voyage")
            else VoyageAIEmbeddingModel(name=embedding_model_name, **kwargs)
        )
        return HybridEmbeddingModel(
            models=[
                dense_model,
                SparseEmbeddingModel(**kwargs),
            ]
        )
    if embedding == "sparse":
        return SparseEmbeddingModel(**kwargs)
    return OpenAIEmbeddingModel(name=embedding, **kwargs)


def vector_store_factory(embedding: str) -> NumpyVectorStore:
    return NumpyVectorStore(embedding_model=embedding_model_factory(embedding))
