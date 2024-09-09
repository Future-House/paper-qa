from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Sequence
from enum import Enum
from inspect import signature
from typing import Any, cast

import numpy as np
import tiktoken
from litellm import Router, aembedding, token_counter
from pydantic import BaseModel, ConfigDict, Field, model_validator

from paperqa.prompts import default_system_prompt
from paperqa.types import Doc, Embeddable, LLMResult, Text
from paperqa.utils import is_coroutine_callable

PromptRunner = Callable[
    [dict, list[Callable[[str], None]] | None, str | None],
    Awaitable[LLMResult],
]


def expects_name_kwarg(func: Callable) -> bool:
    return "name" in signature(func).parameters


def prepare_args(func: Callable, chunk: str, name: str | None) -> tuple[tuple, dict]:
    if expects_name_kwarg(func):
        return (chunk,), {"name": name}
    return (chunk,), {}


async def do_callbacks(
    async_callbacks: list[Callable],
    sync_callbacks: list[Callable],
    chunk: str,
    name: str | None,
) -> None:
    for f in async_callbacks:
        args, kwargs = prepare_args(f, chunk, name)
        await f(*args, **kwargs)
    for f in sync_callbacks:
        args, kwargs = prepare_args(f, chunk, name)
        f(*args, **kwargs)


class EmbeddingModes(str, Enum):
    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingModel(ABC, BaseModel):
    name: str

    def set_mode(self, mode: EmbeddingModes) -> None:
        """Several embedding models have a 'mode' or prompt which affects output."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass


class LiteLLMEmbeddingModel(EmbeddingModel):
    name: str = Field(default="text-embedding-3-small")
    embedding_kwargs: dict = Field(default_factory=dict)

    async def embed_documents(
        self, texts: list[str], batch_size: int = 16
    ) -> list[list[float]]:
        N = len(texts)
        embeddings = []
        for i in range(0, N, batch_size):
            response = await aembedding(
                self.name, input=texts[i : i + batch_size], **self.embedding_kwargs
            )
            embeddings.extend([e["embedding"] for e in response.data])
        return embeddings


class SparseEmbeddingModel(EmbeddingModel):
    """This is a very simple keyword search model - probably best to be mixed with others."""

    name: str = "sparse"
    ndim: int = 256
    enc: Any = Field(default_factory=lambda: tiktoken.get_encoding("cl100k_base"))

    async def embed_documents(self, texts) -> list[list[float]]:
        enc_batch = self.enc.encode_ordinary_batch(texts)
        # now get frequency of each token rel to length
        return [
            np.bincount([xi % self.ndim for xi in x], minlength=self.ndim).astype(float)  # type: ignore[misc]
            / len(x)
            for x in enc_batch
        ]


class HybridEmbeddingModel(EmbeddingModel):
    name: str = "hybrid-embed"
    models: list[EmbeddingModel]

    async def embed_documents(self, texts):
        all_embeds = await asyncio.gather(
            *[m.embed_documents(texts) for m in self.models]
        )
        return np.concatenate(all_embeds, axis=1)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str | None
    prompt_tokens: int
    completion_tokens: int

    def __str__(self):
        return self.text


class LLMModel(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    llm_type: str | None = None
    name: str
    llm_result_callback: (
        Callable[[LLMResult], None] | Callable[[LLMResult], Awaitable[None]] | None
    ) = Field(
        default=None,
        description=(
            "An async callback that will be executed on each"
            " LLMResult (different than callbacks that execute on each chunk)"
        ),
        exclude=True,
    )
    config: dict = Field(default_factory=dict)

    async def acomplete(self, prompt: str) -> Chunk:
        """Return the completion as string and the number of tokens in the prompt and completion."""
        raise NotImplementedError

    async def acomplete_iter(self, prompt: str) -> AsyncIterable[Chunk]:  # noqa: ARG002
        """Return an async generator that yields chunks of the completion.

        Only the last tuple will be non-zero.
        """
        raise NotImplementedError
        if False:  # type: ignore[unreachable]
            yield  # Trick mypy: https://github.com/python/mypy/issues/5070#issuecomment-1050834495

    async def achat(self, messages: Iterable[dict[str, str]]) -> Chunk:
        """Return the completion as string and the number of tokens in the prompt and completion."""
        raise NotImplementedError

    async def achat_iter(
        self, messages: Iterable[dict[str, str]]  # noqa: ARG002
    ) -> AsyncIterable[Chunk]:
        """Return an async generator that yields chunks of the completion.

        Only the last tuple will be non-zero.
        """
        raise NotImplementedError
        if False:  # type: ignore[unreachable]
            yield  # Trick mypy: https://github.com/python/mypy/issues/5070#issuecomment-1050834495

    def infer_llm_type(self) -> str:
        return "completion"

    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # gross approximation

    async def run_prompt(
        self,
        prompt: str,
        data: dict,
        skip_system: bool = False,
        system_prompt: str = default_system_prompt,
        callbacks: list[Callable] | None = None,
        name: str | None = None,
    ) -> LLMResult:
        if self.llm_type is None:
            self.llm_type = self.infer_llm_type()
        if self.llm_type == "chat":
            return await self._run_chat(
                prompt, data, skip_system, system_prompt, callbacks, name
            )
        if self.llm_type == "completion":
            return await self._run_completion(
                prompt, data, skip_system, system_prompt, callbacks, name
            )
        raise ValueError(f"Unknown llm_type {self.llm_type!r}.")

    async def _run_chat(
        self,
        prompt: str,
        data: dict,
        skip_system: bool = False,
        system_prompt: str = default_system_prompt,
        callbacks: list[Callable] | None = None,
        name: str | None = None,
    ) -> LLMResult:
        """Run a chat prompt.

        Args:
            prompt: Prompt to use.
            data: Keys for the input variables that will be formatted into prompt.
            skip_system: Set True to skip the system prompt.
            system_prompt: System prompt to use.
            callbacks: Optional functions to call with each chunk of the completion.
            name: Optional name for the result.

        Returns:
            Result of the chat.
        """
        system_message_prompt = {"role": "system", "content": system_prompt}
        human_message_prompt = {"role": "user", "content": prompt}
        messages = [
            {"role": m["role"], "content": m["content"].format(**data)}
            for m in (
                [human_message_prompt]
                if skip_system
                else [system_message_prompt, human_message_prompt]
            )
        ]
        result = LLMResult(
            model=self.name,
            name=name,
            prompt=messages,
            prompt_count=(
                sum(self.count_tokens(m["content"]) for m in messages)
                + sum(self.count_tokens(m["role"]) for m in messages)
            ),
        )

        start_clock = asyncio.get_running_loop().time()
        if callbacks is None:
            chunk = await self.achat(messages)
            output = chunk.text
        else:
            sync_callbacks = [f for f in callbacks if not is_coroutine_callable(f)]
            async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]
            completion = self.achat_iter(messages)
            text_result = []
            async for chunk in completion:
                if chunk.text:
                    if result.seconds_to_first_token == 0:
                        result.seconds_to_first_token = (
                            asyncio.get_running_loop().time() - start_clock
                        )
                    text_result.append(chunk.text)
                    await do_callbacks(
                        async_callbacks, sync_callbacks, chunk.text, name
                    )
            output = "".join(text_result)
        usage = chunk.prompt_tokens, chunk.completion_tokens
        if sum(usage) > 0:
            result.prompt_count, result.completion_count = usage
        elif output:
            result.completion_count = self.count_tokens(output)
        result.text = output or ""
        result.seconds_to_last_token = asyncio.get_running_loop().time() - start_clock
        if self.llm_result_callback:
            if is_coroutine_callable(self.llm_result_callback):
                await self.llm_result_callback(result)  # type: ignore[misc]
            else:
                self.llm_result_callback(result)
        return result

    async def _run_completion(
        self,
        prompt: str,
        data: dict,
        skip_system: bool = False,
        system_prompt: str = default_system_prompt,
        callbacks: Iterable[Callable] | None = None,
        name: str | None = None,
    ) -> LLMResult:
        """Run a completion prompt.

        Args:
            prompt: Prompt to use.
            data: Keys for the input variables that will be formatted into prompt.
            skip_system: Set True to skip the system prompt.
            system_prompt: System prompt to use.
            callbacks: Optional functions to call with each chunk of the completion.
            name: Optional name for the result.

        Returns:
            Result of the completion.
        """
        formatted_prompt: str = (
            prompt if skip_system else system_prompt + "\n\n" + prompt
        ).format(**data)
        result = LLMResult(
            model=self.name,
            name=name,
            prompt=formatted_prompt,
            prompt_count=self.count_tokens(formatted_prompt),
        )

        start_clock = asyncio.get_running_loop().time()
        if callbacks is None:
            chunk = await self.acomplete(formatted_prompt)
            output = chunk.text
        else:
            sync_callbacks = [f for f in callbacks if not is_coroutine_callable(f)]
            async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]

            completion = self.acomplete_iter(formatted_prompt)
            text_result = []
            async for chunk in completion:
                if chunk.text:
                    if result.seconds_to_first_token == 0:
                        result.seconds_to_first_token = (
                            asyncio.get_running_loop().time() - start_clock
                        )
                    text_result.append(chunk.text)
                    await do_callbacks(
                        async_callbacks, sync_callbacks, chunk.text, name
                    )
            output = "".join(text_result)
        usage = chunk.prompt_tokens, chunk.completion_tokens
        if sum(usage) > 0:
            result.prompt_count, result.completion_count = usage
        elif output:
            result.completion_count = self.count_tokens(output)
        result.text = output or ""
        result.seconds_to_last_token = asyncio.get_running_loop().time() - start_clock
        if self.llm_result_callback:
            if is_coroutine_callable(self.llm_result_callback):
                await self.llm_result_callback(result)  # type: ignore[misc]
            else:
                self.llm_result_callback(result)
        return result


DEFAULT_VERTEX_SAFETY_SETTINGS: list[dict[str, str]] = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
]


class LiteLLMModel(LLMModel):
    """A wrapper around the litellm library.

    `config` should have two high level keys:
        `model_list`: stores a list of all model configurations
          (see https://docs.litellm.ai/docs/routing)
        `router_kwargs`: kwargs for the Router class

    This way users can specify routing strategies, retries, etc.

    """

    config: dict = Field(default_factory=dict)
    name: str = "gpt-4o-mini"
    _router: Router | None = None

    @model_validator(mode="before")
    @classmethod
    def maybe_set_config_attribute(cls, data: dict[str, Any]) -> dict[str, Any]:
        """If a user only gives a name, make a sensible config dict for them."""
        if "name" in data and "config" not in data:
            data["config"] = {
                "model_list": [
                    {
                        "model_name": data["name"],
                        "litellm_params": {"model": data["name"]}
                        | (
                            {}
                            if "gemini" not in data["name"]
                            else {"safety_settings": DEFAULT_VERTEX_SAFETY_SETTINGS}
                        ),
                    }
                ],
                "router_kwargs": {"num_retries": 3, "retry_after": 5},
            }
        # we only support one "model name" for now, here we validate
        if (
            "config" in data
            and len({m["model_name"] for m in data["config"]["model_list"]}) > 1
        ):
            raise ValueError("Only one model name per router is supported for now")
        return data

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the _router attribute as it's not picklable
        state["_router"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def router(self):
        if self._router is None:
            self._router = Router(
                model_list=self.config["model_list"],
                **self.config.get("router_kwargs", {}),
            )
        return self._router

    async def acomplete(self, prompt: str) -> Chunk:
        response = await self.router.atext_completion(model=self.name, prompt=prompt)
        return Chunk(
            text=response.choices[0].text,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    async def acomplete_iter(self, prompt: str) -> AsyncIterable[Chunk]:
        completion = await self.router.atext_completion(
            model=self.name,
            prompt=prompt,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in completion:
            yield Chunk(
                text=chunk.choices[0].text, prompt_tokens=0, completion_tokens=0
            )
        if hasattr(chunk, "usage") and hasattr(chunk.usage, "prompt_tokens"):
            yield Chunk(
                text=chunk.choices[0].text, prompt_tokens=0, completion_tokens=0
            )

    async def achat(self, messages: Iterable[dict[str, str]]) -> Chunk:
        response = await self.router.acompletion(self.name, messages)
        return Chunk(
            text=response.choices[0].message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    async def achat_iter(
        self, messages: Iterable[dict[str, str]]
    ) -> AsyncIterable[Chunk]:
        completion = await self.router.acompletion(
            self.name, messages, stream=True, stream_options={"include_usage": True}
        )
        async for chunk in completion:
            yield Chunk(
                text=chunk.choices[0].delta.content,
                prompt_tokens=0,
                completion_tokens=0,
            )
        if hasattr(chunk, "usage") and hasattr(chunk.usage, "prompt_tokens"):
            yield Chunk(
                text=None,
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
            )

    def infer_llm_type(self) -> str:
        if all(
            "text-completion" in m.get("litellm_params", {}).get("model", "")
            for m in self.config["model_list"]
        ):
            return "completion"
        return "chat"

    def count_tokens(self, text: str) -> int:
        return token_counter(model=self.name, text=text)


def cosine_similarity(a, b):
    norm_product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return a @ b.T / norm_product


class VectorStore(BaseModel, ABC):
    """Interface for vector store - very similar to LangChain's VectorStore to be compatible."""

    # can be tuned for different tasks
    mmr_lambda: float = Field(default=0.9)
    model_config = ConfigDict(extra="forbid")
    texts_hashes: set[int] = Field(default_factory=set)

    def __contains__(self, item):
        return hash(item) in self.texts_hashes

    def __len__(self):
        return len(self.texts_hashes)

    @abstractmethod
    def add_texts_and_embeddings(self, texts: Sequence[Embeddable]) -> None:
        [self.texts_hashes.add(hash(t)) for t in texts]  # type: ignore[func-returns-value]

    @abstractmethod
    async def similarity_search(
        self, query: str, k: int, embedding_model: EmbeddingModel
    ) -> tuple[Sequence[Embeddable], list[float]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    async def max_marginal_relevance_search(
        self, query: str, k: int, fetch_k: int, embedding_model: EmbeddingModel
    ) -> tuple[Sequence[Embeddable], list[float]]:
        """Vectorized implementation of Maximal Marginal Relevance (MMR) search.

        Args:
            query: Query vector.
            k: Number of results to return.
            fetch_k: Number of results to fetch from the vector store.
            embedding_model: model used to embed the query

        Returns:
            List of tuples (doc, score) of length k.
        """
        if fetch_k < k:
            raise ValueError("fetch_k must be greater or equal to k")

        texts, scores = await self.similarity_search(query, fetch_k, embedding_model)
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
        super().add_texts_and_embeddings(texts)
        self.texts.extend(texts)
        self._embeddings_matrix = np.array([t.embedding for t in self.texts])

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


# TODO: unsure if this is still supported via v5 release
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
        raise NotImplementedError(
            "Langchain has updated vectorstore internals and this is not yet supported"
        )
        # # we have to separate out store from the rest of the data
        # # because langchain objects are not serializable
        # store_builder = None
        # if "store_builder" in data:
        #     store_builder = LangchainVectorStore.check_store_builder(
        #         data.pop("store_builder")
        #     )
        # if "cls" in data and "embedding_model" in data:
        #     # make a little closure
        #     cls = data.pop("cls")
        #     embedding_model = data.pop("embedding_model")

        #     def candidate(x, y):
        #         return cls.from_embeddings(x, embedding_model, y)

        #     store_builder = LangchainVectorStore.check_store_builder(candidate)
        # super().__init__(**data)
        # self._store_builder = store_builder

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
        super().add_texts_and_embeddings(texts)
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
        self, query: str, k: int, embedding_model: EmbeddingModel  # noqa: ARG002
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


def embedding_model_factory(embedding: str, **kwargs) -> EmbeddingModel:

    if embedding.startswith("hybrid"):
        embedding_model_name = "-".join(embedding.split("-")[1:])
        return HybridEmbeddingModel(
            models=[
                LiteLLMEmbeddingModel(name=embedding_model_name),
                SparseEmbeddingModel(**kwargs),
            ]
        )
    if embedding == "sparse":
        return SparseEmbeddingModel(**kwargs)

    return LiteLLMEmbeddingModel(name=embedding, embedding_kwargs=kwargs)
