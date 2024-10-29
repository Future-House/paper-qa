from __future__ import annotations

import asyncio
import contextlib
import functools
from abc import ABC, abstractmethod
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
)
from enum import StrEnum
from inspect import isasyncgenfunction, signature
from sys import version_info
from typing import Any, TypeVar, cast

import litellm
import numpy as np
import tiktoken
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from paperqa.prompts import default_system_prompt
from paperqa.rate_limiter import GLOBAL_LIMITER
from paperqa.types import Embeddable, LLMResult
from paperqa.utils import is_coroutine_callable

PromptRunner = Callable[
    [dict, list[Callable[[str], None]] | None, str | None],
    Awaitable[LLMResult],
]

MODEL_COST_MAP = litellm.get_model_cost_map("")


def prepare_args(func: Callable, chunk: str, name: str | None) -> tuple[tuple, dict]:
    with contextlib.suppress(TypeError):
        if "name" in signature(func).parameters:
            return (chunk,), {"name": name}
    return (chunk,), {}


async def do_callbacks(
    async_callbacks: Iterable[Callable[..., Awaitable]],
    sync_callbacks: Iterable[Callable[..., Any]],
    chunk: str,
    name: str | None,
) -> None:
    for f in async_callbacks:
        args, kwargs = prepare_args(f, chunk, name)
        await f(*args, **kwargs)
    for f in sync_callbacks:
        args, kwargs = prepare_args(f, chunk, name)
        f(*args, **kwargs)


class EmbeddingModes(StrEnum):
    DOCUMENT = "document"
    QUERY = "query"


# Estimate from OpenAI's FAQ
# https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
CHARACTERS_PER_TOKEN_ASSUMPTION: float = 4.0
# Added tokens from user/role message
# Need to add while doing rate limits
# Taken from empirical counts in tests
EXTRA_TOKENS_FROM_USER_ROLE: int = 7


class EmbeddingModel(ABC, BaseModel):
    name: str
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional `rate_limit` key, value must be a RateLimitItem or RateLimitItem"
            " string for parsing"
        ),
    )

    async def check_rate_limit(self, token_count: float, **kwargs) -> None:
        if "rate_limit" in self.config:
            await GLOBAL_LIMITER.try_acquire(
                ("client", self.name),
                self.config["rate_limit"],
                weight=max(int(token_count), 1),
                **kwargs,
            )

    def set_mode(self, mode: EmbeddingModes) -> None:
        """Several embedding models have a 'mode' or prompt which affects output."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass


class LiteLLMEmbeddingModel(EmbeddingModel):

    name: str = Field(default="text-embedding-3-small")
    config: dict[str, Any] = Field(
        default_factory=dict,  # See below field_validator for injection of kwargs
        description=(
            "The optional `rate_limit` key's value must be a RateLimitItem or"
            " RateLimitItem string for parsing. The optional `kwargs` key is keyword"
            " arguments to pass to the litellm.aembedding function. Note that LiteLLM's"
            " Router is not used here."
        ),
    )

    @field_validator("config")
    @classmethod
    def set_up_default_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "kwargs" not in value:
            value["kwargs"] = get_litellm_retrying_config(
                timeout=120,  # 2-min timeout seemed reasonable
            )
        return value

    def _truncate_if_large(self, texts: list[str]) -> list[str]:
        """Truncate texts if they are too large by using litellm cost map."""
        if self.name not in MODEL_COST_MAP:
            return texts
        max_tokens = MODEL_COST_MAP[self.name]["max_input_tokens"]
        # heuristic about ratio of tokens to characters
        conservative_char_token_ratio = 3
        maybe_too_large = max_tokens * conservative_char_token_ratio
        if any(len(t) > maybe_too_large for t in texts):
            try:
                enct = tiktoken.encoding_for_model("cl100k_base")
                enc_batch = enct.encode_ordinary_batch(texts)
                return [enct.decode(t[:max_tokens]) for t in enc_batch]
            except KeyError:
                return [t[: max_tokens * conservative_char_token_ratio] for t in texts]

        return texts

    async def embed_documents(
        self, texts: list[str], batch_size: int = 16
    ) -> list[list[float]]:
        texts = self._truncate_if_large(texts)
        N = len(texts)
        embeddings = []
        for i in range(0, N, batch_size):

            await self.check_rate_limit(
                sum(
                    len(t) / CHARACTERS_PER_TOKEN_ASSUMPTION
                    for t in texts[i : i + batch_size]
                )
            )

            response = await litellm.aembedding(
                self.name,
                input=texts[i : i + batch_size],
                **self.config.get("kwargs", {}),
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

    def set_mode(self, mode: EmbeddingModes) -> None:
        # Set mode for all component models
        for model in self.models:
            model.set_mode(mode)


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """An embedding model using SentenceTransformers."""

    name: str = Field(default="multi-qa-MiniLM-L6-cos-v1")
    config: dict[str, Any] = Field(default_factory=dict)
    _model: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Please install paper-qa[local] to use"
                " SentenceTransformerEmbeddingModel."
            ) from exc

        self._model = SentenceTransformer(self.name)

    def set_mode(self, mode: EmbeddingModes) -> None:
        # SentenceTransformer does not support different modes.
        pass

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Asynchronously embed a list of documents using SentenceTransformer.

        Args:
            texts: A list of text documents to embed.

        Returns:
            A list of embedding vectors.
        """
        # Extract additional configurations if needed
        batch_size = self.config.get("batch_size", 32)
        device = self.config.get("device", "cpu")

        # Update the model's device if necessary
        if device:
            self._model.to(device)

        # Run the synchronous encode method in a thread pool to avoid blocking the event loop.
        embeddings = await asyncio.to_thread(
            lambda: self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,  # Disabled progress bar
                batch_size=batch_size,
                device=device,
            ),
        )
        # If embeddings are returned as numpy arrays, convert them to lists.
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        return embeddings


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
        if False:  # type: ignore[unreachable]  # pylint: disable=using-constant-test
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
        if False:  # type: ignore[unreachable]  # pylint: disable=using-constant-test
            yield  # Trick mypy: https://github.com/python/mypy/issues/5070#issuecomment-1050834495

    def infer_llm_type(self) -> str:
        return "completion"

    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # gross approximation

    async def run_prompt(
        self,
        prompt: str,
        data: dict,
        callbacks: list[Callable] | None = None,
        name: str | None = None,
        skip_system: bool = False,
        system_prompt: str = default_system_prompt,
    ) -> LLMResult:
        if self.llm_type is None:
            self.llm_type = self.infer_llm_type()
        if self.llm_type == "chat":
            return await self._run_chat(
                prompt, data, callbacks, name, skip_system, system_prompt
            )
        if self.llm_type == "completion":
            return await self._run_completion(
                prompt, data, callbacks, name, skip_system, system_prompt
            )
        raise ValueError(f"Unknown llm_type {self.llm_type!r}.")

    async def _run_chat(
        self,
        prompt: str,
        data: dict,
        callbacks: list[Callable] | None = None,
        name: str | None = None,
        skip_system: bool = False,
        system_prompt: str = default_system_prompt,
    ) -> LLMResult:
        """Run a chat prompt.

        Args:
            prompt: Prompt to use.
            data: Keys for the input variables that will be formatted into prompt.
            callbacks: Optional functions to call with each chunk of the completion.
            name: Optional name for the result.
            skip_system: Set True to skip the system prompt.
            system_prompt: System prompt to use.

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
            completion = await self.achat_iter(messages)  # type: ignore[misc]
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
        callbacks: Iterable[Callable] | None = None,
        name: str | None = None,
        skip_system: bool = False,
        system_prompt: str = default_system_prompt,
    ) -> LLMResult:
        """Run a completion prompt.

        Args:
            prompt: Prompt to use.
            data: Keys for the input variables that will be formatted into prompt.
            callbacks: Optional functions to call with each chunk of the completion.
            name: Optional name for the result.
            skip_system: Set True to skip the system prompt.
            system_prompt: System prompt to use.

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


LLMModelOrChild = TypeVar("LLMModelOrChild", bound=LLMModel)


def rate_limited(
    func: Callable[[LLMModelOrChild, Any], Awaitable[Chunk] | AsyncIterable[Chunk]],
) -> Callable[
    [LLMModelOrChild, Any, Any],
    Awaitable[Chunk | AsyncIterator[Chunk] | AsyncIterator[LLMModelOrChild]],
]:
    """Decorator to rate limit relevant methods of an LLMModel."""

    @functools.wraps(func)
    async def wrapper(
        self: LLMModelOrChild, *args: Any, **kwargs: Any
    ) -> Chunk | AsyncIterator[Chunk] | AsyncIterator[LLMModelOrChild]:

        if not hasattr(self, "check_rate_limit"):
            raise NotImplementedError(
                f"Model {self.name} must have a `check_rate_limit` method."
            )

        # Estimate token count based on input
        if func.__name__ in {"acomplete", "acomplete_iter"}:
            prompt = args[0] if args else kwargs.get("prompt", "")
            token_count = (
                len(prompt) / CHARACTERS_PER_TOKEN_ASSUMPTION
                + EXTRA_TOKENS_FROM_USER_ROLE
            )
        elif func.__name__ in {"achat", "achat_iter"}:
            messages = args[0] if args else kwargs.get("messages", [])
            token_count = len(str(messages)) / CHARACTERS_PER_TOKEN_ASSUMPTION
        else:
            token_count = 0  # Default if method is unknown

        await self.check_rate_limit(token_count)

        # If wrapping a generator, count the tokens for each
        # portion before yielding
        if isasyncgenfunction(func):

            async def rate_limited_generator() -> AsyncGenerator[LLMModelOrChild, None]:
                async for item in func(self, *args, **kwargs):
                    token_count = 0
                    if isinstance(item, Chunk):
                        token_count = int(
                            len(item.text or "") / CHARACTERS_PER_TOKEN_ASSUMPTION
                        )
                    await self.check_rate_limit(token_count)
                    yield item

            return rate_limited_generator()

        result = await func(self, *args, **kwargs)  # type: ignore[misc]

        if func.__name__ in {"acomplete", "achat"} and isinstance(result, Chunk):
            await self.check_rate_limit(result.completion_tokens)
        return result

    return wrapper


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


IS_PYTHON_BELOW_312 = version_info < (3, 12)
if not IS_PYTHON_BELOW_312:
    _DeploymentTypedDictValidator = TypeAdapter(
        list[litellm.DeploymentTypedDict],
        config=ConfigDict(arbitrary_types_allowed=True),
    )


def get_litellm_retrying_config(timeout: float = 60.0) -> dict[str, Any]:
    """Get retrying configuration for litellm.acompletion and litellm.aembedding."""
    return {"num_retries": 3, "timeout": timeout}


class PassThroughRouter(litellm.Router):
    """Router that is just a wrapper on LiteLLM's normal free functions."""

    def __init__(self, **kwargs):
        self._default_kwargs = kwargs

    async def atext_completion(self, *args, **kwargs):
        return await litellm.atext_completion(*args, **(self._default_kwargs | kwargs))

    async def acompletion(self, *args, **kwargs):
        return await litellm.acompletion(*args, **(self._default_kwargs | kwargs))


class LiteLLMModel(LLMModel):
    """A wrapper around the litellm library."""

    config: dict = Field(
        default_factory=dict,
        description=(
            "Configuration of this model containing several important keys. The"
            " optional `model_list` key stores a list of all model configurations"
            " (SEE: https://docs.litellm.ai/docs/routing). The optional"
            " `router_kwargs` key is keyword arguments to pass to the Router class."
            " Inclusion of a key `pass_through_router` with a truthy value will lead"
            " to using not using LiteLLM's Router, instead just LiteLLM's free"
            f" functions (see {PassThroughRouter.__name__}). Rate limiting applies"
            " regardless of `pass_through_router` being present. The optional"
            " `rate_limit` key is a dictionary keyed by model group name with values"
            " of type limits.RateLimitItem (in tokens / minute) or valid"
            " limits.RateLimitItem string for parsing."
        ),
    )
    name: str = "gpt-4o-mini"
    _router: litellm.Router | None = None

    @model_validator(mode="before")
    @classmethod
    def maybe_set_config_attribute(cls, data: dict[str, Any]) -> dict[str, Any]:
        """If a user only gives a name, make a sensible config dict for them."""
        if "config" not in data:
            data["config"] = {}
        if "name" in data and "model_list" not in data["config"]:
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
            } | data["config"]

        if "router_kwargs" not in data["config"]:
            data["config"]["router_kwargs"] = {}
        data["config"]["router_kwargs"] = (
            get_litellm_retrying_config() | data["config"]["router_kwargs"]
        )
        if not data["config"].get("pass_through_router"):
            data["config"]["router_kwargs"] = {"retry_after": 5} | data["config"][
                "router_kwargs"
            ]

        # we only support one "model name" for now, here we validate
        model_list = data["config"]["model_list"]
        if IS_PYTHON_BELOW_312:
            if not isinstance(model_list, list):
                # Work around https://github.com/BerriAI/litellm/issues/5664
                raise TypeError(f"model_list must be a list, not a {type(model_list)}.")
        else:
            # pylint: disable-next=possibly-used-before-assignment
            _DeploymentTypedDictValidator.validate_python(model_list)
        if len({m["model_name"] for m in model_list}) > 1:
            raise ValueError("Only one model name per model list is supported for now.")
        return data

    def __getstate__(self):
        # Prevent _router from being pickled, SEE: https://stackoverflow.com/a/2345953
        state = super().__getstate__()
        state["__dict__"] = state["__dict__"].copy()
        state["__dict__"].pop("_router", None)
        return state

    @property
    def router(self) -> litellm.Router:
        if self._router is None:
            router_kwargs: dict = self.config.get("router_kwargs", {})
            if self.config.get("pass_through_router"):
                self._router = PassThroughRouter(**router_kwargs)
            else:
                self._router = litellm.Router(
                    model_list=self.config["model_list"], **router_kwargs
                )
        return self._router

    async def check_rate_limit(self, token_count: float, **kwargs) -> None:
        if "rate_limit" in self.config:
            await GLOBAL_LIMITER.try_acquire(
                ("client", self.name),
                self.config["rate_limit"].get(self.name, None),
                weight=max(int(token_count), 1),
                **kwargs,
            )

    @rate_limited
    async def acomplete(self, prompt: str) -> Chunk:  # type: ignore[override]
        response = await self.router.atext_completion(model=self.name, prompt=prompt)
        return Chunk(
            text=response.choices[0].text,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    @rate_limited
    async def acomplete_iter(  # type: ignore[override]
        self, prompt: str
    ) -> AsyncIterable[Chunk]:
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

    @rate_limited
    async def achat(  # type: ignore[override]
        self, messages: Iterable[dict[str, str]]
    ) -> Chunk:
        response = await self.router.acompletion(self.name, list(messages))
        return Chunk(
            text=cast(litellm.Choices, response.choices[0]).message.content,
            prompt_tokens=response.usage.prompt_tokens,  # type: ignore[attr-defined]
            completion_tokens=response.usage.completion_tokens,  # type: ignore[attr-defined]
        )

    @rate_limited
    async def achat_iter(  # type: ignore[override]
        self, messages: Iterable[dict[str, str]]
    ) -> AsyncIterable[Chunk]:
        completion = await self.router.acompletion(
            self.name,
            list(messages),
            stream=True,
            stream_options={"include_usage": True},
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
        return litellm.token_counter(model=self.name, text=text)


def cosine_similarity(a, b):
    norm_product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return a @ b.T / norm_product


class VectorStore(BaseModel, ABC):
    """Interface for vector store - very similar to LangChain's VectorStore to be compatible."""

    model_config = ConfigDict(extra="forbid")

    # can be tuned for different tasks
    mmr_lambda: float = Field(default=0.9)
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

    def add_texts_and_embeddings(self, texts: Iterable[Embeddable]) -> None:
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
