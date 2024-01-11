import re
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Coroutine,
    Sequence,
    cast,
    get_args,
    get_type_hints,
)

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .prompts import default_system_prompt
from .types import Embeddable
from .utils import batch_iter, flatten, gather_with_concurrency


def guess_model_type(model_name: str) -> str:
    import openai

    model_type = get_type_hints(
        openai.types.chat.completion_create_params.CompletionCreateParamsBase
    )["model"]
    model_union = get_args(get_args(model_type)[1])
    model_arr = list(model_union)
    if model_name in model_arr:
        return "chat"
    return "completion"


def is_openai_model(model_name):
    import openai

    model_type = get_type_hints(
        openai.types.chat.completion_create_params.CompletionCreateParamsBase
    )["model"]
    model_union = get_args(get_args(model_type)[1])
    model_arr = list(model_union)

    complete_model_types = get_type_hints(
        openai.types.completion_create_params.CompletionCreateParamsBase
    )["model"]
    complete_model_union = get_args(get_args(complete_model_types)[1])
    complete_model_arr = list(complete_model_union)

    return model_name in model_arr or model_name in complete_model_arr


def process_llm_config(llm_config: dict) -> dict:
    """Remove model_type and try to set max_tokens"""
    result = {k: v for k, v in llm_config.items() if k != "model_type"}
    if "max_tokens" not in result or result["max_tokens"] == -1:
        model = llm_config["model"]
        # now we guess!
        if model.startswith("gpt-4") or (
            model.startswith("gpt-3.5") and "1106" in model
        ):
            result["max_tokens"] = 4096
        else:
            result["max_tokens"] = 2048  # ?
    return result


async def embed_documents(
    client: AsyncOpenAI, texts: list[str], embedding_model: str
) -> list[list[float]]:
    """Embed a list of documents with batching"""
    if client is None:
        raise ValueError(
            "Your client is None - did you forget to set it after pickling?"
        )
    response = await client.embeddings.create(
        model=embedding_model, input=texts, encoding_format="float"
    )
    return [e.embedding for e in response.data]


class EmbeddingModel(ABC, BaseModel):
    @abstractmethod
    async def embed_documents(self, client: Any, texts: list[str]) -> list[list[float]]:
        pass


class OpenAIEmbeddingModel(EmbeddingModel):
    embedding_model: str = Field(default="text-embedding-ada-002")

    async def embed_documents(self, client: Any, texts: list[str]) -> list[list[float]]:
        return await embed_documents(
            cast(AsyncOpenAI, client), texts, self.embedding_model
        )


class LLMModel(ABC, BaseModel):
    llm_type: str | None = None
    model_config = ConfigDict(extra="forbid")

    async def acomplete(self, client: Any, prompt: str) -> str:
        raise NotImplementedError

    async def acomplete_iter(self, client: Any, prompt: str) -> Any:
        """Return an async generator that yields chunks of the completion.

        I cannot get mypy to understand the override, so marked as Any"""
        raise NotImplementedError

    async def achat(self, client: Any, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    async def achat_iter(self, client: Any, messages: list[dict[str, str]]) -> Any:
        """Return an async generator that yields chunks of the completion.

        I cannot get mypy to understand the override, so marked as Any"""
        raise NotImplementedError

    def infer_llm_type(self, client: Any) -> str:
        return "completion"

    def make_chain(
        self,
        client: Any,
        prompt: str,
        skip_system: bool = False,
        system_prompt: str = default_system_prompt,
    ) -> Callable[[dict, list[Callable[[str], None]] | None], Coroutine[Any, Any, str]]:
        """Create a function to execute a batch of prompts

        Args:
            client: a ephemeral client to use
            prompt: The prompt to use
            skip_system: Whether to skip the system prompt
            system_prompt: The system prompt to use

        Returns:
            A function to execute a prompt. Its signature is:
            execute(data: dict, callbacks: list[Callable[[str], None]]] | None = None) -> str
            where data is a dict with keys for the input variables that will be formatted into prompt
            and callbacks is a list of functions to call with each chunk of the completion.
        """
        # check if it needs to be set
        if self.llm_type is None:
            self.llm_type = self.infer_llm_type(client)
        if self.llm_type == "chat":
            system_message_prompt = dict(role="system", content=system_prompt)
            human_message_prompt = dict(role="user", content=prompt)
            if skip_system:
                chat_prompt = [human_message_prompt]
            else:
                chat_prompt = [system_message_prompt, human_message_prompt]

            async def execute(
                data: dict, callbacks: list[Callable[[str], None]] | None = None
            ) -> str:
                messages = chat_prompt[:-1] + [
                    dict(role="user", content=chat_prompt[-1]["content"].format(**data))
                ]
                if callbacks is None:
                    output = await self.achat(client, messages)
                else:
                    completion = self.achat_iter(client, messages)  # type: ignore
                    result = []
                    async for chunk in completion:  # type: ignore
                        if chunk:
                            result.append(chunk)
                            [f(chunk) for f in callbacks]
                    output = "".join(result)
                return output

            return execute
        elif self.llm_type == "completion":
            if skip_system:
                completion_prompt = prompt
            else:
                completion_prompt = system_prompt + "\n\n" + prompt

            async def execute(
                data: dict, callbacks: list[Callable[[str], None]] | None = None
            ) -> str:
                if callbacks is None:
                    output = await self.acomplete(
                        client, completion_prompt.format(**data)
                    )
                else:
                    completion = self.acomplete_iter(  # type: ignore
                        client,
                        completion_prompt.format(**data),
                    )
                    result = []
                    async for chunk in completion:  # type: ignore
                        if chunk:
                            result.append(chunk)
                            [f(chunk) for f in callbacks]
                    output = "".join(result)
                return output

            return execute
        raise ValueError(f"Unknown llm_type: {self.llm_type}")


class OpenAILLMModel(LLMModel):
    config: dict = Field(default=dict(model="gpt-3.5-turbo", temperature=0.1))

    def _check_client(self, client: Any) -> AsyncOpenAI:
        if client is None:
            raise ValueError(
                "Your client is None - did you forget to set it after pickling?"
            )
        if not isinstance(client, AsyncOpenAI):
            raise ValueError(
                f"Your client is not a required AsyncOpenAI client. It is a {type(client)}"
            )
        return cast(AsyncOpenAI, client)

    @model_validator(mode="after")
    @classmethod
    def guess_llm_type(cls, data: Any) -> Any:
        m = cast(OpenAILLMModel, data)
        m.llm_type = guess_model_type(m.config["model"])
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
            messages=messages, **process_llm_config(self.config)  # type: ignore
        )
        return completion.choices[0].message.content or ""

    async def achat_iter(self, client: Any, messages: list[dict[str, str]]) -> Any:
        aclient = self._check_client(client)
        completion = await aclient.chat.completions.create(
            messages=messages, **process_llm_config(self.config), stream=True  # type: ignore
        )
        async for chunk in cast(AsyncGenerator, completion):
            yield chunk.choices[0].delta.content


class LangchainLLMModel(LLMModel):
    """A wrapper around the wrapper langchain"""

    def infer_llm_type(self, client: Any) -> str:
        from langchain_core.language_models.chat_models import BaseChatModel

        if isinstance(client, BaseChatModel):
            return "chat"
        return "completion"

    async def acomplete(self, client: Any, prompt: str) -> str:
        return await client.ainvoke(prompt)

    async def acomplete_iter(self, client: Any, prompt: str) -> Any:
        async for chunk in cast(AsyncGenerator, client.astream(prompt)):
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
        return (await client.ainvoke(lc_messages)).content

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
        async for chunk in client.astream(lc_messages):
            yield chunk.content


class LangchainEmbeddingModel(EmbeddingModel):
    """A wrapper around the wrapper langchain"""

    async def embed_documents(self, client: Any, texts: list[str]) -> list[list[float]]:
        return await client.aembed_documents(texts)


class LlamaEmbeddingModel(EmbeddingModel):
    embedding_model: str = Field(default="llama")

    batch_size: int = Field(default=4)
    concurrency: int = Field(default=1)

    async def embed_documents(self, client: Any, texts: list[str]) -> list[list[float]]:
        cast(AsyncOpenAI, client)

        async def process(texts: list[str]) -> list[float]:
            for i in range(3):
                # access httpx client directly to avoid type casting
                response = await client._client.post(
                    client.base_url.join("../embedding"), json={"content": texts}
                )
                body = response.json()
                if len(texts) == 1:
                    if type(body) != dict or body.get("embedding") is None:
                        continue
                    return [body["embedding"]]
                else:
                    if type(body) != list or body[0] != "results":
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
    embedding_model: str = Field(default="multi-qa-MiniLM-L6-cos-v1")
    _model: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers to use this model")

        self._model = SentenceTransformer(self.embedding_model)

    async def embed_documents(self, client: Any, texts: list[str]) -> list[list[float]]:
        from sentence_transformers import SentenceTransformer

        embeddings = cast(SentenceTransformer, self._model).encode(texts)
        return embeddings


def cosine_similarity(a, b):
    dot_product = np.dot(a, b.T)
    norm_product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return dot_product / norm_product


class VectorStore(BaseModel, ABC):
    """Interface for vector store - very similar to LangChain's VectorStore to be compatible"""

    embedding_model: EmbeddingModel = Field(default=OpenAIEmbeddingModel())
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

    async def max_marginal_relevance_search(
        self, client: Any, query: str, k: int, fetch_k: int, lambda_: float = 0.5
    ) -> tuple[Sequence[Embeddable], list[float]]:
        """Vectorized implementation of Maximal Marginal Relevance (MMR) search.

        Args:
            query: Query vector.
            k: Number of results to return.
            lambda_: Weighting of relevance and diversity.

        Returns:
            List of tuples (doc, score) of length k.
        """
        if fetch_k < k:
            raise ValueError("fetch_k must be greater or equal to k")

        texts, scores = await self.similarity_search(client, query, fetch_k)
        if len(texts) <= k:
            return texts, scores

        embeddings = np.array([t.embedding for t in texts])
        np_scores = np.array(scores)
        similarity_matrix = cosine_similarity(embeddings, embeddings)

        selected_indices = [0]
        remaining_indices = list(range(1, len(texts)))

        while len(selected_indices) < k:
            selected_similarities = similarity_matrix[:, selected_indices]
            max_sim_to_selected = selected_similarities.max(axis=1)

            mmr_scores = lambda_ * np_scores - (1 - lambda_) * max_sim_to_selected
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
        if len(self.texts) == 0:
            return [], []
        np_query = np.array(
            (await self.embedding_model.embed_documents(client, [query]))[0]
        )
        similarity_scores = cosine_similarity(
            np_query.reshape(1, -1), self._embeddings_matrix
        )[0]
        similarity_scores = np.nan_to_num(similarity_scores, nan=-np.inf)
        sorted_indices = np.argsort(similarity_scores)[::-1]
        return (
            [self.texts[i] for i in sorted_indices[:k]],
            [similarity_scores[i] for i in sorted_indices[:k]],
        )


class LangchainVectorStore(VectorStore):
    """A wrapper around the wrapper langchain"""

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
        if s > 10:
            s = int(s / 10)  # sometimes becomes out of 100
        return s
    last_few = text[-15:]
    scores = re.findall(r"([0-9]+)", last_few)
    if scores:
        s = int(scores[-1])
        if s > 10:
            s = int(s / 10)  # sometimes becomes out of 100
        return s
    if len(text) < 100:
        return 1
    return 5
