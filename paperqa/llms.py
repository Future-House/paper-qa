import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, cast, get_args, get_type_hints

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, model_validator

from .prompts import default_system_prompt


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
    llm_type: str = "completion"

    @abstractmethod
    async def acomplete(self, client: Any, prompt: str) -> str:
        pass

    @abstractmethod
    async def acomplete_iter(self, client: Any, prompt: str) -> Any:
        """Return an async generator that yields chunks of the completion.

        I cannot get mypy to understand the override, so marked as Any"""
        pass

    @abstractmethod
    async def achat(self, client: Any, messages: list[dict[str, str]]) -> str:
        pass

    @abstractmethod
    async def achat_iter(self, client: Any, messages: list[dict[str, str]]) -> Any:
        """Return an async generator that yields chunks of the completion.

        I cannot get mypy to understand the override, so marked as Any"""
        pass

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
        if client is None:
            raise ValueError(
                "Your client is None - did you forget to set it after pickling?"
            )
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

    @model_validator(mode="after")
    @classmethod
    def guess_llm_type(cls, data: Any) -> Any:
        m = cast(OpenAILLMModel, data)
        m.llm_type = guess_model_type(m.config["model"])
        return m

    async def acomplete(self, client: Any, prompt: str) -> str:
        completion = await client.completions.create(
            prompt=prompt, **process_llm_config(self.config)
        )
        return completion.choices[0].text

    async def acomplete_iter(self, client: Any, prompt: str) -> Any:
        completion = await client.completions.create(
            prompt=prompt, **process_llm_config(self.config), stream=True
        )
        async for chunk in completion:
            yield chunk.choices[0].text

    async def achat(self, client: Any, messages: list[dict[str, str]]) -> str:
        completion = await client.chat.completions.create(
            messages=messages, **process_llm_config(self.config)
        )
        return completion.choices[0].message.content

    async def achat_iter(self, client: Any, messages: list[dict[str, str]]) -> Any:
        completion = await client.chat.completions.create(
            messages=messages, **process_llm_config(self.config), stream=True
        )
        async for chunk in completion:
            yield chunk.choices[0].delta.content


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
