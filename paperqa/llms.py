import re
from typing import Any, Awaitable, Callable, get_args, get_type_hints

from openai import AsyncOpenAI

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


def make_chain(
    client: AsyncOpenAI,
    prompt: str,
    llm_config: dict,
    skip_system: bool = False,
    system_prompt: str = default_system_prompt,
) -> Awaitable[Any]:
    """Create a function to execute a batch of prompts

    Args:
        client: OpenAI client
        prompt: The prompt to use
        llm_config: The config to use
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
    if llm_config["model_type"] == "chat":
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
                completion = await client.chat.completions.create(
                    messages=messages, **process_llm_config(llm_config)
                )
                output = completion.choices[0].message.content
            else:
                completion = await client.chat.completions.create(
                    messages=messages, **process_llm_config(llm_config), stream=True
                )
                result = []
                async for chunk in completion:
                    c = chunk.choices[0].delta.content
                    if c:
                        result.append(c)
                        [f(c) for f in callbacks]
                output = "".join(result)
            return output

        return execute
    elif llm_config["model_type"] == "completion":
        if skip_system:
            completion_prompt = prompt
        else:
            completion_prompt = system_prompt + "\n\n" + prompt

        async def execute(
            data: dict, callbacks: list[Callable[[str], None]] | None = None
        ) -> str:
            if callbacks is None:
                completion = await client.completions.create(
                    prompt=completion_prompt.format(**data),
                    **process_llm_config(llm_config),
                )
                output = completion.choices[0].text
            else:
                completion = await client.completions.create(
                    prompt=completion_prompt.format(**data),
                    **process_llm_config(llm_config),
                    stream=True,
                )
                result = []
                async for chunk in completion:
                    c = chunk.choices[0].text
                    if c:
                        result.append(c)
                        [f(c) for f in callbacks]
                output = "".join(result)
            return output

        return execute
    else:
        raise NotImplementedError(f"Unknown model type {llm_config['model_type']}")


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
