import re
from typing import Any, Dict, List, Optional, cast

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import BasePromptTemplate, PromptTemplate, StringPromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import LLMResult, SystemMessage

from .types import CBManager

memory_prompt = PromptTemplate(
    input_variables=["memory", "start"],
    template="Previous answers that may be helpful:\n\n{memory}\n\n"
    "----------------------------------------\n\n"
    "{start}",
)


class FallbackLLMChain(LLMChain):
    """Chain that falls back to synchronous generation if the async generation fails."""

    async def agenerate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CBManager] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        try:
            run_manager = cast(AsyncCallbackManagerForChainRun, run_manager)
            return await super().agenerate(input_list, run_manager=run_manager)
        except NotImplementedError:
            run_manager = cast(CallbackManagerForChainRun, run_manager)
            return self.generate(input_list)


# TODO: If upstream is fixed remove this


class ExtendedHumanMessagePromptTemplate(HumanMessagePromptTemplate):
    prompt: BasePromptTemplate


def make_chain(
    prompt: StringPromptTemplate,
    llm: BaseLanguageModel,
    skip_system: bool = False,
    memory: Optional[BaseChatMemory] = None,
) -> FallbackLLMChain:
    if memory and len(memory.load_memory_variables({})["memory"]) > 0:
        # we copy the prompt so we don't modify the original
        # TODO: Figure out pipeline prompts to avoid this
        # the problem with pipeline prompts is that
        # the memory is a constant (or partial), not  a prompt
        # and I cannot seem to make an empty prompt (or str)
        # work as an input to pipeline prompt
        assert isinstance(
            prompt, PromptTemplate
        ), "Memory only works with prompt templates - see comment above"
        assert "memory" in memory.load_memory_variables({})
        new_prompt = PromptTemplate(
            input_variables=prompt.input_variables,
            template=memory_prompt.format(
                start=prompt.template, **memory.load_memory_variables({})
            ),
        )
        prompt = new_prompt
    if type(llm) == ChatOpenAI:
        system_message_prompt = SystemMessage(
            content="Answer in an unbiased, concise, scholarly tone. "
            "You may refuse to answer if there is insufficient information. "
            "If there are ambiguous terms or acronyms, first define them. ",
        )
        human_message_prompt = ExtendedHumanMessagePromptTemplate(prompt=prompt)
        if skip_system:
            chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
        else:
            chat_prompt = ChatPromptTemplate.from_messages(
                [system_message_prompt, human_message_prompt]
            )
        return FallbackLLMChain(prompt=chat_prompt, llm=llm)
    return FallbackLLMChain(prompt=prompt, llm=llm)


def get_score(text: str) -> int:
    score = re.search(r"[sS]core[:is\s]+([0-9]+)", text)
    if not score:
        score = re.search(r"\(([0-9])\w*\/", text)
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
