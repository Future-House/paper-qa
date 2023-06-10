import re
from typing import Any, Dict, List, Optional, cast

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import LLMResult, SystemMessage

from .types import CBManager


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


def make_chain(
    prompt: StringPromptTemplate, llm: BaseLanguageModel, skip_system: bool = False
) -> FallbackLLMChain:
    if type(llm) == ChatOpenAI:
        system_message_prompt = SystemMessage(
            content="Answer in an unbiased, concise, scholarly tone. "
            "You may refuse to answer if there is insufficient information. "
            "If there are ambiguous terms or acronyms, first define them. ",
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=prompt)
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
    if score:
        return int(score.group(1))
    if len(text) < 100:
        return 1
    return 5
