import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Collection, Iterable
from typing import TypeAlias

from aviary.core import Message
from llmclient import MultipleCompletionLLMModel as LLMModel
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

BinningFunction: TypeAlias = Callable[
    [str | LLMModel, Collection[str]], Awaitable[set[Collection[str]]]
]


class Categories(BaseModel):
    """Simple holder for categories."""

    categories: set[str] = Field(description="Phrases describing unique categories.")


CATEGORY_DISCERN_PROMPT_TEMPLATE = (
    "Analyze the below responses and suggest unique categories they can fall into."
    " If categories are not obvious, do not be afraid to propose zero categories."
    " Don't restate the responses, just provide the"
    " categories.\n\nResponses:\n{responses}"
)


async def discern_categories(
    model: str | LLMModel,
    texts: Iterable[str],
    category_discern_prompt_template: str = CATEGORY_DISCERN_PROMPT_TEMPLATE,
) -> set[str]:
    """Bin the free-text answers into groups using the input LLM model."""
    llm_model = LLMModel(name=model) if isinstance(model, str) else model
    result = await llm_model.call_single(
        messages=[
            Message(
                content=category_discern_prompt_template.format(
                    responses="\n".join(f"- {t}" for t in texts)
                )
            )
        ],
        output_type=Categories,
    )
    if (
        not result.messages
        or len(result.messages) != 1
        or result.messages[0].content is None
    ):
        raise ValueError(f"Failed to extract categories from LLM result {result}.")
    return Categories.model_validate_json(result.messages[0].content).categories


async def llm_categorize_then_bin(
    model: str | LLMModel, texts: Collection[str]
) -> set[Collection[str]]:
    categories = await discern_categories(model, texts)
    binning_function = make_binning_function(model, categories)
    assigned_categories = await asyncio.gather(*(binning_function(t) for t in texts))
    binned_categories = defaultdict(set)
    for text, category in zip(texts, assigned_categories, strict=True):
        binned_categories[category].add(text)
    return {frozenset(v) for v in binned_categories.values()}


CATEGORY_ASSIGNMENT_PROMPT_TEMPLATE = (
    "Which of the categories in {categories} best fits the text? Respond with just"
    " one category name.\n\n{{text}}\n\nCategory:"
)
FAILED_CATEGORIZATION = ""


def make_binning_function(
    model: str | LLMModel,
    categories: Collection[str],
    category_assignment_prompt_template: str = CATEGORY_ASSIGNMENT_PROMPT_TEMPLATE,
    failed_categorization: str = FAILED_CATEGORIZATION,
) -> Callable[[str], Awaitable[str]]:
    """Make a function to bin text into a category using the input LLM model."""
    llm_model = LLMModel(name=model) if isinstance(model, str) else model
    # Use list to avoid collision with prompt template {} syntax
    pt = category_assignment_prompt_template.format(categories=list(categories))

    async def discern_category(text: str) -> str:
        result = await llm_model.call_single(
            messages=[Message(content=pt.format(text=text))]
        )
        if (
            not result.messages
            or len(result.messages) != 1
            or result.messages[0].content is None
        ):
            raise ValueError(f"Failed to extract category from LLM result {result}.")
        category = result.messages[0].content.strip()
        if category not in categories:
            logger.warning(
                f"Failed to categorize {category} into known categories {categories}."
            )
            return failed_categorization
        return category

    return discern_category
