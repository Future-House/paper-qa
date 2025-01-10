import asyncio
import itertools
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Collection, Iterable
from random import Random
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


SAME_CATEGORY_PROMPT_TEMPLATE = (
    "Are the following two groups of texts about the same general topic? Respond with"
    " 'yes' or 'no'.\n\nText group 1:\n{group1}\n\nText group 2:\n{group2}"
)


async def llm_tournament_bin(
    model: str | LLMModel,
    texts: list[str],
    randomize: bool | int | Random | None = True,
    discern_similarity_prompt_template: str = SAME_CATEGORY_PROMPT_TEMPLATE,
) -> set[Collection[str]]:
    llm_model = LLMModel(name=model) if isinstance(model, str) else model
    if isinstance(randomize, bool) and randomize:
        Random(42).shuffle(texts)
    elif isinstance(randomize, int):
        Random(randomize).shuffle(texts)
    elif isinstance(randomize, Random):
        randomize.shuffle(texts)

    async def check_similarity(
        group1: Collection[str], group2: Collection[str]
    ) -> bool:
        result = await llm_model.call_single(
            messages=[
                Message(
                    content=discern_similarity_prompt_template.format(
                        group1=list(group1), group2=list(group2)
                    )
                )
            ]
        )
        if (
            not result.messages
            or len(result.messages) != 1
            or result.messages[0].content is None
        ):
            raise ValueError(f"Failed to extract similarity from LLM result {result}.")
        return result.messages[0].content.strip().lower().startswith("yes")

    async def process_group_pairs(
        g1_g2_ids: tuple[int, int], previously_checked: Collection[tuple[int, int]]
    ) -> tuple[int, int, bool] | None:
        if g1_g2_ids in previously_checked:
            return None
        g1_id, g2_id = g1_g2_ids
        return g1_id, g2_id, await check_similarity(groups[g1_id], groups[g2_id])

    groups: dict[int, set[str]] = {i: {texts[i]} for i in range(len(texts))}
    already_compared: set[tuple[int, int]] = set()
    converged = False  # Do while
    while not converged:
        results_or_none = await asyncio.gather(
            *(
                process_group_pairs(pair, already_compared)
                for pair in itertools.combinations(groups, r=2)
            )
        )
        to_merge: set[tuple[int, int]] = set()
        for g1_id_g2_id_similar_or_None in results_or_none:
            if g1_id_g2_id_similar_or_None is None:
                continue
            g1_id, g2_id, similar = g1_id_g2_id_similar_or_None
            (already_compared if not similar else to_merge).add((g1_id, g2_id))
        converged = not bool(to_merge)
        merged_groups: set[int] = set()
        for g1_id, g2_id in to_merge:
            if g1_id in merged_groups or g2_id in merged_groups:
                continue  # Avoid "race condition" of merging a group twice
            groups[max(groups) + 1] = {*groups.pop(g1_id), *groups.pop(g2_id)}
            merged_groups.update({g1_id, g2_id})

    return {frozenset(v) for v in groups.values()}
