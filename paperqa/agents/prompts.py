from __future__ import annotations

from paperqa import PromptCollection

# I wanted to to make this an Enum
# but there is a metaclass conflict
# so we must instead have some logic here
# and some logic on named_prompt in the QueryRequest model
# https://github.com/pydantic/pydantic/issues/2173
STATIC_PROMPTS: dict[str, PromptCollection] = {
    "default": PromptCollection(
        qa=(
            "Answer the question below with the context.\n\n"
            "Context:\n\n{context}\n\n----\n\n"
            "Question: {question}\n\n"
            "Write an answer based on the context. "
            "If the context provides insufficient information and "
            "the question cannot be directly answered, reply "
            '"I cannot answer." '
            "For each part of your answer, indicate which sources most support "
            "it via citation keys at the end of sentences, "
            "like (Example2012Example pages 3-4). Only cite from the context "
            "below and only use the valid keys. "
            "Write in the style of a direct email containing only key details, equations, and quantities. "
            'Avoid using adverb phrases like "furthermore", "additionally", and "moreover." '
            "This will go directly onto a website for public viewing, so do not include any "
            "process details about following these instructions.\n\n"
            "Answer ({answer_length}):\n"
        ),
        select=(
            "Select papers that may help answer the question below. "
            "Papers are listed as $KEY: $PAPER_INFO. "
            "Return a list of keys, separated by commas. "
            'Return "None", if no papers are applicable. '
            "Choose papers that are relevant, from reputable sources, and timely "
            "(if the question requires timely information). \n\n"
            "Question: {question}\n\n"
            "Papers: {papers}\n\n"
            "Selected keys:"
        ),
        pre=(
            "We are collecting background information for the question/task below. "
            "Provide a brief summary of definitions, acronyms, or background information (about 50 words) that "
            "could help answer the question. Do not answer it directly. Ignore formatting instructions. "
            "Do not answer if there is nothing to contribute. "
            "\n\nQuestion:\n{question}\n\n"
        ),
        post=None,
        system=(
            "Answer in a direct and concise tone. "
            "Your audience is an expert, so be highly specific. "
            "If there are ambiguous terms or acronyms, be explicit."
        ),
        skip_summary=False,
        json_summary=True,
        summary_json=(
            "Excerpt from {citation}\n\n----\n\n{text}\n\n----\n\n"
            "Query: {question}\n\n"
        ),
        summary_json_system="Provide a summary of the excerpt that could help answer the question based on the excerpt.  "  # noqa: E501
        "The excerpt may be irrelevant. Do not directly answer the question - only summarize relevant information. "
        "Respond with the following JSON format:\n\n"
        '{{\n"summary": "...",\n"relevance_score": "..."}}\n\n'
        "where `summary` is relevant information from text ({summary_length}), "
        "and `relevance_score` is "
        "the relevance of `summary` to answer the question (integer out of 10).",
    ),
    "wikicrow": PromptCollection(
        qa=(
            "Answer the question below with the context.\n\n"
            "Context:\n\n{context}\n\n----\n\n"
            "Question: {question}\n\n"
            "Write an answer based on the context. "
            "If the context provides insufficient information and "
            "the question cannot be directly answered, reply "
            '"I cannot answer." '
            "For each part of your answer, indicate which sources most support "
            "it via citation keys at the end of sentences, "
            "like (Example2012Example pages 3-4). Only cite from the context "
            "below and only use the valid keys. Write in the style of a "
            "Wikipedia article, with concise sentences and coherent paragraphs. "
            "The context comes from a variety of sources and is only a summary, "
            "so there may inaccuracies or ambiguities. Make sure the gene_names exactly match "
            "the gene name in the question before using a context. "
            "This answer will go directly onto "
            "Wikipedia, so do not add any extraneous information.\n\n"
            "Answer ({answer_length}):"
        ),
        select=(
            "Select papers that may help answer the question below. "
            "Papers are listed as $KEY: $PAPER_INFO. "
            "Return a list of keys, separated by commas. "
            'Return "None", if no papers are applicable. '
            "Choose papers that are relevant, from reputable sources, and timely "
            "(if the question requires timely information). \n\n"
            "Question: {question}\n\n"
            "Papers: {papers}\n\n"
            "Selected keys:"
        ),
        pre=(
            "We are collecting background information for the question/task below. "
            "Provide a brief summary of definitions, acronyms, or background information (about 50 words) that "
            "could help answer the question. Do not answer it directly. Ignore formatting instructions. "
            "Do not answer if there is nothing to contribute. "
            "\n\nQuestion:\n{question}\n\n"
        ),
        post=None,
        system=(
            "Answer in a direct and concise tone. "
            "Your audience is an expert, so be highly specific. "
            "If there are ambiguous terms or acronyms, be explicit."
        ),
        skip_summary=False,
        json_summary=True,
        summary_json=(
            "Excerpt from {citation}\n\n----\n\n{text}\n\n----\n\n"
            "Query: {question}\n\n"
        ),
        summary_json_system="Provide a summary of the excerpt that could help answer the question based on the excerpt.  "  # noqa: E501
        "The excerpt may be irrelevant. Do not directly answer the question - only summarize relevant information. "
        "Respond with the following JSON format:\n\n"
        '{{\n"summary": "...",\n"gene_name: "...",\n"relevance_score": "..."}}\n\n'
        "where `summary` is relevant information from text ({summary_length}), "
        "`gene_name` is the gene discussed in the excerpt (may be different than query), "
        "and `relevance_score` is "
        "the relevance of `summary` to answer the question (integer out of 10).",
    ),
}

# for backwards compatibility
STATIC_PROMPTS["json"] = STATIC_PROMPTS["default"]
