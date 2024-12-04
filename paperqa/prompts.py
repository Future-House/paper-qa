summary_prompt = (
    "Summarize the excerpt below to help answer a question.\n\nExcerpt from"
    " {citation}\n\n----\n\n{text}\n\n----\n\nQuestion: {question}\n\nDo not directly"
    " answer the question, instead summarize to give evidence to help answer the"
    " question. Stay detailed; report specific numbers, equations, or direct quotes"
    ' (marked with quotation marks). Reply "Not applicable" if the excerpt is'
    " irrelevant. At the end of your response, provide an integer score from 1-10 on a"
    " newline indicating relevance to question. Do not explain your score.\n\nRelevant"
    " Information Summary ({summary_length}):"
)

summary_json_prompt = (
    "Excerpt from {citation}\n\n----\n\n{text}\n\n----\n\nQuestion: {question}\n\n"
)

# The below "cannot answer" sentinel phrase should:
# 1. Lead to complete tool being called with has_successful_answer=False
# 2. Can be used for unit testing
CANNOT_ANSWER_PHRASE = "I cannot answer"
qa_prompt = (
    "Answer the question below with the context.\n\n"
    "Context (with relevance scores):\n\n{context}\n\n----\n\n"
    "Question: {question}\n\n"
    "Write an answer based on the context. "
    "If the context provides insufficient information reply "
    f'"{CANNOT_ANSWER_PHRASE}." '
    "For each part of your answer, indicate which sources most support "
    "it via citation keys at the end of sentences, "
    "like {example_citation}. Only cite from the context "
    "below and only use the valid keys. Write in the style of a "
    "Wikipedia article, with concise sentences and coherent paragraphs. "
    "The context comes from a variety of sources and is only a summary, "
    "so there may inaccuracies or ambiguities. If quotes are present and "
    "relevant, use them in the answer. This answer will go directly onto "
    "Wikipedia, so do not add any extraneous information.\n\n"
    "Answer ({answer_length}):"
)

select_paper_prompt = (
    "Select papers that may help answer the question below. "
    "Papers are listed as $KEY: $PAPER_INFO. "
    "Return a list of keys, separated by commas. "
    'Return "None", if no papers are applicable. '
    "Choose papers that are relevant, from reputable sources, and timely "
    "(if the question requires timely information).\n\n"
    "Question: {question}\n\n"
    "Papers: {papers}\n\n"
    "Selected keys:"
)
citation_prompt = (
    "Provide the citation for the following text in MLA Format. "
    "Do not write an introductory sentence. "
    "If reporting date accessed, the current year is 2024\n\n"
    "{text}\n\n"
    "Citation:"
)

structured_citation_prompt = (
    "Extract the title, authors, and doi as a JSON from this MLA citation. "
    "If any field can not be found, return it as null. "
    "Use title, authors, and doi as keys, author's value should be a list of authors. "
    "{citation}\n\n"
    "Citation JSON:"
)

default_system_prompt = (
    "Answer in a direct and concise tone. "
    "Your audience is an expert, so be highly specific. "
    "If there are ambiguous terms or acronyms, first define them."
)

# NOTE: we use double curly braces here so it's not considered an f-string template
summary_json_system_prompt = """\
Provide a summary of the relevant information that could help answer the question based on the excerpt. Respond with the following JSON format:

{{
  "summary": "...",
  "relevance_score": "..."
}}

where `summary` is relevant information from the text - {summary_length} words. `relevance_score` is an integer 1-10 for the relevance of `summary` to the question.
"""  # noqa: E501

env_system_prompt = (
    # Matching https://github.com/langchain-ai/langchain/blob/langchain%3D%3D0.2.3/libs/langchain/langchain/agents/openai_functions_agent/base.py#L213-L215
    "You are a helpful AI assistant."
)
env_reset_prompt = (
    "Use the tools to answer the question: {question}"
    "\n\nWhen the answer looks sufficient,"
    " you can terminate by calling the {complete_tool_name} tool."
    " If the answer does not look sufficient,"
    " and you have already tried to answer several times with different evidence,"
    " terminate by calling the {complete_tool_name} tool."
    " The current status of evidence/papers/cost is {status}"
)

# Prompt templates for use with LitQA
QA_PROMPT_TEMPLATE = "Q: {question}\n\nOptions:\n{options}"
EVAL_PROMPT_TEMPLATE = (
    "Given the following question and a proposed answer to the question, return the"
    " single-letter choice in the question that matches the proposed answer."
    " If the proposed answer is blank or an empty string,"
    " or multiple options are matched, respond with '0'."
    "\n\nQuestion: {qa_prompt}"
    "\n\nProposed Answer: {qa_answer}"
    "\n\nSingle Letter Answer:"
)

CONTEXT_OUTER_PROMPT = "{context_str}\n\nValid Keys: {valid_keys}"
CONTEXT_INNER_PROMPT_NOT_DETAILED = "{name}: {text}"
CONTEXT_INNER_PROMPT = f"{CONTEXT_INNER_PROMPT_NOT_DETAILED}\nFrom {{citation}}"
