summary_prompt = (
    "Summarize the excerpt below to help answer a question.\n\n"
    "Excerpt from {citation}\n\n----\n\n{text}\n\n----\n\n"
    "Question: {question}\n\n"
    "Do not directly answer the question, instead summarize to give evidence to help "
    "answer the question. Stay detailed; report specific numbers, equations, or "
    'direct quotes (marked with quotation marks). Reply "Not applicable" if the '
    "excerpt is irrelevant. At the end of your response, provide an integer score "
    "from 1-10 on a newline indicating relevance to question. Do not explain your score."
    "\n\nRelevant Information Summary ({summary_length}):"
)

summary_json_prompt = (
    "Excerpt from {citation}\n\n----\n\n{text}\n\n----\n\n" "Question: {question}\n\n"
)

qa_prompt = (
    "Answer the question below with the context.\n\n"
    "Context (with relevance scores):\n\n{context}\n\n----\n\n"
    "Question: {question}\n\n"
    "Write an answer based on the context. "
    "If the context provides insufficient information and "
    "the question cannot be directly answered, reply "
    '"I cannot answer."'
    "For each part of your answer, indicate which sources most support "
    "it via citation keys at the end of sentences, "
    "like (Example2012Example pages 3-4). Only cite from the context "
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
    "(if the question requires timely information). \n\n"
    "Question: {question}\n\n"
    "Papers: {papers}\n\n"
    "Selected keys:"
)
citation_prompt = (
    "Provide the citation for the following text in MLA Format. "
    "If reporting date accessed, the current year is 2024\n\n"
    "{text}\n\n"
    "Citation:"
)

default_system_prompt = (
    "Answer in a direct and concise tone. "
    "Your audience is an expert, so be highly specific. "
    "If there are ambiguous terms or acronyms, first define them. "
)

summary_json_system_prompt = """\
Provide a summary of the relevant information that could help answer the question based on the excerpt. Respond with the following JSON format:

{{
  "summary": "...",
  "relevance_score": "..."
}}

where `summary` is relevant information from text - {summary_length} words and `relevance_score` is the relevance of `summary` to answer question (out of 10)
"""  # noqa: E501
