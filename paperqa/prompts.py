from datetime import datetime

# ruff: noqa: E501

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
    f"If reporting date accessed, the current year is {datetime.now().year}\n\n"
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
"""

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


# Prompts for LFRQA
lfrqa_system_prompt = (
    # From RAG-QA Arena (https://arxiv.org/pdf/2407.13998) Table 13 and 15
    "Pairwise Evaluation: Instruction and Rubric\n"
    "We will show you a query and a pair of answers to the query. "
    "You need to provide your preference over this pair of answers.\n"
    "First, try your best to determine whether the information in an answer can help "
    "truthfully answer the query. Then rate your preference based on Helpfulness and "
    "Truthfulness.\n"
    "- Helpfulness: information that is helpful/relevant to answer the query. An ideal "
    "answer consists of only information that is helpful/relevant to answer the query.\n"
    "- Truthfulness: information that you believe is correct to answer the query. By our "
    "definition, truthful information should be helpful information. If you find it "
    "difficult to determine the truthfulness of some information, consider it untruthful. "
    "Often time, this is due to not enough context provided in the answer. Another source "
    "of untruthfulness is when conflicting information is presented, and the answer does "
    "not reconcile them in a coherent way.\n"
    "<rubric>\n"
    "Here is how you judge (in the order of importance):\n"
    "- If one answer has all truthful information while the other has some untruthful "
    "information, prefer the all-truthful one.\n"
    "- If both have some untruthful information, prefer the one with less untruthful "
    "information.\n"
    "- If both have all truthful information, prefer the one with more truthful or helpful "
    "information.\n"
    "- If two answers look equally good, or it is too hard to judge using the three cases "
    "above, then you are 'not sure' which one is better.\n"
    "</rubric>\n\n"
    "Label 1\n"
    "Query\n"
    "difference between 'publicly' and 'publically'.\n"
    "<query> difference between 'publicly' and 'publically'. </query>\n"
    "<answer 1> Both 'publicly' and 'publically' bear no difference in meaning, as they are "
    "essentially alternative spellings of the same concept. Publicly is more widely used, "
    "but the existence of 'publically' in reputable sources like the OED means it cannot be "
    "dismissed as simply incorrect. Some opinions hold that 'publicly' is the older irregular "
    "form, still preached by a lot of grammars, and 'publically,' on the other hand, is the "
    "newer and regular form. </answer 1>\n"
    "<answer 2> There is no difference in meaning between 'publicly' and 'publically'; they "
    "are alternative spellings of the same word. </answer 2>\n"
    "Both <answer 1> and <answer 2> are truthful. However, <answer 1> provides more truthful "
    "information as the context to compare the two terms. Therefore, <answer 1> is better.\n\n"
    "Label 2\n"
    "Query\n"
    "what did European/American historical cooks do with the egg whites?\n"
    "<query> what did European/American historical cooks do with the egg whites? </query>\n"
    "<answer 1> Historical European and American cooks used egg whites for making egg white "
    "omelettes and egg white pasta, as well as for stiffening clothing, similar to how starch "
    "is used today. Egg whites have found their place in various non-culinary applications "
    "throughout history, such as in the clarification of beer and wine, in the conservation of "
    "books through bookbinding and gilding, and in makeup as an ancient form of nail polish. "
    "</answer 1>\n"
    "<answer 2> They were also utilized historically as a stiffening agent for clothing, similar "
    "to how starch is used today. The culinary landscape was not left untouched, with egg whites "
    "making their way into recipes for omelettes and pastas as early as the 15th century. "
    "</answer 2>\n"
    "Both <answer 1> and <answer 2> provide several usages of egg whites for European/American "
    "cooks.\n"
    "<answer 2> clearly provides more options with fully explained details. Therefore, <answer 2> "
    "is better.\n\n"
    "Label 0\n"
    "Query\n"
    "should utf-16 be considered harmful?\n"
    "<query> should utf-16 be considered harmful? </query>\n"
    "<answer 1> The question of whether UTF-16 should be considered harmful is subject to differing "
    "opinions. One perspective suggests that UTF-16 is harmful due to a historical misconception "
    "about character encoding and recommends UTF-8 as the superior choice for various text interfaces. "
    "Another viewpoint argues that UTF-16 is not inherently harmful, emphasizing its utility in specific "
    "scenarios where it serves as a compromise between simplicity and compactness. The choice between "
    "UTF-16 and other encodings like UTF-8 depends on the specific requirements of the application, such "
    "as compatibility with ASCII or the need to efficiently encode certain character sets. </answer 1>\n"
    "<answer 2> UTF-16 should not be considered harmful. However, contrasting views argue that UTF-16 "
    "should indeed be considered harmful. Some argue that the very reason UTF-16 exists is because some "
    "time ago there used to be a misguided belief that WideChar is going to be what UCS-4 now is. "
    "Additionally, the harmfulness of UTF-16 is tied to issues with exercising code. </answer 2>\n"
    "Both <answer 1> and <answer 2> reconcile the two conflicting views with detailed explanation.\n"
    "I am not sure which one is better."
)

lfrqa_prompt_template = (
    # From RAG-QA Arena (https://arxiv.org/pdf/2407.13998) Table 14
    "Query is in the <query></query> tags. Answer 1 is in <answer 1></answer 1>,"
    "and Answer 2 is in <answer 2></answer 2>.\n"
    "<query> {question} </query>\n"
    "<answer 1> {answer1} </answer 1>\n"
    "<answer 2> {answer2} </answer 2>\n"
    "Review the rubric in <rubric> tags,\n"
    "- if you prefer <answer 1>, output 1.\n"
    "- if you prefer <answer 2>, output 2.\n"
    "- if you are not sure, output 0.\n"
    "First, think step by step, put your thinking in <thinking></thinking> tags.\n"
    "Your thinking must be shorter than 50 words.\n"
    "Then, provide your rating inside <rating></rating> tags.\n"
    "Remember your rating should be 0 if you are not sure, and your rating must be either 0, 1, or 2."
)
