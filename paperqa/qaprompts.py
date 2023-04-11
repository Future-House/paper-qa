import langchain.prompts as prompts
from datetime import datetime
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate


summary_prompt = prompts.PromptTemplate(
    input_variables=["question", "context_str", "citation"],
    template="Summarize and provide direct quotes from the text below to help answer a question. "
    "Do not directly answer the question, instead summarize and "
    "quote to give evidence to help answer the question. "
    "Do not use outside sources. "
    'Reply with "Not applicable" if the text is unrelated to the question. '
    "Use 150 or less words."
    "\n\n"
    "{context_str}\n"
    "Extracted from {citation}\n"
    "Question: {question}\n"
    "Relevant Information Summary:",
)


qa_prompt = prompts.PromptTemplate(
    input_variables=["question", "context_str", "length"],
    template="Write an answer ({length}) "
    "for the question below based on the provided context. "
    "If the context provides insufficient information, "
    'reply "I cannot answer". '
    "For each sentence in your answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences, like (Example2012). "
    "Answer in an unbiased, comprehensive, and scholarly tone. "
    "Use Markdown for formatting code or text, and try to use direct quotes to support arguments.\n\n"
    "{context_str}\n"
    "Question: {question}\n"
    "Answer: ",
)


search_prompt = prompts.PromptTemplate(
    input_variables=["question"],
    template="We want to answer the following question: {question} \n"
    "Provide three keyword searches (one search per line) "
    "that will find papers to help answer the question. Do not use boolean operators. "
    "Recent years are 2021, 2022, 2023.\n\n"
    "1.",
)


select_paper_prompt = prompts.PromptTemplate(
    input_variables=["instructions", "papers"],
    template="Select papers according to instructions below. "
    "Papers are listed as $KEY: $PAPER_INFO. "
    "Return a list of keys, separated by commas. "
    'Return "None", if no papers are applicable. \n\n'
    "Instructions: {instructions}\n\n"
    "{papers}\n\n"
    "Selected keys:",
)


def _get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y")


citation_prompt = prompts.PromptTemplate(
    input_variables=["text"],
    template="Provide a possible citation for the following text in MLA Format. Today's date is {date}\n"
    "{text}\n\n"
    "Citation:",
    partial_variables={"date": _get_datetime},
)


def make_chain(prompt, llm):
    if type(llm) == ChatOpenAI:
        system_message_prompt = SystemMessage(
            content="You are a scholarly researcher that answers in an unbiased, scholarly tone. "
            "You sometimes refuse to answer if there is insufficient information.",
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=prompt)
        prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
    return LLMChain(prompt=prompt, llm=llm)
