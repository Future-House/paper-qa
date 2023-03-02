import langchain.prompts as prompts
from datetime import datetime

summary_prompt = prompts.PromptTemplate(
    input_variables=["question", "context_str"],
    template="Summarize and provide direct quotes from the text below to help answer a question. "
    "Do not directly answer the question, instead provide a summary and quotes with the context of the question. "
    "Do not use outside sources. "
    'Reply with "Not applicable" if the text is unrelated to the question. '
    "Use 75 or less words."
    "\n\n"
    "{context_str}\n"
    "\n"
    "Question: {question}\n"
    "Relevant Information Summary:",
)


qa_prompt = prompts.PromptTemplate(
    input_variables=["question", "context_str", "length"],
    template="Write an answer ({length}) "
    "for the question below solely based on the provided context. "
    "If the context is irrelevant, "
    'reply "I cannot answer". '
    "For each sentence in your answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences, like (Example2012). "
    "Answer in an unbiased and scholarly tone. Make clear what is your opinion. "
    "Use Markdown for formatting code or text, and try to use direct quotes to support arguments.\n\n"
    "{context_str}\n"
    "Question: {question}\n"
    "Answer: ",
)


search_prompt = prompts.PromptTemplate(
    input_variables=["question"],
    template="We want to answer the following question: {question} \n"
    "Provide three different targeted keyword searches (one search per line) "
    "that will find papers that help answer the question. Do not use boolean operators. "
    "Recent years are 2021, 2022, 2023.\n\n"
    "1.",
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

chat_pref = [
    {
        "role": "system",
        "content": "You are a scholarly researcher that answers in an unbiased, scholarly tone. "
        "You sometimes refuse to answer if there is insufficient information.",
    }
]
