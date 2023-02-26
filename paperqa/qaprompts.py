import langchain.prompts as prompts

summary_prompt = prompts.PromptTemplate(
    input_variables=["question", "context_str"],
    template="Provide relevant information that will help answer a question only from the context below. "
    "Summarize the information in an unbiased tone. Use direct quotes "
    "where possible. Do not directly answer the question. "
    'Reply with "Not applicable" if the context is irrelevant to the question. '
    "Use 35 or less words."
    "\n\n"
    "{context_str}\n"
    "\n"
    "Question: {question}\n"
    "Relevant Information Summary:",
)


qa_prompt = prompts.PromptTemplate(
    input_variables=["question", "context_str", "length"],
    template="Write a comprehensive answer ({length}) "
    "for the question below solely based on the provided context. "
    "If the context is insufficient, "
    'reply "I cannot answer". '
    "For each sentence in your answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences, like (Foo2012). "
    "Answer in an unbiased, balanced, and scientific tone. "
    "Use Markdown for formatting code or text.\n\n"
    "{context_str}\n"
    "Question: {question}\n"
    "Answer: ",
)

edit_prompt = prompts.PromptTemplate(
    input_variables=["question", "answer"],
    template="The original question is: {question} "
    "We have been provided the following answer: {answer} "
    "Part of it may be truncated, please edit the answer to make it complete. "
    "If it appears to be complete, repeat it unchanged.\n\n",
)


search_prompt = prompts.PromptTemplate(
    input_variables=["question"],
    template="We want to answer the following question: {question} \n"
    "Provide three different targeted keyword searches (one search per line) "
    "that will find papers that help answer the question. Do not use boolean operators. "
    "Recent years are 2021, 2022, 2023.\n\n"
    "1.",
)
