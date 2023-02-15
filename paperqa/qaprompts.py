import langchain.prompts as prompts

summary_prompt = prompts.PromptTemplate(
    input_variables=["question", "context_str"],
    template="Provide relevant information that will help answer a question from the context below. "
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
    "If the context is insufficient "
    'answer, reply "I cannot answer". '
    "For each sentence in your answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences, like (Foo2012). "
    "Answer in an unbiased, balanced, and scientific tone. "
    "Try to use the direct quotes, if present, from the context. "
    # "write a complete unbiased answer prefixed by \"Answer:\""
    "\n--------------------\n"
    "{context_str}\n"
    "----------------------\n"
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
