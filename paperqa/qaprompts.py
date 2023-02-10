import langchain.prompts as prompts
import langchain.chains as chains
import langchain.llms as llms

_distill_prompt = prompts.PromptTemplate(
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
distill_chain = chains.LLMChain(
    llm=llms.OpenAI(temperature=0.1), prompt=_distill_prompt
)

_query_prompt = prompts.PromptTemplate(
    input_variables=["question"],
    template="I would like to find scholarly papers to answer this question: {question}. "
    'A search query that would bring up papers related to this answer would be: "',
)
query_chain = chains.LLMChain(
    llm=llms.OpenAI(temperature=0.05, model_kwargs={"stop": ['"']}),
    prompt=_query_prompt,
)

_qa_prompt = prompts.PromptTemplate(
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
qa_chain = chains.LLMChain(llm=llms.OpenAI(temperature=0.1), prompt=_qa_prompt)

_edit_prompt = prompts.PromptTemplate(
    input_variables=["question", "answer"],
    template="The original question is: {question} "
    "We have been provided the following answer: {answer} "
    "Part of it may be truncated, please edit the answer to make it complete. "
    "If it appears to be complete, repeat it unchanged.\n\n",
)
edit_chain = chains.LLMChain(llm=llms.OpenAI(temperature=0.1), prompt=_edit_prompt)
