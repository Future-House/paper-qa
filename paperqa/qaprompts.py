import langchain.agents as agents
import langchain.prompts as prompts
import langchain.chains as chains
import langchain.llms as llms

_distill_prompt = prompts.PromptTemplate(
    input_variables=["question", "context_str"],
    template="Provide relevant information that will help answer a question from the context below. "
    "Summarize the information in an unbiased tone. Use direct quotes "
    "where possible. Do not directly answer the question. "
    'State "Not applicable" if the context is irrelevant. '
    "Use 35 or less words."
    "\n\n"
    "{context_str}\n"
    "\n"
    "Question: {question}\n"
    "Relevant Information:",
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
    input_variables=["question", "context_str"],
    template="Generate a comprehensive answer (about 100 words) "
    "for the question below solely based on the provided context. "
    "If the context is insufficient to "
    'answer, reply "I cannot answer". '
    "For each sentence in your answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences like (Foo2012). "
    "Answer in an unbiased, balanced, and scientific tone. "
    # "To answer, start by writing out the reasoning and then "
    # "write a complete unbiased answer prefixed by \"Answer:\""
    "\n--------------------\n"
    "{context_str}\n"
    "----------------------\n"
    "Question: {question}\n"
    "Answer: ",
)
qa_chain = chains.LLMChain(llm=llms.OpenAI(temperature=0.1), prompt=_qa_prompt)
