from langchain.tools import BaseTool
from langchain.chains import LLMChain
from .qaprompts import select_paper_prompt, make_chain
from .docs import Answer, Docs
import paperscraper
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI


def status(answer: Answer, docs: Docs):
    return f"| Papers: {len(docs.doc_previews)} Evidence: {len(answer.contexts)}"


class PaperChoice(BaseTool):
    name = "Choose Papers"
    description = "Ask a researcher to select papers based expert knowledge and paper citations. Only provide instructions as string for the researcher."
    docs: Docs = None
    answer: Answer = None
    chain: LLMChain = None

    def __init__(self, docs, answer):
        # call the parent class constructor
        super(PaperChoice, self).__init__()

        self.docs = docs
        self.answer = answer
        self.chain = make_chain(select_paper_prompt, self.docs.summary_llm)

    def _run(self, query: str) -> str:
        papers = [f"{d[1]}: {d[-1]}" for d in self.docs.doc_previews]
        result = self.chain.run(instructions=query, papers="\n".join(papers))
        if "None" in result:
            result = "No relevant papers found"
        return result + status(self.answer, self.docs)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class ReadPapers(BaseTool):
    name = "Gather Evidence"
    description = (
        "Give a specific question to a researcher that will return evidence for it. "
        "Optionally, you may specify papers using their key provided by the Choose Papers tool. "
        "Use the format: $QUESTION or use format $QUESTION|$KEY1,$KEY2,..."
    )
    docs: Docs = None
    answer: Answer = None

    def __init__(self, docs, answer):
        # call the parent class constructor
        super(ReadPapers, self).__init__()

        self.docs = docs
        self.answer = answer

    def _run(self, query: str) -> str:
        if "|" in query:
            question, keys = query.split("|")
            keys = [k.strip() for k in keys.split(",")]
        else:
            question = query
            keys = None
        # swap out the question
        old = self.answer.question
        self.answer.question = question
        # generator, so run it
        list(self.docs.get_evidence(self.answer, key_filter=keys))
        self.answer.question = old
        return status(self.answer, self.docs)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class AnswerTool(BaseTool):
    name = "Answer Question"
    description = "Ask a researcher to answer a question using evidence from papers. Input is the question to be answered."
    docs: Docs = None
    answer: Answer = None

    def __init__(self, docs, answer):
        # call the parent class constructor
        super(AnswerTool, self).__init__()

        self.docs = docs
        self.answer = answer

    def _run(self, query: str) -> str:
        self.docs.query(query, answer=self.answer)
        if "cannot answer" in self.answer.answer:
            self.answer = Answer(self.answer.question)
            return "Failed to answer question. Deleting evidence." + status(
                self.answer, self.docs
            )
        return self.answer.answer + status(self.answer, self.docs)

    def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class Search(BaseTool):
    name = "Search for Papers"
    description = (
        "Search for papers using Google Scholar. Input should be a string keywords."
    )
    docs: Docs = None
    answer: Answer = None

    def __init__(self, docs, answer):
        # call the parent class constructor
        super(Search, self).__init__()

        self.docs = docs
        self.answer = answer

    def _run(self, query: str) -> str:
        papers = paperscraper.search_papers(query, verbose=False)
        for path, data in papers.items():
            try:
                self.docs.add(path)
            except:
                pass
        return status(self.answer, self.docs)

    def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


def make_tools(docs, answer):

    tools = []

    tools.append(ReadPapers(docs, answer))
    tools.append(PaperChoice(docs, answer))
    tools.append(AnswerTool(docs, answer))
    tools.append(Search(docs, answer))

    return tools


def run_agent(docs, question, llm=None, budget=10000):
    if llm is None:
        llm = ChatOpenAI(temperature=0, model="gpt-4")
    answer = Answer(question)
    tools = make_tools(docs, answer)
    mrkl = initialize_agent(
        tools, llm, agent="chat-zero-shot-react-description", verbose=True
    )
    mrkl.run(
        f"Answer question: {question}. Find papers, gather evidence, and answer. "
        "Once you have five pieces of evidence, call the Answer tool."
    )

    return answer
