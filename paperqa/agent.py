from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor

from .docs import Answer, Docs
from .qaprompts import make_chain, select_paper_prompt


def status(answer: Answer, docs: Docs):
    return f" Status: Current Papers: {len(docs.doc_previews())} Current Evidence: {len(answer.contexts)} Current Cost: ${answer.cost:.2f}"


class PaperSelection(BaseTool):
    name = "Select Papers"
    description = "Select from current papers. Provide a desired question to answer as a string to use for choosing papers."
    docs: Docs = None
    answer: Answer = None
    chain: LLMChain = None

    def __init__(self, docs, answer):
        # call the parent class constructor
        super(PaperSelection, self).__init__()

        self.docs = docs
        self.answer = answer
        self.chain = make_chain(select_paper_prompt, self.docs.summary_llm)

    def _run(self, query: str) -> str:
        result = self.docs.doc_match(query)
        if result is None or result.strip().startswith("None"):
            return "No relevant papers found."
        return result + status(self.answer, self.docs)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class ReadPapers(BaseTool):
    name = "Gather Evidence"
    description = (
        "Give a specific question to a researcher that will return evidence for it. "
        "Optionally, you may specify papers using their key provided by the Select Papers tool. "
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
        l0 = len(self.answer.contexts)
        self.docs.get_evidence(self.answer, key_filter=keys)
        l1 = len(self.answer.contexts)
        self.answer.question = old
        return f"Added {l1 - l0} pieces of evidence." + status(self.answer, self.docs)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class AnswerTool(BaseTool):
    name = "Propose Answer"
    description = "Ask a researcher to propose an answer using evidence from papers. The input is the question to be answered."
    docs: Docs = None
    answer: Answer = None
    return_direct = True

    def __init__(self, docs, answer):
        # call the parent class constructor
        super(AnswerTool, self).__init__()

        self.docs = docs
        self.answer = answer

    def _run(self, query: str) -> str:
        self.answer = self.docs.query(query, answer=self.answer)
        if "cannot answer" in self.answer.answer:
            self.answer = Answer(self.answer.question)
            return (
                "Failed to answer question. Deleting evidence. Consider rephrasing question or evidence statement."
                + status(self.answer, self.docs)
            )
        return self.answer.answer + status(self.answer, self.docs)

    def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class Search(BaseTool):
    name = "Paper Search"
    description = (
        "Search for papers to add to cur. Input should be a string of keywords."
    )
    docs: Docs = None
    answer: Answer = None

    def __init__(self, docs, answer):
        # call the parent class constructor
        super(Search, self).__init__()

        self.docs = docs
        self.answer = answer

    def _run(self, query: str) -> str:
        try:
            import paperscraper
        except ImportError:
            raise ImportError(
                "Please install paperscraper (github.com/blackadad/paper-scraper) to use agent"
            )

        papers = paperscraper.search_papers(
            query, limit=5, _limit=20, verbose=False, pdir=self.docs.index_path
        )
        for path, data in papers.items():
            try:
                self.docs.add(path, citation=data["citation"])
            except:
                pass
        return status(self.answer, self.docs)

    def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


def make_tools(docs, answer):
    tools = []

    tools.append(Search(docs, answer))
    tools.append(PaperSelection(docs, answer))
    tools.append(ReadPapers(docs, answer))
    tools.append(AnswerTool(docs, answer))
    return tools


def run_agent(docs, question, llm=None):
    if llm is None:
        llm = ChatOpenAI(temperature=0.0, model_name="gpt-4")
    answer = Answer(question)
    tools = make_tools(docs, answer)
    mrkl = RetryAgentExecutor.from_agent_and_tools(
        tools=tools,
        agent=ChatZeroShotAgent.from_llm_and_tools(llm, tools),
        verbose=True,
    )
    mrkl.run(
        f"Answer question: {question}. Search for papers, gather evidence, and answer. If you do not have enough evidence, you can search for more papers (preferred) or gather more evidence. You may rephrase or breaking-up the question in those steps. "
        "Once you have five pieces of evidence, or you have tried for a while, call the Propose Answer tool. "
    )

    return answer
