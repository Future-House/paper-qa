from langchain.tools import BaseTool
from .docs import Answer, Docs
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI


def status(answer: Answer, docs: Docs):
    return f" Status: Current Papers: {len(docs.doc_previews())} Current Evidence: {len(answer.contexts)} Current Cost: {answer.cost}"


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
        self.docs.get_evidence(self.answer, key_filter=keys)
        self.answer.question = old
        return status(self.answer, self.docs)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class AnswerTool(BaseTool):
    name = "Propose Answer"
    description = "Ask a researcher to propose an answer using evidence from papers. The input is the question to be answered."
    docs: Docs = None
    answer: Answer = None

    def __init__(self, docs, answer):
        # call the parent class constructor
        super(AnswerTool, self).__init__()

        self.docs = docs
        self.answer = answer

    def _run(self, query: str) -> str:
        self.answer = self.docs.query(
            query, answer=self.answer, length_prompt="length as long as needed"
        )
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
    name = "Paper Search"
    description = "Search for papers to add to current papers. Input should be a string of keywords."
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

        papers = paperscraper.search_papers(query, limit=20, verbose=False)
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
    # putting here until langchain PR is merged
    from langchain.tools.exception.tool import ExceptionTool

    tools = []

    tools.append(Search(docs, answer))
    tools.append(ReadPapers(docs, answer))
    tools.append(AnswerTool(docs, answer))
    tools.append(ExceptionTool())
    return tools


def run_agent(docs, question, llm=None):
    if llm is None:
        llm = ChatOpenAI(temperature=0.1, model="gpt-4")
    answer = Answer(question)
    tools = make_tools(docs, answer)
    mrkl = initialize_agent(
        tools, llm, agent="chat-zero-shot-react-description", verbose=True
    )
    mrkl.run(
        f"Answer question: {question}. Search for papers, gather evidence, and answer. "
        "Once you have at least five pieces of evidence, call the Propose Answer tool. "
        "If you do not have enough evidence, search with different keywords. "
    )

    return answer
