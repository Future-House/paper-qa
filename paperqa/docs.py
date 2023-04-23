import json
from typing import List, Optional, Tuple, Union
from functools import reduce
import sys
import asyncio
from .readers import read_doc
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.callbacks import get_openai_callback
import math
import string
from typing import Union, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import langchain.prompts as prompts
from datetime import datetime
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate

StrPath = Union[str, Path]


@dataclass
class Answer:
    """A class to hold the answer to a question."""

    question: str
    answer: str = ""
    context: str = ""
    contexts: List[Any] = None
    references: str = ""
    formatted_answer: str = ""
    passages: Dict[str, str] = None
    tokens: int = 0
    cost: float = 0

    def __post_init__(self):
        """Initialize the answer."""
        if self.contexts is None:
            self.contexts = []
        if self.passages is None:
            self.passages = {}

    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer


@dataclass
class Context:
    """A class to hold the context of a question."""

    key: str
    citation: str
    context: str
    text: str

    def __str__(self) -> str:
        """Return the context as a string."""
        return self.context


def _get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y")


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


def maybe_is_text(s, thresh=2.5):
    if len(s) == 0:
        return False
    # Calculate the entropy of the string
    entropy = 0
    for c in string.printable:
        p = s.count(c) / len(s)
        if p > 0:
            entropy += -p * math.log2(p)

    # Check if the entropy is within a reasonable range for text
    if entropy > thresh:
        return True
    return False


def md5sum(file_path: StrPath) -> str:
    import hashlib

    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


class Docs:
    """A collection of documents to be used for answering questions."""

    def __init__(
            self,
            chunk_size_limit: int = 3000,
            llm: Optional[Union[LLM, str]] = None,
            summary_llm: Optional[Union[LLM, str]] = None,
            name: str = "default",
            index_path: Optional[Path] = None,
            embeddings: Optional[Embeddings] = None,
    ) -> None:
        """Initialize the collection of documents.

        Args:
            chunk_size_limit: The maximum number of characters to use for a single chunk of text.
            llm: The language model to use for answering questions. Default - OpenAI chat-gpt-turbo
            summary_llm: The language model to use for summarizing documents. If None, llm is used.
            name: The name of the collection.
            index_path: The path to the index file IF pickled. If None, defaults to using name in $HOME/.paperqa/name
        """
        self.docs = dict()
        self.chunk_size_limit = chunk_size_limit
        self.keys = set()
        self._faiss_index = None
        self._doc_index = None
        self.update_llm(llm, summary_llm)
        if index_path is None:
            index_path = Path.cwd() / "data" / name
        self.index_path = index_path
        self.name = name
        if embeddings is None:
            embeddings = OpenAIEmbeddings()
        self.embeddings = embeddings

    def update_llm(
            self,
            llm: Optional[Union[LLM, str]] = None,
            summary_llm: Optional[Union[LLM, str]] = None,
    ) -> None:
        """Update the LLM for answering questions."""
        if llm is None:
            llm = "gpt-3.5-turbo"
        if type(llm) is str:
            llm = ChatOpenAI(temperature=0.1, model=llm)
        if type(summary_llm) is str:
            summary_llm = ChatOpenAI(temperature=0.1, model=summary_llm)
        self.llm = llm
        if summary_llm is None:
            summary_llm = llm
        self.summary_llm = summary_llm
        self.summary_chain = make_chain(prompt=prompts.PromptTemplate(
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
        ), llm=summary_llm)
        self.qa_chain = make_chain(prompt=prompts.PromptTemplate(
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
        ), llm=llm)

    def add(
            self,
            path: str,
            citation: Optional[str] = None,
            key: Optional[str] = None,
            disable_check: bool = False,
            chunk_chars: Optional[int] = 3000,
            overwrite: bool = False,
    ) -> None:
        """Add a document to the collection."""

        # first check to see if we already have this document
        # this way we don't make api call to create citation on file we already have
        md5 = md5sum(path)
        if path in self.docs:
            raise ValueError(f"Document {path} already in collection.")
        suffix = ""
        while key + suffix in self.keys:
            # move suffix to next letter
            if suffix == "":
                suffix = "a"
            else:
                suffix = chr(ord(suffix) + 1)
        key += suffix

        texts, metadata = read_doc(path, citation, key, chunk_chars=chunk_chars)
        if not texts:
            return
        # loose check to see if document was loaded
        #
        if len("".join(texts)) < 10 or (
                not disable_check and not maybe_is_text("".join(texts))
        ):
            raise ValueError(
                f"This does not look like a text document: {path}. Path disable_check to ignore this error."
            )
        if self._faiss_index is not None:
            self._faiss_index.add_texts(texts, metadatas=metadata)
        if self._doc_index is not None:
            self._doc_index.add_texts([citation], metadatas=[{"key": key}])
        self.docs[path] = dict(texts=texts, metadata=metadata, key=key, md5=md5)
        self.keys.add(key)

    def clear(self) -> None:
        """Clear the collection of documents."""
        self.docs = dict()
        self.keys = set()
        self._faiss_index = None
        self._doc_index = None
        # delete index file
        pkl = self.index_path / "index.pkl"
        if pkl.exists():
            pkl.unlink()
        fs = self.index_path / "index.faiss"
        if fs.exists():
            fs.unlink()

    def doc_previews(self) -> List[Tuple[int, str, str]]:
        """Return a list of tuples of (key, citation) for each document."""
        return [
            (
                len(doc["texts"]),
                doc["metadata"][0]["dockey"],
                doc["metadata"][0]["citation"],
            )
            for doc in self.docs.values()
        ]

    def doc_match(self, query: str, k: int = 25) -> List[str]:
        """Return a list of documents that match the query."""
        if len(self.docs) == 0:
            return ""
        if self._doc_index is None:
            texts = [doc["metadata"][0]["citation"] for doc in self.docs.values()]
            metadatas = [
                {"key": doc["metadata"][0]["dockey"]} for doc in self.docs.values()
            ]
            self._doc_index = FAISS.from_texts(
                texts, metadatas=metadatas, embedding=self.embeddings
            )
        docs = self._doc_index.similarity_search(query, k=k)
        template = prompts.PromptTemplate(
            input_variables=["instructions", "papers"],
            template="Select papers according to instructions below. "
                     "Papers are listed as $KEY: $PAPER_INFO. "
                     "Return a list of keys, separated by commas. "
                     'Return "None", if no papers are applicable. \n\n'
                     "Instructions: {instructions}\n\n"
                     "{papers}\n\n"
                     "Selected keys:",
        )
        chain = make_chain(template, self.summary_llm)
        papers = [f"{d.metadata['key']}: {d.page_content}" for d in docs]
        print("Yooooooooooooooooo", template.format(instructions=query, papers="\n".join(papers)))
        result = chain.run(instructions=query, papers="\n".join(papers))
        print(json.dumps(result, indent=2))
        return result

    def _build_faiss_index(self):
        if self._faiss_index is None:
            texts = reduce(
                lambda x, y: x + y, [doc["texts"] for doc in self.docs.values()], []
            )
            metadatas = reduce(
                lambda x, y: x + y, [doc["metadata"] for doc in self.docs.values()], []
            )
            self._faiss_index = FAISS.from_texts(
                texts, self.embeddings, metadatas=metadatas
            )

    def get_evidence(
            self,
            answer: Answer,
            k: int = 3,
            max_sources: int = 5,
            key_filter: Optional[List[str]] = None,
    ) -> Answer:
        # special case for jupyter notebooks
        if "get_ipython" in globals() or "google.colab" in sys.modules:
            import nest_asyncio

            nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.aget_evidence(
                answer,
                k=k,
                max_sources=max_sources,
                key_filter=key_filter,
            )
        )

    async def aget_evidence(
            self,
            answer: Answer,
            k: int = 3,
            max_sources: int = 5,
            key_filter: Optional[List[str]] = None,
    ) -> Answer:
        if len(self.docs) == 0:
            return answer
        if self._faiss_index is None:
            self._build_faiss_index()
        _k = k
        if key_filter is not None:
            _k = k * 10  # heuristic
        docs = self._faiss_index.similarity_search(
            answer.question, k=_k, fetch_k=5 * _k
        )

        async def process(doc):
            if key_filter is not None and doc.metadata["dockey"] not in key_filter:
                return None
            # check if it is already in answer (possible in agent setting)
            if doc.metadata["key"] in [c.key for c in answer.contexts]:
                return None
            c = Context(
                key=doc.metadata["key"],
                citation=doc.metadata["citation"],
                context=await self.summary_chain.arun(
                    question=answer.question,
                    context_str=doc.page_content,
                    citation=doc.metadata["citation"],
                ),
                text=doc.page_content,
            )
            if "Not applicable" not in c.context:
                return c
            return None

        with get_openai_callback() as cb:
            contexts = await asyncio.gather(*[process(doc) for doc in docs])
        answer.tokens += cb.total_tokens
        answer.cost += cb.total_cost
        contexts = [c for c in contexts if c is not None]
        if len(contexts) == 0:
            return answer
        contexts = sorted(contexts, key=lambda x: len(x.context), reverse=True)
        contexts = contexts[:max_sources]
        # add to answer (if not already there)
        keys = [c.key for c in answer.contexts]
        for c in contexts:
            if c.key not in keys:
                answer.contexts.append(c)

        context_str = "\n\n".join(
            [
                f"{c.key}: {c.context}"
                for c in answer.contexts
                if "Not applicable" not in c.context
            ]
        )
        valid_keys = [
            c.key for c in answer.contexts if "Not applicable" not in c.context
        ]
        if len(valid_keys) > 0:
            context_str += "\n\nValid keys: " + ", ".join(valid_keys)
        answer.context = context_str
        return answer

    def query(
            self,
            query: str,
            k: int = 10,
            max_sources: int = 5,
            length_prompt: str = "about 100 words",
            answer: Optional[Answer] = None,
            key_filter: Optional[bool] = None,
    ) -> Answer:
        # special case for jupyter notebooks
        if "get_ipython" in globals() or "google.colab" in sys.modules:
            import nest_asyncio

            nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.aquery(
                query,
                k=k,
                max_sources=max_sources,
                length_prompt=length_prompt,
                answer=answer,
                key_filter=key_filter,
            )
        )

    async def aquery(
            self,
            query: str,
            k: int = 10,
            max_sources: int = 5,
            length_prompt: str = "about 100 words",
            answer: Optional[Answer] = None,
            key_filter: Optional[bool] = None,
    ) -> Answer:
        if k < max_sources:
            raise ValueError("k should be greater than max_sources")
        if answer is None:
            answer = Answer(query)
        if len(answer.contexts) == 0:
            if key_filter or (key_filter is None and len(self.docs) > 5):
                with get_openai_callback() as cb:
                    keys = self.doc_match(answer.question)
                answer.tokens += cb.total_tokens
                answer.cost += cb.total_cost
            answer = await self.aget_evidence(
                answer,
                k=k,
                max_sources=max_sources,
                key_filter=keys if key_filter else None,
            )
        context_str, contexts = answer.context, answer.contexts
        bib = dict()
        passages = dict()
        if len(context_str) < 10:
            answer_text = (
                "I cannot answer this question due to insufficient information."
            )
        else:
            with get_openai_callback() as cb:
                answer_text = await self.qa_chain.arun(
                    question=query, context_str=context_str, length=length_prompt
                )
            answer.tokens += cb.total_tokens
            answer.cost += cb.total_cost
        # it still happens lol
        if "(Foo2012)" in answer_text:
            answer_text = answer_text.replace("(Foo2012)", "")
        for c in contexts:
            key = c.key
            text = c.context
            citation = c.citation
            # do check for whole key (so we don't catch Callahan2019a with Callahan2019)
            skey = key.split(" ")[0]
            if skey + " " in answer_text or skey + ")" or skey + "," in answer_text:
                bib[skey] = citation
                passages[key] = text
        references = "\n\n".join(
            [f"{i + 1}. ({k}): {c}" for i, (k, c) in enumerate(bib.items())]
        )
        formatted_answer = f"Question: {query}\n\n{answer_text}\n"
        formatted_answer += f"\nContext\n\n{answer.context}\n"
        formatted_answer += f"\nTokens Used: {answer.tokens} Cost: ${answer.cost:.2f}"
        answer.answer = answer_text
        answer.formatted_answer = formatted_answer
        answer.references = references
        answer.passages = passages
        return answer
