from typing import List, Optional, Tuple, Dict, Callable, Any
from functools import reduce
import os
import os
from pathlib import Path
import re
from .utils import maybe_is_text, maybe_is_truncated
from .qaprompts import (
    summary_prompt,
    qa_prompt,
    search_prompt,
    citation_prompt,
    chat_pref,
)
from dataclasses import dataclass
from .readers import read_doc
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI, OpenAIChat
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.cache import SQLiteCache
import langchain
from datetime import datetime

CACHE_PATH = Path.home() / ".paperqa" / "llm_cache.db"
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
langchain.llm_cache = SQLiteCache(CACHE_PATH)


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

    def __post_init__(self):
        """Initialize the answer."""
        if self.contexts is None:
            self.contexts = []
        if self.passages is None:
            self.passages = {}

    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer


class Docs:
    """A collection of documents to be used for answering questions."""

    def __init__(
        self,
        chunk_size_limit: int = 3000,
        llm: Optional[LLM] = None,
        summary_llm: Optional[LLM] = None,
        name: str = "default",
        index_path: Optional[Path] = None,
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
        if llm is None:
            llm = OpenAIChat(temperature=0.1, max_tokens=512, prefix_messages=chat_pref)
        if summary_llm is None:
            summary_llm = llm
        self.update_llm(llm, summary_llm)
        if index_path is None:
            index_path = Path.home() / ".paperqa" / name
        self.index_path = index_path
        self.name = name

    def update_llm(self, llm: LLM, summary_llm: Optional[LLM] = None) -> None:
        """Update the LLM for answering questions."""
        self.llm = llm
        if summary_llm is None:
            summary_llm = llm
        self.summary_llm = summary_llm
        self.summary_chain = LLMChain(prompt=summary_prompt, llm=summary_llm)
        self.qa_chain = LLMChain(prompt=qa_prompt, llm=llm)
        self.search_chain = LLMChain(prompt=search_prompt, llm=llm)
        self.cite_chain = LLMChain(prompt=citation_prompt, llm=llm)

    def add(
        self,
        path: str,
        citation: Optional[str] = None,
        key: Optional[str] = None,
        disable_check: bool = False,
        chunk_chars: Optional[int] = 3000,
    ) -> None:
        """Add a document to the collection."""

        if citation is None:
            # peak first chunk
            texts, _ = read_doc(path, "", "", chunk_chars=chunk_chars)
            with get_openai_callback() as cb:
                citation = self.cite_chain.run(texts[0])
            if len(citation) < 3 or "Unknown" in citation or "insufficient" in citation:
                citation = f"Unknown, {os.path.basename(path)}, {datetime.now().year}"

        if path in self.docs:
            raise ValueError(f"Document {path} already in collection.")
        if key is None:
            # get first name and year from citation
            try:
                author = re.search(r"([A-Z][a-z]+)", citation).group(1)
            except AttributeError:
                # panicking - no word??
                raise ValueError(
                    f"Could not parse key from citation {citation}. Consider just passing key explicitly - e.g. docs.py (path, citation, key='mykey')"
                )
            try:
                year = re.search(r"(\d{4})", citation).group(1)
            except AttributeError:
                year = ""
            key = f"{author}{year}"
        suffix = ""
        while key + suffix in self.keys:
            # move suffix to next letter
            if suffix == "":
                suffix = "a"
            else:
                suffix = chr(ord(suffix) + 1)
        key += suffix
        self.keys.add(key)

        texts, metadata = read_doc(path, citation, key, chunk_chars=chunk_chars)
        # loose check to see if document was loaded
        #
        if len("".join(texts)) < 10 or (
            not disable_check and not maybe_is_text("".join(texts))
        ):
            raise ValueError(
                f"This does not look like a text document: {path}. Path disable_check to ignore this error."
            )

        self.docs[path] = dict(texts=texts, metadata=metadata, key=key)
        if self._faiss_index is not None:
            self._faiss_index.add_texts(texts, metadatas=metadata)

    def clear(self) -> None:
        """Clear the collection of documents."""
        self.docs = dict()
        self.keys = set()
        self._faiss_index = None
        # delete index file
        pkl = self.index_path / "index.pkl"
        if pkl.exists():
            pkl.unlink()
        fs = self.index_path / "index.faiss"
        if fs.exists():
            fs.unlink()

    @property
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

    # to pickle, we have to save the index as a file
    def __getstate__(self):
        if self._faiss_index is None and len(self.docs) > 0:
            self._build_faiss_index()
        state = self.__dict__.copy()
        if self._faiss_index is not None:
            state["_faiss_index"].save_local(self.index_path)
        del state["_faiss_index"]
        # remove LLMs (they can have callbacks, which can't be pickled)
        del state["summary_chain"]
        del state["qa_chain"]
        del state["cite_chain"]
        del state["search_chain"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            self._faiss_index = FAISS.load_local(self.index_path, OpenAIEmbeddings())
        except:
            # they use some special exception type, but I don't want to import it
            self._faiss_index = None
        self.update_llm(
            OpenAIChat(temperature=0.1, max_tokens=512, prefix_messages=chat_pref)
        )

    def _build_faiss_index(self):
        if self._faiss_index is None:
            texts = reduce(
                lambda x, y: x + y, [doc["texts"] for doc in self.docs.values()], []
            )
            metadatas = reduce(
                lambda x, y: x + y, [doc["metadata"] for doc in self.docs.values()], []
            )
            self._faiss_index = FAISS.from_texts(
                texts, OpenAIEmbeddings(), metadatas=metadatas
            )

    def get_evidence(
        self,
        answer: Answer,
        k: int = 3,
        max_sources: int = 5,
        marginal_relevance: bool = True,
    ) -> str:
        if self._faiss_index is None:
            self._build_faiss_index()

        # want to work through indices but less k
        if marginal_relevance:
            docs = self._faiss_index.max_marginal_relevance_search(
                answer.question, k=k, fetch_k=5 * k
            )
        else:
            docs = self._faiss_index.similarity_search(
                answer.question, k=k, fetch_k=5 * k
            )
        for doc in docs:
            c = (
                doc.metadata["key"],
                doc.metadata["citation"],
                self.summary_chain.run(
                    question=answer.question, context_str=doc.page_content
                ),
                doc.page_content,
            )
            if "Not applicable" not in c[2]:
                answer.contexts.append(c)
                yield answer
            if len(answer.contexts) == max_sources:
                break
        context_str = "\n\n".join(
            [f"{k}: {s}" for k, c, s, t in answer.contexts if "Not applicable" not in s]
        )
        valid_keys = [k for k, c, s, t in answer.contexts if "Not applicable" not in s]
        if len(valid_keys) > 0:
            context_str += "\n\nValid keys: " + ", ".join(valid_keys)
        answer.context = context_str
        yield answer

    def generate_search_query(self, query: str) -> List[str]:
        """Generate a list of search strings that can be used to find
        relevant papers.

        Args:
            query (str): The query to generate search strings for.
        """

        search_query = self.search_chain.run(question=query)
        queries = [s for s in search_query.split("\n") if len(s) > 3]
        # remove 2., 3. from queries
        queries = [re.sub(r"^\d+\.\s*", "", q) for q in queries]
        return queries

    def query_gen(
        self,
        query: str,
        k: int = 10,
        max_sources: int = 5,
        length_prompt: str = "about 100 words",
        marginal_relevance: bool = True,
    ):
        yield from self._query(
            query,
            k=k,
            max_sources=max_sources,
            length_prompt=length_prompt,
            marginal_relevance=marginal_relevance,
        )

    def query(
        self,
        query: str,
        k: int = 10,
        max_sources: int = 5,
        length_prompt: str = "about 100 words",
        marginal_relevance: bool = True,
    ):
        for answer in self._query(
            query,
            k=k,
            max_sources=max_sources,
            length_prompt=length_prompt,
            marginal_relevance=marginal_relevance,
        ):
            pass
        return answer

    def _query(
        self,
        query: str,
        k: int,
        max_sources: int,
        length_prompt: str,
        marginal_relevance: bool,
    ):
        if k < max_sources:
            raise ValueError("k should be greater than max_sources")
        tokens = 0
        answer = Answer(query)
        with get_openai_callback() as cb:
            for answer in self.get_evidence(
                answer,
                k=k,
                max_sources=max_sources,
                marginal_relevance=marginal_relevance,
            ):
                yield answer
            tokens += cb.total_tokens
        context_str, citations = answer.context, answer.contexts
        bib = dict()
        passages = dict()
        if len(context_str) < 10:
            answer_text = (
                "I cannot answer this question due to insufficient information."
            )
        else:
            with get_openai_callback() as cb:
                answer_text = self.qa_chain.run(
                    question=query, context_str=context_str, length=length_prompt
                )
                tokens += cb.total_tokens
        # it still happens lol
        if "(Foo2012)" in answer_text:
            answer_text = answer_text.replace("(Foo2012)", "")
        for key, citation, summary, text in citations:
            # do check for whole key (so we don't catch Callahan2019a with Callahan2019)
            skey = key.split(" ")[0]
            if skey + " " in answer_text or skey + ")" in answer_text:
                bib[skey] = citation
                passages[key] = text
        bib_str = "\n\n".join(
            [f"{i+1}. ({k}): {c}" for i, (k, c) in enumerate(bib.items())]
        )
        formatted_answer = f"Question: {query}\n\n{answer_text}\n"
        if len(bib) > 0:
            formatted_answer += f"\nReferences\n\n{bib_str}\n"
        formatted_answer += f"\nTokens Used: {tokens} Cost: ${tokens/1000 * 0.02:.2f}"
        answer.answer = answer_text
        answer.formatted_answer = formatted_answer
        answer.references = bib_str
        answer.passages = passages
        answer.tokens = tokens
        yield answer
