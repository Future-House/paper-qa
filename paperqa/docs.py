from typing import List, Optional, Tuple, Union
from functools import reduce
import os
import sys
import asyncio
from pathlib import Path
import re
from .paths import CACHE_PATH
from .utils import maybe_is_text, md5sum
from .qaprompts import (
    summary_prompt,
    qa_prompt,
    search_prompt,
    citation_prompt,
    select_paper_prompt,
    make_chain,
)
from .types import Answer, Context
from .readers import read_doc
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import LLM
from langchain.callbacks import get_openai_callback
from langchain.cache import SQLiteCache
import langchain
from datetime import datetime

os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
langchain.llm_cache = SQLiteCache(CACHE_PATH)


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
            index_path = Path.home() / ".paperqa" / name
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
        self.summary_chain = make_chain(prompt=summary_prompt, llm=summary_llm)
        self.qa_chain = make_chain(prompt=qa_prompt, llm=llm)
        self.search_chain = make_chain(prompt=search_prompt, llm=summary_llm)
        self.cite_chain = make_chain(prompt=citation_prompt, llm=summary_llm)

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

        if citation is None:
            # peak first chunk
            texts, _ = read_doc(path, "", "", chunk_chars=chunk_chars)
            with get_openai_callback():
                citation = self.cite_chain.run(texts[0])
            if len(citation) < 3 or "Unknown" in citation or "insufficient" in citation:
                citation = f"Unknown, {os.path.basename(path)}, {datetime.now().year}"

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

        texts, metadata = read_doc(path, citation, key, chunk_chars=chunk_chars)
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
        docs = self._doc_index.max_marginal_relevance_search(query, k=k)
        chain = make_chain(select_paper_prompt, self.summary_llm)
        papers = [f"{d.metadata['key']}: {d.page_content}" for d in docs]
        result = chain.run(instructions=query, papers="\n".join(papers))
        return result

    # to pickle, we have to save the index as a file

    def __getstate__(self):
        if self._faiss_index is None and len(self.docs) > 0:
            self._build_faiss_index()
        state = self.__dict__.copy()
        if self._faiss_index is not None:
            state["_faiss_index"].save_local(self.index_path)
        del state["_faiss_index"]
        del state["_doc_index"]
        # remove LLMs (they can have callbacks, which can't be pickled)
        del state["summary_chain"]
        del state["qa_chain"]
        del state["cite_chain"]
        del state["search_chain"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            self._faiss_index = FAISS.load_local(self.index_path, self.embeddings)
        except:
            # they use some special exception type, but I don't want to import it
            self._faiss_index = None
        if not hasattr(self, "_doc_index"):
            self._doc_index = None
        self.update_llm(None, None)

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
        marginal_relevance: bool = True,
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
                marginal_relevance=marginal_relevance,
                key_filter=key_filter,
            )
        )

    async def aget_evidence(
        self,
        answer: Answer,
        k: int = 3,
        max_sources: int = 5,
        marginal_relevance: bool = True,
        key_filter: Optional[List[str]] = None,
    ) -> Answer:
        if len(self.docs) == 0:
            return answer
        if self._faiss_index is None:
            self._build_faiss_index()
        _k = k
        if key_filter is not None:
            _k = k * 10  # heuristic
        # want to work through indices but less k
        if marginal_relevance:
            docs = self._faiss_index.max_marginal_relevance_search(
                answer.question, k=_k, fetch_k=5 * _k
            )
        else:
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

    def query(
        self,
        query: str,
        k: int = 10,
        max_sources: int = 5,
        length_prompt: str = "about 100 words",
        marginal_relevance: bool = True,
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
                marginal_relevance=marginal_relevance,
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
        marginal_relevance: bool = True,
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
                marginal_relevance=marginal_relevance,
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
        bib_str = "\n\n".join(
            [f"{i+1}. ({k}): {c}" for i, (k, c) in enumerate(bib.items())]
        )
        formatted_answer = f"Question: {query}\n\n{answer_text}\n"
        if len(bib) > 0:
            formatted_answer += f"\nReferences\n\n{bib_str}\n"
        formatted_answer += f"\nTokens Used: {answer.tokens} Cost: ${answer.cost:.2f}"
        answer.answer = answer_text
        answer.formatted_answer = formatted_answer
        answer.references = bib_str
        answer.passages = passages
        return answer
