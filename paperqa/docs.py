import asyncio
import os
import re
import sys
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import langchain
from langchain.cache import SQLiteCache
from langchain.callbacks import OpenAICallbackHandler, get_openai_callback
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.base import LLM
from langchain.vectorstores import FAISS

from .paths import CACHE_PATH
from .qaprompts import (
    citation_prompt,
    make_chain,
    qa_prompt,
    search_prompt,
    select_paper_prompt,
    summary_prompt,
    get_score,
)
from .readers import read_doc
from .types import Answer, Context
from .utils import maybe_is_text, md5sum, gather_with_concurrency, guess_is_4xx

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
        max_concurrent: int = 5,
    ) -> None:
        """Initialize the collection of documents.

        Args:
            chunk_size_limit: The maximum number of characters to use for a single chunk of text.
            llm: The language model to use for answering questions. Default - OpenAI chat-gpt-turbo
            summary_llm: The language model to use for summarizing documents. If None, llm is used.
            name: The name of the collection.
            index_path: The path to the index file IF pickled. If None, defaults to using name in $HOME/.paperqa/name
            embeddings: The embeddings to use for indexing documents. Default - OpenAI embeddings
            max_concurrent: Number of concurrent LLM model calls to make
        """
        self.docs = []
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
        self.max_concurrent = max_concurrent
        self._deleted_keys = set()

    def update_llm(
        self,
        llm: Optional[Union[LLM, str]] = None,
        summary_llm: Optional[Union[LLM, str]] = None,
    ) -> None:
        """Update the LLM for answering questions."""
        if llm is None and os.environ.get("OPENAI_API_KEY") is not None:
            llm = "gpt-3.5-turbo"
        if type(llm) is str:
            llm = ChatOpenAI(temperature=0.1, model_name=llm)
        if type(summary_llm) is str:
            summary_llm = ChatOpenAI(temperature=0.1, model_name=summary_llm)
        self.llm = llm
        if summary_llm is None:
            summary_llm = llm
        self.summary_llm = summary_llm

    def get_unique_key(self, key: str) -> str:
        """Create a unique key given proposed key"""
        suffix = ""
        while key + suffix in self.keys:
            # move suffix to next letter
            if suffix == "":
                suffix = "a"
            else:
                suffix = chr(ord(suffix) + 1)
        key += suffix
        return key

    def add(
        self,
        path: str,
        citation: Optional[str] = None,
        key: Optional[str] = None,
        disable_check: bool = False,
        chunk_chars: Optional[int] = 3000,
    ) -> str:
        """Add a document to the collection."""

        # first check to see if we already have this document
        # this way we don't make api call to create citation on file we already have
        hash = md5sum(path)
        if hash in [d["hash"] for d in self.docs]:
            raise ValueError(f"Document {path} already in collection.")

        if citation is None:
            # skip system because it's too hesitant to answer
            cite_chain = make_chain(
                prompt=citation_prompt, llm=self.summary_llm, skip_system=True
            )
            # peak first chunk
            texts, _ = read_doc(path, "", "", chunk_chars=chunk_chars)
            if len(texts) == 0:
                raise ValueError(f"Could not read document {path}. Is it empty?")
            citation = cite_chain.run(texts[0])
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
        key = self.get_unique_key(key)
        texts, metadata = read_doc(path, citation, key, chunk_chars=chunk_chars)
        # loose check to see if document was loaded
        #
        if len("".join(texts)) < 10 or (
            not disable_check and not maybe_is_text("".join(texts))
        ):
            raise ValueError(
                f"This does not look like a text document: {path}. Path disable_check to ignore this error."
            )
        self.add_texts(texts, metadata, hash)
        return key

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[dict],
        hash: str,
        text_embeddings: Optional[List[List[float]]] = None,
    ):
        """Add chunked texts to the collection. This is useful if you have already chunked the texts yourself.

        The metadatas should have the following keys: citation, dockey (same as key arg), and key (unique key for each chunk).
        The hash is a unique identifier for the document. It is used to check if the document has already been added.
        """
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have the same length.")
        key = metadatas[0]["dockey"]
        citation = metadatas[0]["citation"]
        if key in self.keys:
            new_key = self.get_unique_key(key)
            for metadata in metadatas:
                metadata["dockey"] = new_key
                metadata["key"] = metadata["key"].replace(key, new_key)
            key = new_key
        if text_embeddings is None:
            text_embeddings = self.embeddings.embed_documents(texts)
        if self._faiss_index is not None:
            self._faiss_index.add_embeddings(
                list(zip(texts, text_embeddings)), metadatas=metadatas
            )
        elif self._doc_index is not None:
            self._doc_index.add_texts([citation], metadatas=[{"key": key}])
        self.docs.append(
            dict(
                texts=texts,
                metadata=metadatas,
                key=key,
                hash=hash,
                text_embeddings=text_embeddings,
            )
        )
        self.keys.add(key)

    def delete(self, key: str) -> None:
        """Delete a document from the collection."""
        if key not in self.keys:
            return
        self.keys.remove(key)
        self.docs = [doc for doc in self.docs if doc["key"] != key]
        self._deleted_keys.add(key)

    def clear(self) -> None:
        """Clear the collection of documents."""
        self.docs = []
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
                doc["hash"],
            )
            for doc in self.docs
        ]

    async def adoc_match(
        self, query: str, k: int = 25, callbacks: List[AsyncCallbackHandler] = []
    ) -> List[str]:
        """Return a list of documents that match the query."""
        if len(self.docs) == 0:
            return ""
        if self._doc_index is None:
            texts = [doc["metadata"][0]["citation"] for doc in self.docs]
            metadatas = [{"key": doc["metadata"][0]["dockey"]} for doc in self.docs]
            self._doc_index = FAISS.from_texts(
                texts, metadatas=metadatas, embedding=self.embeddings
            )
        docs = self._doc_index.max_marginal_relevance_search(
            query, k=k + len(self._deleted_keys)
        )
        docs = [doc for doc in docs if doc.metadata["key"] not in self._deleted_keys]
        chain = make_chain(select_paper_prompt, self.summary_llm)
        papers = [f"{d.metadata['key']}: {d.page_content}" for d in docs]
        result = await chain.arun(
            question=query, papers="\n".join(papers), callbacks=callbacks
        )
        return result

    def doc_match(
        self, query: str, k: int = 25, callbacks: List[AsyncCallbackHandler] = []
    ) -> List[str]:
        """Return a list of documents that match the query."""
        if len(self.docs) == 0:
            return ""
        if self._doc_index is None:
            texts = [doc["metadata"][0]["citation"] for doc in self.docs]
            metadatas = [{"key": doc["metadata"][0]["dockey"]} for doc in self.docs]
            self._doc_index = FAISS.from_texts(
                texts, metadatas=metadatas, embedding=self.embeddings
            )
        docs = self._doc_index.max_marginal_relevance_search(
            query, k=k + len(self._deleted_keys)
        )
        docs = [doc for doc in docs if doc.metadata["key"] not in self._deleted_keys]
        chain = make_chain(select_paper_prompt, self.summary_llm)
        papers = [f"{d.metadata['key']}: {d.page_content}" for d in docs]
        result = chain.run(
            question=query, papers="\n".join(papers), callbacks=callbacks
        )
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._faiss_index is not None:
            state["_faiss_index"].save_local(self.index_path)
        del state["_faiss_index"]
        del state["_doc_index"]
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
        # must be a better way to have backwards compatibility
        if not hasattr(self, "_deleted_keys"):
            self._deleted_keys = set()
        if not hasattr(self, "max_concurrent"):
            self.max_concurrent = 5
        self.update_llm(None, None)

    def _build_faiss_index(self):
        if self._faiss_index is None:
            texts = reduce(lambda x, y: x + y, [doc["texts"] for doc in self.docs], [])
            text_embeddings = reduce(
                lambda x, y: x + y, [doc["text_embeddings"] for doc in self.docs], []
            )
            metadatas = reduce(
                lambda x, y: x + y, [doc["metadata"] for doc in self.docs], []
            )
            self._faiss_index = FAISS.from_embeddings(
                # wow adding list to the zip was tricky
                text_embeddings=list(zip(texts, text_embeddings)),
                embedding=self.embeddings,
                metadatas=metadatas,
            )

    def get_evidence(
        self,
        answer: Answer,
        k: int = 3,
        max_sources: int = 5,
        marginal_relevance: bool = True,
        key_filter: Optional[List[str]] = None,
        get_callbacks: Callable[[str], AsyncCallbackHandler] = lambda x: [],
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
                get_callbacks=get_callbacks,
            )
        )

    async def aget_evidence(
        self,
        answer: Answer,
        k: int = 3,
        max_sources: int = 5,
        marginal_relevance: bool = True,
        key_filter: Optional[List[str]] = None,
        get_callbacks: Callable[[str], AsyncCallbackHandler] = lambda x: [],
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
        # ok now filter
        if key_filter is not None:
            docs = [doc for doc in docs if doc.metadata["dockey"] in key_filter][:k]

        async def process(doc):
            if doc.metadata["dockey"] in self._deleted_keys:
                return None, None
            # check if it is already in answer (possible in agent setting)
            if doc.metadata["key"] in [c.key for c in answer.contexts]:
                return None, None
            callbacks = [OpenAICallbackHandler()] + get_callbacks(
                "evidence:" + doc.metadata["key"]
            )
            summary_chain = make_chain(summary_prompt, self.summary_llm)
            # This is dangerous because it
            # could mask errors that are important
            # I also cannot know what the exception
            # type is because any model could be used
            # my best idea is see if there is a 4XX
            # http code in the exception
            try:
                context = await summary_chain.arun(
                    question=answer.question,
                    context_str=doc.page_content,
                    citation=doc.metadata["citation"],
                    callbacks=callbacks,
                )
            except Exception as e:
                if guess_is_4xx(e):
                    return None, None
                raise e
            c = Context(
                key=doc.metadata["key"],
                citation=doc.metadata["citation"],
                context=context,
                text=doc.page_content,
                score=get_score(context),
            )
            if "not applicable" not in c.context.casefold():
                return c, callbacks[0]
            return None, None

        results = await gather_with_concurrency(
            self.max_concurrent, *[process(doc) for doc in docs]
        )
        # filter out failures
        results = [r for r in results if r[0] is not None]
        answer.tokens += sum([cb.total_tokens for _, cb in results])
        answer.cost += sum([cb.total_cost for _, cb in results])
        contexts = [c for c, _ in results if c is not None]
        if len(contexts) == 0:
            return answer
        contexts = sorted(contexts, key=lambda x: x.score, reverse=True)
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

        search_chain = make_chain(prompt=search_prompt, llm=self.summary_llm)
        search_query = search_chain.run(question=query)
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
        get_callbacks: Callable[[str], AsyncCallbackHandler] = lambda x: [],
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
                get_callbacks=get_callbacks,
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
        get_callbacks: Callable[[str], AsyncCallbackHandler] = lambda x: [],
    ) -> Answer:
        if k < max_sources:
            raise ValueError("k should be greater than max_sources")
        if answer is None:
            answer = Answer(query)
        if len(answer.contexts) == 0:
            if key_filter or (key_filter is None and len(self.docs) > max_sources):
                callbacks = [OpenAICallbackHandler()] + get_callbacks("filter")
                keys = await self.adoc_match(answer.question, callbacks=callbacks)
                answer.tokens += callbacks[0].total_tokens
                answer.cost += callbacks[0].total_cost
                key_filter = True if len(keys) > 0 else False
            answer = await self.aget_evidence(
                answer,
                k=k,
                max_sources=max_sources,
                marginal_relevance=marginal_relevance,
                key_filter=keys if key_filter else None,
                get_callbacks=get_callbacks,
            )
        context_str, contexts = answer.context, answer.contexts
        bib = dict()
        passages = dict()
        if len(context_str) < 10:
            answer_text = (
                "I cannot answer this question due to insufficient information."
            )
        else:
            cb = OpenAICallbackHandler()
            callbacks = [cb] + get_callbacks("answer")
            qa_chain = make_chain(qa_prompt, self.llm)
            answer_text = await qa_chain.arun(
                question=query,
                context_str=context_str,
                length=length_prompt,
                callbacks=callbacks,
            )
            answer.tokens += cb.total_tokens
            answer.cost += cb.total_cost
        # it still happens lol
        if "(Example2012)" in answer_text:
            answer_text = answer_text.replace("(Example2012)", "")
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
