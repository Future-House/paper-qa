import asyncio
import os
import re
import sys
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Set, Union, cast

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import FAISS

try:
    from pydantic.v1 import BaseModel, validator
except ImportError:
    from pydantic import BaseModel, validator

from .chains import get_score, make_chain
from .paths import PAPERQA_DIR
from .readers import read_doc
from .types import Answer, CallbackFactory, Context, Doc, DocKey, PromptCollection, Text
from .utils import (
    gather_with_concurrency,
    get_llm_name,
    guess_is_4xx,
    maybe_is_html,
    maybe_is_pdf,
    maybe_is_text,
    md5sum,
    name_in_text,
    strip_citations,
)


class Docs(BaseModel, arbitrary_types_allowed=True, smart_union=True):
    """A collection of documents to be used for answering questions."""

    docs: Dict[DocKey, Doc] = {}
    texts: List[Text] = []
    docnames: Set[str] = set()
    texts_index: Optional[VectorStore] = None
    doc_index: Optional[VectorStore] = None
    llm: Union[str, BaseLanguageModel] = ChatOpenAI(
        temperature=0.1, model="gpt-3.5-turbo", client=None
    )
    summary_llm: Optional[Union[str, BaseLanguageModel]] = None
    name: str = "default"
    index_path: Optional[Path] = PAPERQA_DIR / name
    embeddings: Embeddings = OpenAIEmbeddings(client=None)
    max_concurrent: int = 5
    deleted_dockeys: Set[DocKey] = set()
    prompts: PromptCollection = PromptCollection()
    memory: bool = False
    memory_model: Optional[BaseChatMemory] = None
    jit_texts_index: bool = False
    # This is used to strip indirect citations that come up from the summary llm
    strip_citations: bool = True

    # TODO: Not sure how to get this to work
    # while also passing mypy checks
    @validator("llm", "summary_llm")
    def check_llm(cls, v: Union[BaseLanguageModel, str]) -> BaseLanguageModel:
        if type(v) is str:
            return ChatOpenAI(temperature=0.1, model=v, client=None)
        return cast(BaseLanguageModel, v)

    @validator("summary_llm", always=True)
    def copy_llm_if_not_set(cls, v, values):
        return v or values["llm"]

    @validator("memory_model", always=True)
    def check_memory_model(cls, v, values):
        if values["memory"]:
            if v is None:
                return ConversationTokenBufferMemory(
                    llm=values["summary_llm"],
                    max_token_limit=512,
                    memory_key="memory",
                    human_prefix="Question",
                    ai_prefix="Answer",
                    input_key="Question",
                    output_key="Answer",
                )
            if v.memory_variables()[0] != "memory":
                raise ValueError("Memory model must have memory_variables=['memory']")
            return values["memory_model"]
        return None

    def clear_docs(self):
        self.texts = []
        self.docs = {}
        self.docnames = set()

    def update_llm(
        self,
        llm: Union[BaseLanguageModel, str],
        summary_llm: Optional[Union[BaseLanguageModel, str]] = None,
    ) -> None:
        """Update the LLM for answering questions."""
        if type(llm) is str:
            llm = ChatOpenAI(temperature=0.1, model=llm, client=None)
        if type(summary_llm) is str:
            summary_llm = ChatOpenAI(temperature=0.1, model=summary_llm, client=None)
        self.llm = cast(BaseLanguageModel, llm)
        if summary_llm is None:
            summary_llm = llm
        self.summary_llm = cast(BaseLanguageModel, summary_llm)

    def _get_unique_name(self, docname: str) -> str:
        """Create a unique name given proposed name"""
        suffix = ""
        while docname + suffix in self.docnames:
            # move suffix to next letter
            if suffix == "":
                suffix = "a"
            else:
                suffix = chr(ord(suffix) + 1)
        docname += suffix
        return docname

    def add_file(
        self,
        file: BinaryIO,
        citation: Optional[str] = None,
        docname: Optional[str] = None,
        dockey: Optional[DocKey] = None,
        chunk_chars: int = 3000,
    ) -> Optional[str]:
        """Add a document to the collection."""
        # just put in temp file and use existing method
        suffix = ".txt"
        if maybe_is_pdf(file):
            suffix = ".pdf"
        elif maybe_is_html(file):
            suffix = ".html"

        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(file.read())
            f.seek(0)
            return self.add(
                Path(f.name),
                citation=citation,
                docname=docname,
                dockey=dockey,
                chunk_chars=chunk_chars,
            )

    def add_url(
        self,
        url: str,
        citation: Optional[str] = None,
        docname: Optional[str] = None,
        dockey: Optional[DocKey] = None,
        chunk_chars: int = 3000,
    ) -> Optional[str]:
        """Add a document to the collection."""
        import urllib.request

        with urllib.request.urlopen(url) as f:
            # need to wrap to enable seek
            file = BytesIO(f.read())
            return self.add_file(
                file,
                citation=citation,
                docname=docname,
                dockey=dockey,
                chunk_chars=chunk_chars,
            )

    def add(
        self,
        path: Path,
        citation: Optional[str] = None,
        docname: Optional[str] = None,
        disable_check: bool = False,
        dockey: Optional[DocKey] = None,
        chunk_chars: int = 3000,
    ) -> Optional[str]:
        """Add a document to the collection."""
        if dockey is None:
            dockey = md5sum(path)
        if citation is None:
            # skip system because it's too hesitant to answer
            cite_chain = make_chain(
                prompt=self.prompts.cite,
                llm=cast(BaseLanguageModel, self.summary_llm),
                skip_system=True,
            )
            # peak first chunk
            fake_doc = Doc(docname="", citation="", dockey=dockey)
            texts = read_doc(path, fake_doc, chunk_chars=chunk_chars, overlap=100)
            if len(texts) == 0:
                raise ValueError(f"Could not read document {path}. Is it empty?")
            citation = cite_chain.run(texts[0].text)
            if len(citation) < 3 or "Unknown" in citation or "insufficient" in citation:
                citation = f"Unknown, {os.path.basename(path)}, {datetime.now().year}"

        if docname is None:
            # get first name and year from citation
            match = re.search(r"([A-Z][a-z]+)", citation)
            if match is not None:
                author = match.group(1)  # type: ignore
            else:
                # panicking - no word??
                raise ValueError(
                    f"Could not parse docname from citation {citation}. "
                    "Consider just passing key explicitly - e.g. docs.py "
                    "(path, citation, key='mykey')"
                )
            year = ""
            match = re.search(r"(\d{4})", citation)
            if match is not None:
                year = match.group(1)  # type: ignore
            docname = f"{author}{year}"
        docname = self._get_unique_name(docname)
        doc = Doc(docname=docname, citation=citation, dockey=dockey)
        texts = read_doc(path, doc, chunk_chars=chunk_chars, overlap=100)
        # loose check to see if document was loaded
        if (
            len(texts) == 0
            or len(texts[0].text) < 10
            or (not disable_check and not maybe_is_text(texts[0].text))
        ):
            raise ValueError(
                f"This does not look like a text document: {path}. Path disable_check to ignore this error."
            )
        if self.add_texts(texts, doc):
            return docname
        return None

    def add_texts(
        self,
        texts: List[Text],
        doc: Doc,
    ) -> bool:
        """Add chunked texts to the collection. This is useful if you have already chunked the texts yourself.

        Returns True if the document was added, False if it was already in the collection.
        """
        if doc.dockey in self.docs:
            return False
        if len(texts) == 0:
            raise ValueError("No texts to add.")
        if doc.docname in self.docnames:
            new_docname = self._get_unique_name(doc.docname)
            for t in texts:
                t.name = t.name.replace(doc.docname, new_docname)
            doc.docname = new_docname
        if texts[0].embeddings is None:
            text_embeddings = self.embeddings.embed_documents([t.text for t in texts])
            for i, t in enumerate(texts):
                t.embeddings = text_embeddings[i]
        else:
            text_embeddings = cast(List[List[float]], [t.embeddings for t in texts])
        if self.texts_index is not None:
            try:
                # TODO: Simplify - super weird
                vec_store_text_and_embeddings = list(
                    map(lambda x: (x.text, x.embeddings), texts)
                )
                self.texts_index.add_embeddings(  # type: ignore
                    vec_store_text_and_embeddings,
                    metadatas=[t.dict(exclude={"embeddings", "text"}) for t in texts],
                )
            except AttributeError:
                raise ValueError("Need a vector store that supports adding embeddings.")
        if self.doc_index is not None:
            self.doc_index.add_texts([doc.citation], metadatas=[doc.dict()])
        self.docs[doc.dockey] = doc
        self.texts += texts
        self.docnames.add(doc.docname)
        return True

    def delete(
        self, name: Optional[str] = None, dockey: Optional[DocKey] = None
    ) -> None:
        """Delete a document from the collection."""
        if name is not None:
            doc = next((doc for doc in self.docs.values() if doc.docname == name), None)
            if doc is None:
                return
            self.docnames.remove(doc.docname)
            dockey = doc.dockey
        del self.docs[dockey]
        self.deleted_dockeys.add(dockey)

    async def adoc_match(
        self,
        query: str,
        k: int = 25,
        rerank: Optional[bool] = None,
        get_callbacks: CallbackFactory = lambda x: None,
    ) -> Set[DocKey]:
        """Return a list of dockeys that match the query."""
        if self.doc_index is None:
            if len(self.docs) == 0:
                return set()
            texts = [doc.citation for doc in self.docs.values()]
            metadatas = [d.dict() for d in self.docs.values()]
            self.doc_index = FAISS.from_texts(
                texts, metadatas=metadatas, embedding=self.embeddings
            )
        matches = self.doc_index.max_marginal_relevance_search(
            query, k=k + len(self.deleted_dockeys)
        )
        # filter the matches
        matches = [
            m for m in matches if m.metadata["dockey"] not in self.deleted_dockeys
        ]
        try:
            # for backwards compatibility (old pickled objects)
            matched_docs = [self.docs[m.metadata["dockey"]] for m in matches]
        except KeyError:
            matched_docs = [Doc(**m.metadata) for m in matches]
        if len(matched_docs) == 0:
            return set()
        # this only works for gpt-4 (in my testing)
        try:
            if (
                rerank is None
                and get_llm_name(cast(BaseLanguageModel, self.llm)).startswith("gpt-4")
                or rerank is True
            ):
                chain = make_chain(
                    self.prompts.select,
                    cast(BaseLanguageModel, self.llm),
                    skip_system=True,
                )
                papers = [f"{d.docname}: {d.citation}" for d in matched_docs]
                result = await chain.arun(  # type: ignore
                    question=query,
                    papers="\n".join(papers),
                    callbacks=get_callbacks("filter"),
                )
                return set([d.dockey for d in matched_docs if d.docname in result])
        except AttributeError:
            pass
        return set([d.dockey for d in matched_docs])

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.texts_index is not None and self.index_path is not None:
            state["texts_index"].save_local(self.index_path)
        del state["texts_index"]
        del state["doc_index"]
        return {"__dict__": state, "__fields_set__": self.__fields_set__}

    def __setstate__(self, state):
        object.__setattr__(self, "__dict__", state["__dict__"])
        object.__setattr__(self, "__fields_set__", state["__fields_set__"])
        try:
            self.texts_index = FAISS.load_local(self.index_path, self.embeddings)
        except Exception:
            # they use some special exception type, but I don't want to import it
            self.texts_index = None
        self.doc_index = None

    def _build_texts_index(self, keys: Optional[Set[DocKey]] = None):
        if keys is not None and self.jit_texts_index:
            del self.texts_index
            self.texts_index = None
        if self.texts_index is None:
            texts = self.texts
            if keys is not None:
                texts = [t for t in texts if t.doc.dockey in keys]
            if len(texts) == 0:
                return
            raw_texts = [t.text for t in texts]
            text_embeddings = [t.embeddings for t in texts]
            metadatas = [t.dict(exclude={"embeddings", "text"}) for t in texts]
            self.texts_index = FAISS.from_embeddings(
                # wow adding list to the zip was tricky
                text_embeddings=list(zip(raw_texts, text_embeddings)),
                embedding=self.embeddings,
                metadatas=metadatas,
            )

    def clear_memory(self):
        """Clear the memory of the model."""
        if self.memory_model is not None:
            self.memory_model.clear()

    def get_evidence(
        self,
        answer: Answer,
        k: int = 10,
        max_sources: int = 5,
        marginal_relevance: bool = True,
        get_callbacks: CallbackFactory = lambda x: None,
        detailed_citations: bool = False,
        disable_vector_search: bool = False,
        disable_summarization: bool = False,
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
                get_callbacks=get_callbacks,
                detailed_citations=detailed_citations,
                disable_vector_search=disable_vector_search,
                disable_summarization=disable_summarization,
            )
        )

    async def aget_evidence(
        self,
        answer: Answer,
        k: int = 10,  # Number of vectors to retrieve
        max_sources: int = 5,  # Number of scored contexts to use
        marginal_relevance: bool = True,
        get_callbacks: CallbackFactory = lambda x: None,
        detailed_citations: bool = False,
        disable_vector_search: bool = False,
        disable_summarization: bool = False,
    ) -> Answer:
        if disable_vector_search:
            k = k * 10000
        if len(self.docs) == 0 and self.doc_index is None:
            return answer
        self._build_texts_index(keys=answer.dockey_filter)
        if self.texts_index is None:
            return answer
        self.texts_index = cast(VectorStore, self.texts_index)
        _k = k
        if answer.dockey_filter is not None:
            _k = k * 10  # heuristic
        if marginal_relevance:
            matches = self.texts_index.max_marginal_relevance_search(
                answer.question, k=_k, fetch_k=5 * _k
            )
        else:
            matches = self.texts_index.similarity_search(
                answer.question, k=_k, fetch_k=5 * _k
            )
        # ok now filter
        if answer.dockey_filter is not None:
            matches = [
                m
                for m in matches
                if m.metadata["doc"]["dockey"] in answer.dockey_filter
            ]

        # check if it is deleted
        matches = [
            m
            for m in matches
            if m.metadata["doc"]["dockey"] not in self.deleted_dockeys
        ]

        # check if it is already in answer
        cur_names = [c.text.name for c in answer.contexts]
        matches = [m for m in matches if m.metadata["name"] not in cur_names]

        # now finally cut down
        matches = matches[:k]

        async def process(match):
            callbacks = get_callbacks("evidence:" + match.metadata["name"])
            summary_chain = make_chain(
                self.prompts.summary,
                self.summary_llm,
                memory=self.memory_model,
                system_prompt=self.prompts.system,
            )
            # This is dangerous because it
            # could mask errors that are important- like auth errors
            # I also cannot know what the exception
            # type is because any model could be used
            # my best idea is see if there is a 4XX
            # http code in the exception
            try:
                citation = match.metadata["doc"]["citation"]
                if detailed_citations:
                    citation = match.metadata["name"] + ": " + citation
                if self.prompts.skip_summary:
                    context = match.page_content
                else:
                    context = await summary_chain.arun(
                        question=answer.question,
                        # Add name so chunk is stated
                        citation=citation,
                        summary_length=answer.summary_length,
                        text=match.page_content,
                        callbacks=callbacks,
                    )
            except Exception as e:
                if guess_is_4xx(str(e)):
                    return None
                raise e
            if "not applicable" in context.lower() or "not relevant" in context.lower():
                return None
            if self.strip_citations:
                # remove citations that collide with our grounded citations (for the answer LLM)
                context = strip_citations(context)
            c = Context(
                context=context,
                text=Text(
                    text=match.page_content,
                    name=match.metadata["name"],
                    doc=Doc(**match.metadata["doc"]),
                ),
                score=get_score(context),
            )
            return c

        if disable_summarization:
            contexts = [
                Context(
                    context=match.page_content,
                    score=10,
                    text=Text(
                        text=match.page_content,
                        name=match.metadata["name"],
                        doc=Doc(**match.metadata["doc"]),
                    ),
                )
                for match in matches
            ]

        else:
            results = await gather_with_concurrency(
                self.max_concurrent, *[process(m) for m in matches]
            )
            # filter out failures
            contexts = [c for c in results if c is not None]

        answer.contexts = sorted(
            contexts + answer.contexts, key=lambda x: x.score, reverse=True
        )
        answer.contexts = answer.contexts[:max_sources]
        context_str = "\n\n".join(
            [
                f"{c.text.name}: {c.context}"
                + (f"\n\n Based on {c.text.doc.citation}" if detailed_citations else "")
                for c in answer.contexts
            ]
        )

        valid_names = [c.text.name for c in answer.contexts]
        context_str += "\n\nValid keys: " + ", ".join(valid_names)
        answer.context = context_str
        return answer

    def query(
        self,
        query: str,
        k: int = 10,
        max_sources: int = 5,
        length_prompt="about 100 words",
        marginal_relevance: bool = True,
        answer: Optional[Answer] = None,
        key_filter: Optional[bool] = None,
        get_callbacks: CallbackFactory = lambda x: None,
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
        get_callbacks: CallbackFactory = lambda x: None,
    ) -> Answer:
        if k < max_sources:
            raise ValueError("k should be greater than max_sources")
        if answer is None:
            answer = Answer(question=query, answer_length=length_prompt)
        if len(answer.contexts) == 0:
            # this is heuristic - k and len(docs) are not
            # comparable - one is chunks and one is docs
            if key_filter or (key_filter is None and len(self.docs) > k):
                keys = await self.adoc_match(
                    answer.question, get_callbacks=get_callbacks
                )
                if len(keys) > 0:
                    answer.dockey_filter = keys
            answer = await self.aget_evidence(
                answer,
                k=k,
                max_sources=max_sources,
                marginal_relevance=marginal_relevance,
                get_callbacks=get_callbacks,
            )
        if self.prompts.pre is not None:
            chain = make_chain(
                self.prompts.pre,
                cast(BaseLanguageModel, self.llm),
                memory=self.memory_model,
                system_prompt=self.prompts.system,
            )
            pre = await chain.arun(
                question=answer.question, callbacks=get_callbacks("pre")
            )
            answer.context = answer.context + "\n\nExtra background information:" + pre
        bib = dict()
        if len(answer.context) < 10 and not self.memory:
            answer_text = (
                "I cannot answer this question due to insufficient information."
            )
        else:
            callbacks = get_callbacks("answer")
            qa_chain = make_chain(
                self.prompts.qa,
                cast(BaseLanguageModel, self.llm),
                memory=self.memory_model,
                system_prompt=self.prompts.system,
            )
            answer_text = await qa_chain.arun(
                context=answer.context,
                answer_length=answer.answer_length,
                question=answer.question,
                callbacks=callbacks,
                verbose=True,
            )
        # it still happens
        if "(Example2012)" in answer_text:
            answer_text = answer_text.replace("(Example2012)", "")
        for c in answer.contexts:
            name = c.text.name
            citation = c.text.doc.citation
            # do check for whole key (so we don't catch Callahan2019a with Callahan2019)
            if name_in_text(name, answer_text):
                bib[name] = citation
        bib_str = "\n\n".join(
            [f"{i+1}. ({k}): {c}" for i, (k, c) in enumerate(bib.items())]
        )
        formatted_answer = f"Question: {answer.question}\n\n{answer_text}\n"
        if len(bib) > 0:
            formatted_answer += f"\nReferences\n\n{bib_str}\n"
        answer.answer = answer_text
        answer.formatted_answer = formatted_answer
        answer.references = bib_str

        if self.prompts.post is not None:
            chain = make_chain(
                self.prompts.post,
                cast(BaseLanguageModel, self.llm),
                memory=self.memory_model,
                system_prompt=self.prompts.system,
            )
            post = await chain.arun(**answer.dict(), callbacks=get_callbacks("post"))
            answer.answer = post
            answer.formatted_answer = f"Question: {answer.question}\n\n{post}\n"
            if len(bib) > 0:
                answer.formatted_answer += f"\nReferences\n\n{bib_str}\n"
        if self.memory_model is not None:
            answer.memory = self.memory_model.load_memory_variables(inputs={})["memory"]
            self.memory_model.save_context(
                {"Question": answer.question}, {"Answer": answer.answer}
            )

        return answer
