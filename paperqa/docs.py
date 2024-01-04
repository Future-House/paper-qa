import asyncio
import os
import re
import sys
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator

from .chains import get_score, make_chain
from .paths import PAPERQA_DIR
from .readers import read_doc
from .types import (
    Answer,
    CallbackFactory,
    Context,
    Doc,
    DocKey,
    NumpyVectorStore,
    PromptCollection,
    Text,
    VectorStore,
)
from .utils import (
    gather_with_concurrency,
    guess_is_4xx,
    maybe_is_html,
    maybe_is_pdf,
    maybe_is_text,
    md5sum,
    name_in_text,
    strip_citations,
)


class Docs(BaseModel):
    """A collection of documents to be used for answering questions."""

    _client: AsyncOpenAI
    docs: dict[DocKey, Doc] = {}
    texts: list[Text] = []
    docnames: set[str] = set()
    texts_index: VectorStore = NumpyVectorStore()
    doc_index: VectorStore = NumpyVectorStore()
    llm_config: dict = dict(model="gpt-3.5-turbo")
    summary_llm_config: dict | None = Field(default=None, validate_default=True)
    name: str = "default"
    index_path: Path | None = PAPERQA_DIR / name
    embeddings_model: str = "text-embedding-ada-002"
    batch_size: int = 1
    max_concurrent: int = 5
    deleted_dockeys: set[DocKey] = set()
    prompts: PromptCollection = PromptCollection()
    jit_texts_index: bool = False
    # This is used to strip indirect citations that come up from the summary llm
    strip_citations: bool = True

    def __init__(self, **data):
        if "client" in data:
            client = data.pop("client")
        else:
            client = AsyncOpenAI()
        super().__init__(**data)
        self._client = client

    @field_validator("llm_config", "summary_llm_config")
    @classmethod
    def llm_config_has_type(cls, v):
        if v is not None and "model_type" not in v:
            raise ValueError("Must specify if chat or completion model.")
        return v

    @model_validator(mode="after")
    @classmethod
    def config_summary_llm_conig(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data["summary_llm_config"] is None:
                data["summary_llm_config"] = data["llm_config"]
        return data

    def clear_docs(self):
        self.texts = []
        self.docs = {}
        self.docnames = set()

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
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        chunk_chars: int = 3000,
    ) -> str | None:
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
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        chunk_chars: int = 3000,
    ) -> str | None:
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
        citation: str | None = None,
        docname: str | None = None,
        disable_check: bool = False,
        dockey: DocKey | None = None,
        chunk_chars: int = 3000,
    ) -> str | None:
        """Add a document to the collection."""
        if dockey is None:
            dockey = md5sum(path)
        if citation is None:
            # skip system because it's too hesitant to answer
            cite_chain = make_chain(
                client=self._client,
                prompt=self.prompts.cite,
                llm_config=self.summary_llm_config,
                skip_system=True,
            )
            # peak first chunk
            fake_doc = Doc(docname="", citation="", dockey=dockey)
            texts = read_doc(path, fake_doc, chunk_chars=chunk_chars, overlap=100)
            if len(texts) == 0:
                raise ValueError(f"Could not read document {path}. Is it empty?")
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            citation = loop.run_until_complete(cite_chain(texts[0].text))
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
        texts: list[Text],
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
        if texts[0].embedding is None:
            text_embeddings = self.embeddings.embed_documents([t.text for t in texts])
            for i, t in enumerate(texts):
                t.embedding = text_embeddings[i]
        if doc.embedding is None:
            doc.embedding = self.embeddings.embed_documents([doc.citation])[0]
        if not self.jit_texts_index:
            self.texts_index.add_texts_and_embeddings(texts)
        self.doc_index.add_texts_and_embeddings(doc)
        self.docs[doc.dockey] = doc
        self.texts += texts
        self.docnames.add(doc.docname)
        return True

    def delete(self, name: str | None = None, dockey: DocKey | None = None) -> None:
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
        rerank: bool | None = None,
        get_callbacks: CallbackFactory = lambda x: None,
    ) -> set[DocKey]:
        """Return a list of dockeys that match the query."""
        matches, _ = self.doc_index.max_marginal_relevance_search(
            query, k=k + len(self.deleted_dockeys)
        )
        # filter the matches
        matched_docs = [m for m in matches if m.dockey not in self.deleted_dockeys]
        if len(matched_docs) == 0:
            return set()
        # this only works for gpt-4 (in my testing)
        try:
            if (
                rerank is None
                and self.llm_config.model.startswith("gpt-4")
                or rerank is True
            ):
                chain = make_chain(
                    client=self._client,
                    prompt=self.prompts.select,
                    llm_config=self.llm_config,
                    skip_system=True,
                )
                papers = [f"{d.docname}: {d.citation}" for d in matched_docs]
                result = await chain(
                    data=[dict(question=query, papers="\n".join(papers))],
                    callbacks=get_callbacks("filter"),
                )
                return set([d.dockey for d in matched_docs if d.docname in result])
        except AttributeError:
            pass
        return set([d.dockey for d in matched_docs])

    def _build_texts_index(self, keys: set[DocKey] = None):
        if keys is not None and self.jit_texts_index:
            texts = self.texts
            if keys is not None:
                texts = [t for t in texts if t.doc.dockey in keys]
            if len(texts) == 0:
                return
            self.texts_index.clear()
            self.texts_index.add_texts_and_embeddings(texts)

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
        k: int = 10,  # Number of evidence pieces to retrieve
        max_sources: int = 5,  # Number of scored contexts to use
        marginal_relevance: bool = True,
        get_callbacks: CallbackFactory = lambda x: None,
        detailed_citations: bool = False,
        disable_vector_search: bool = False,
        disable_summarization: bool = False,
    ) -> Answer:
        if len(self.docs) == 0 and self.doc_index is None:
            # do we have no docs?
            return answer
        self._build_texts_index(keys=answer.dockey_filter)
        self.texts_index = cast(VectorStore, self.texts_index)
        _k = k
        if answer.dockey_filter is not None:
            _k = k * 10  # heuristic - get enough so we can downselect
        if disable_vector_search:
            matches = self.texts
        else:
            if marginal_relevance:
                matches = self.texts_index.max_marginal_relevance_search(
                    answer.question, k=_k, fetch_k=5 * _k
                )
            else:
                matches = self.texts_index.similarity_search(answer.question, k=_k)
        # ok now filter (like ones from adoc_match)
        if answer.dockey_filter is not None:
            matches = [m for m in matches if m.doc.dockey in answer.dockey_filter]

        # check if it is deleted
        matches = [m for m in matches if m.doc.dockey not in self.deleted_dockeys]

        # check if it is already in answer
        cur_names = [c.text.name for c in answer.contexts]
        matches = [m for m in matches if m.name not in cur_names]

        # now finally cut down
        matches = matches[:k]

        async def process(match):
            callbacks = get_callbacks("evidence:" + match.name)
            citation = match.doc.citation
            if detailed_citations:
                citation = match.name + ": " + citation

            if self.prompts.skip_summary or disable_summarization:
                context = match.text
                score = 5
            else:
                summary_chain = make_chain(
                    client=self._client,
                    prompt=self.prompts.summary,
                    llm_config=self.summary_llm_config,
                    system_prompt=self.prompts.system,
                )
                # This is dangerous because it
                # could mask errors that are important- like auth errors
                # I also cannot know what the exception
                # type is because any model could be used
                # my best idea is see if there is a 4XX
                # http code in the exception
                try:
                    context = await summary_chain(
                        dict(
                            question=answer.question,
                            # Add name so chunk is stated
                            citation=citation,
                            summary_length=answer.summary_length,
                            text=match.text,
                        ),
                        callbacks=callbacks,
                    )
                except Exception as e:
                    if guess_is_4xx(str(e)):
                        return None
                    raise e
                if (
                    "not applicable" in context.lower()
                    or "not relevant" in context.lower()
                ):
                    return None
                if self.strip_citations:
                    # remove citations that collide with our grounded citations (for the answer LLM)
                    context = strip_citations(context)
                score = get_score(context)
            c = Context(
                context=context,
                # below will remove embedding from Text/Doc
                text=Text(
                    text=match.text,
                    name=match.name,
                    doc=Doc(**match.doc),
                ),
                score=score,
            )
            return c

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
        answer: Answer | None = None,
        key_filter: bool | None = None,
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
        answer: Answer | None = None,
        key_filter: bool | None = None,
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
                client=self._client,
                prompt=self.prompts.pre,
                llm_config=self.llm_config,
                system_prompt=self.prompts.system,
            )
            pre = await chain(
                data=dict(question=answer.question), callbacks=get_callbacks("pre")
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
                client=self._client,
                prompt=self.prompts.qa,
                llm_config=self.llm_config,
                system_prompt=self.prompts.system,
            )
            answer_text = await qa_chain(
                data=dict(
                    context=answer.context,
                    answer_length=answer.answer_length,
                    question=answer.question,
                ),
                callbacks=callbacks,
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
                client=self._client,
                prompt=self.prompts.post,
                llm_config=self.llm_config,
                system_prompt=self.prompts.system,
            )
            post = await chain.arun(
                data=answer.model_dump(), callbacks=get_callbacks("post")
            )
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
