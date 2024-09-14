from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from datetime import datetime
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, cast
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

from paperqa.clients import DEFAULT_CLIENTS, DocMetadataClient
from paperqa.core import llm_parse_json, map_fxn_summary
from paperqa.llms import (
    EmbeddingModel,
    LLMModel,
    NumpyVectorStore,
    PromptRunner,
    VectorStore,
)
from paperqa.paths import PAPERQA_DIR
from paperqa.readers import read_doc
from paperqa.settings import MaybeSettings, get_settings
from paperqa.types import (
    Answer,
    Doc,
    DocDetails,
    DocKey,
    LLMResult,
    Text,
    set_llm_answer_ids,
)
from paperqa.utils import (
    gather_with_concurrency,
    get_loop,
    maybe_is_html,
    maybe_is_pdf,
    maybe_is_text,
    md5sum,
    name_in_text,
)


# this is just to reduce None checks/type checks
async def empty_callback(result: LLMResult) -> None:
    pass


async def print_callback(result: LLMResult) -> None:
    pass


class Docs(BaseModel):
    """A collection of documents to be used for answering questions."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    docs: dict[DocKey, Doc | DocDetails] = Field(default_factory=dict)
    texts: list[Text] = Field(default_factory=list)
    docnames: set[str] = Field(default_factory=set)
    texts_index: VectorStore = Field(default_factory=NumpyVectorStore)
    name: str = Field(default="default", description="Name of this docs collection")
    index_path: Path | None = Field(
        default=PAPERQA_DIR, description="Path to save index", validate_default=True
    )
    deleted_dockeys: set[DocKey] = Field(default_factory=set)

    @field_validator("index_path")
    @classmethod
    def handle_default(cls, value: Path | None, info: ValidationInfo) -> Path | None:
        if value == PAPERQA_DIR:
            return PAPERQA_DIR / info.data["name"]
        return value

    def clear_docs(self) -> None:
        self.texts = []
        self.docs = {}
        self.docnames = set()

    def _get_unique_name(self, docname: str) -> str:
        """Create a unique name given proposed name."""
        suffix = ""
        while docname + suffix in self.docnames:
            # move suffix to next letter
            suffix = "a" if suffix == "" else chr(ord(suffix) + 1)
        docname += suffix
        return docname

    def add_file(
        self,
        file: BinaryIO,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> str | None:
        return get_loop().run_until_complete(
            self.aadd_file(
                file,
                citation=citation,
                docname=docname,
                dockey=dockey,
                settings=settings,
                llm_model=llm_model,
                embedding_model=embedding_model,
            )
        )

    async def aadd_file(
        self,
        file: BinaryIO,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        title: str | None = None,
        doi: str | None = None,
        authors: list[str] | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        **kwargs,
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
            return await self.aadd(
                Path(f.name),
                citation=citation,
                docname=docname,
                dockey=dockey,
                title=title,
                doi=doi,
                authors=authors,
                settings=settings,
                llm_model=llm_model,
                embedding_model=embedding_model,
                **kwargs,
            )

    def add_url(
        self,
        url: str,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> str | None:
        return get_loop().run_until_complete(
            self.aadd_url(
                url,
                citation=citation,
                docname=docname,
                dockey=dockey,
                settings=settings,
                llm_model=llm_model,
                embedding_model=embedding_model,
            )
        )

    async def aadd_url(
        self,
        url: str,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> str | None:
        """Add a document to the collection."""
        import urllib.request

        with urllib.request.urlopen(url) as f:  # noqa: ASYNC210, S310
            # need to wrap to enable seek
            file = BytesIO(f.read())
            return await self.aadd_file(
                file,
                citation=citation,
                docname=docname,
                dockey=dockey,
                settings=settings,
                llm_model=llm_model,
                embedding_model=embedding_model,
            )

    def add(
        self,
        path: Path,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        title: str | None = None,
        doi: str | None = None,
        authors: list[str] | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        **kwargs,
    ) -> str | None:
        return get_loop().run_until_complete(
            self.aadd(
                path,
                citation=citation,
                docname=docname,
                dockey=dockey,
                title=title,
                doi=doi,
                authors=authors,
                settings=settings,
                llm_model=llm_model,
                embedding_model=embedding_model,
                **kwargs,
            )
        )

    async def aadd(  # noqa: PLR0912
        self,
        path: Path,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        title: str | None = None,
        doi: str | None = None,
        authors: list[str] | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        **kwargs,
    ) -> str | None:
        """Add a document to the collection."""
        all_settings = get_settings(settings)
        parse_config = all_settings.parsing
        if dockey is None:
            # md5 sum of file contents (not path!)
            dockey = md5sum(path)
        if llm_model is None:
            llm_model = all_settings.get_llm()
        if citation is None:
            # Peek first chunk
            texts = read_doc(
                path,
                Doc(docname="", citation="", dockey=dockey),  # Fake doc
                chunk_chars=parse_config.chunk_size,
                overlap=parse_config.overlap,
            )
            if not texts:
                raise ValueError(f"Could not read document {path}. Is it empty?")
            result = await llm_model.run_prompt(
                prompt=parse_config.citation_prompt,
                data={"text": texts[0].text},
                skip_system=True,  # skip system because it's too hesitant to answer
            )
            citation = result.text
            if (
                len(citation) < 3  # noqa: PLR2004
                or "Unknown" in citation
                or "insufficient" in citation
            ):
                citation = f"Unknown, {os.path.basename(path)}, {datetime.now().year}"

        if docname is None:
            # get first name and year from citation
            match = re.search(r"([A-Z][a-z]+)", citation)
            if match is not None:
                author = match.group(1)
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
                year = match.group(1)
            docname = f"{author}{year}"
        docname = self._get_unique_name(docname)

        doc = Doc(docname=docname, citation=citation, dockey=dockey)

        # try to extract DOI / title from the citation
        if (doi is title is None) and parse_config.use_doc_details:
            result = await llm_model.run_prompt(
                prompt=parse_config.structured_citation_prompt,
                data={"citation": citation},
                skip_system=True,
            )
            with contextlib.suppress(json.JSONDecodeError):
                clean_text = result.text.strip("`")
                if clean_text.startswith("json"):
                    clean_text = clean_text.replace("json", "", 1)
                citation_json = json.loads(clean_text)
                if citation_title := citation_json.get("title"):
                    title = citation_title
                if citation_doi := citation_json.get("doi"):
                    doi = citation_doi
                if citation_author := citation_json.get("authors"):
                    authors = citation_author
        # see if we can upgrade to DocDetails
        # if not, we can progress with a normal Doc
        # if "overwrite_fields_from_metadata" is used:
        # will map "docname" to "key", and "dockey" to "doc_id"
        if (title or doi) and parse_config.use_doc_details:
            if kwargs.get("metadata_client"):
                metadata_client = kwargs["metadata_client"]
            else:
                metadata_client = DocMetadataClient(
                    session=kwargs.pop("session", None),
                    clients=kwargs.pop("clients", DEFAULT_CLIENTS),
                )

            query_kwargs: dict[str, Any] = {}

            if doi:
                query_kwargs["doi"] = doi
            if authors:
                query_kwargs["authors"] = authors
            if title:
                query_kwargs["title"] = title
            doc = await metadata_client.upgrade_doc_to_doc_details(
                doc, **(query_kwargs | kwargs)
            )

        texts = read_doc(
            path,
            doc,
            chunk_chars=parse_config.chunk_size,
            overlap=parse_config.overlap,
        )
        # loose check to see if document was loaded
        if (
            not texts
            or len(texts[0].text) < 10  # noqa: PLR2004
            or (
                not parse_config.disable_doc_valid_check
                and not maybe_is_text(texts[0].text)
            )
        ):
            raise ValueError(
                f"This does not look like a text document: {path}. Pass disable_check"
                " to ignore this error."
            )
        if await self.aadd_texts(texts, doc, all_settings, embedding_model):
            return docname
        return None

    def add_texts(
        self,
        texts: list[Text],
        doc: Doc,
        settings: MaybeSettings = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> bool:
        return get_loop().run_until_complete(
            self.aadd_texts(
                texts, doc, settings=settings, embedding_model=embedding_model
            )
        )

    async def aadd_texts(
        self,
        texts: list[Text],
        doc: Doc,
        settings: MaybeSettings = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> bool:
        """
        Add chunked texts to the collection.

        NOTE: this is useful if you have already chunked the texts yourself.

        Returns:
            True if the doc was added, otherwise False if already in the collection.
        """
        all_settings = get_settings(settings)

        if embedding_model is None:
            embedding_model = all_settings.get_embedding_model()

        if doc.dockey in self.docs:
            return False
        if not texts:
            raise ValueError("No texts to add.")
        # 1. Calculate text embeddings if not already present, but don't set them into
        # the texts until we've set up the Doc's embedding, so callers can retry upon
        # OpenAI rate limit errors
        text_embeddings: list[list[float]] | None = (
            await embedding_model.embed_documents(texts=[t.text for t in texts])
            if texts[0].embedding is None
            else None
        )
        # 2. Now we can set the text embeddings
        if text_embeddings is not None:
            for t, t_embedding in zip(texts, text_embeddings, strict=True):
                t.embedding = t_embedding
        # 4. Update texts and the Doc's name
        if doc.docname in self.docnames:
            new_docname = self._get_unique_name(doc.docname)
            for t in texts:
                t.name = t.name.replace(doc.docname, new_docname)
            doc.docname = new_docname
        # 5. We do not embed here, because we do it lazily
        self.docs[doc.dockey] = doc
        self.texts += texts
        self.docnames.add(doc.docname)
        return True

    def delete(
        self,
        name: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
    ) -> None:
        """Delete a document from the collection."""
        # name is an alias for docname
        name = docname if name is None else name

        if name is not None:
            doc = next((doc for doc in self.docs.values() if doc.docname == name), None)
            if doc is None:
                return
            self.docnames.remove(doc.docname)
            dockey = doc.dockey
        del self.docs[dockey]
        self.deleted_dockeys.add(dockey)
        self.texts = list(filter(lambda x: x.doc.dockey != dockey, self.texts))

    def _build_texts_index(self) -> None:
        texts = [t for t in self.texts if t not in self.texts_index]
        self.texts_index.add_texts_and_embeddings(texts)

    async def retrieve_texts(
        self,
        query: str,
        k: int,
        settings: MaybeSettings = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> list[Text]:

        settings = get_settings(settings)
        if embedding_model is None:
            embedding_model = settings.get_embedding_model()

        # TODO: should probably happen elsewhere
        self.texts_index.mmr_lambda = settings.texts_index_mmr_lambda

        self._build_texts_index()
        _k = k + len(self.deleted_dockeys)
        matches: list[Text] = cast(
            list[Text],
            (
                await self.texts_index.max_marginal_relevance_search(
                    query, k=_k, fetch_k=2 * _k, embedding_model=embedding_model
                )
            )[0],
        )
        matches = [m for m in matches if m.doc.dockey not in self.deleted_dockeys]
        return matches[:k]

    def get_evidence(
        self,
        query: Answer | str,
        exclude_text_filter: set[str] | None = None,
        settings: MaybeSettings = None,
        callbacks: list[Callable] | None = None,
        embedding_model: EmbeddingModel | None = None,
        summary_llm_model: LLMModel | None = None,
    ) -> Answer:
        return get_loop().run_until_complete(
            self.aget_evidence(
                query=query,
                exclude_text_filter=exclude_text_filter,
                settings=settings,
                callbacks=callbacks,
                embedding_model=embedding_model,
                summary_llm_model=summary_llm_model,
            )
        )

    async def aget_evidence(
        self,
        query: Answer | str,
        exclude_text_filter: set[str] | None = None,
        settings: MaybeSettings = None,
        callbacks: list[Callable] | None = None,
        embedding_model: EmbeddingModel | None = None,
        summary_llm_model: LLMModel | None = None,
    ) -> Answer:

        evidence_settings = get_settings(settings)
        answer_config = evidence_settings.answer
        prompt_config = evidence_settings.prompts

        answer = (
            Answer(question=query, config_md5=evidence_settings.md5)
            if isinstance(query, str)
            else query
        )

        if not self.docs and len(self.texts_index) == 0:
            return answer

        if embedding_model is None:
            embedding_model = evidence_settings.get_embedding_model()

        if summary_llm_model is None:
            summary_llm_model = evidence_settings.get_summary_llm()

        exclude_text_filter = exclude_text_filter or set()
        exclude_text_filter |= {c.text.name for c in answer.contexts}

        _k = answer_config.evidence_k
        if exclude_text_filter:
            _k += len(
                exclude_text_filter
            )  # heuristic - get enough so we can downselect

        if answer_config.evidence_retrieval:
            matches = await self.retrieve_texts(
                answer.question, _k, evidence_settings, embedding_model
            )
        else:
            matches = self.texts

        if exclude_text_filter:
            matches = [m for m in matches if m.text not in exclude_text_filter]

        matches = (
            matches[: answer_config.evidence_k]
            if answer_config.evidence_retrieval
            else matches
        )

        prompt_runner: PromptRunner | None = None
        if not answer_config.evidence_skip_summary:
            if prompt_config.use_json:
                prompt_runner = partial(
                    summary_llm_model.run_prompt,
                    prompt_config.summary_json,
                    system_prompt=prompt_config.summary_json_system,
                )
            else:
                prompt_runner = partial(
                    summary_llm_model.run_prompt,
                    prompt_config.summary,
                    system_prompt=prompt_config.system,
                )

        with set_llm_answer_ids(answer.id):
            results = await gather_with_concurrency(
                answer_config.max_concurrent_requests,
                [
                    map_fxn_summary(
                        text=m,
                        question=answer.question,
                        prompt_runner=prompt_runner,
                        extra_prompt_data={
                            "summary_length": answer_config.evidence_summary_length,
                            "citation": f"{m.name}: {m.doc.citation}",
                        },
                        parser=llm_parse_json if prompt_config.use_json else None,
                        callbacks=callbacks,
                    )
                    for m in matches
                ],
            )

        for _, llm_result in results:
            answer.add_tokens(llm_result)

        answer.contexts += [r for r, _ in results if r is not None]
        return answer

    def query(
        self,
        query: Answer | str,
        settings: MaybeSettings = None,
        callbacks: list[Callable] | None = None,
        llm_model: LLMModel | None = None,
        summary_llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> Answer:
        return get_loop().run_until_complete(
            self.aquery(
                query,
                settings=settings,
                callbacks=callbacks,
                llm_model=llm_model,
                summary_llm_model=summary_llm_model,
                embedding_model=embedding_model,
            )
        )

    async def aquery(  # noqa: PLR0912
        self,
        query: Answer | str,
        settings: MaybeSettings = None,
        callbacks: list[Callable] | None = None,
        llm_model: LLMModel | None = None,
        summary_llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> Answer:

        query_settings = get_settings(settings)
        answer_config = query_settings.answer
        prompt_config = query_settings.prompts

        if llm_model is None:
            llm_model = query_settings.get_llm()
        if summary_llm_model is None:
            summary_llm_model = query_settings.get_summary_llm()
        if embedding_model is None:
            embedding_model = query_settings.get_embedding_model()

        answer = (
            Answer(question=query, config_md5=query_settings.md5)
            if isinstance(query, str)
            else query
        )

        contexts = answer.contexts

        if not contexts:
            answer = await self.aget_evidence(
                answer,
                callbacks=callbacks,
                settings=settings,
                embedding_model=embedding_model,
                summary_llm_model=summary_llm_model,
            )
            contexts = answer.contexts
        pre_str = None
        if prompt_config.pre is not None:
            with set_llm_answer_ids(answer.id):
                pre = await llm_model.run_prompt(
                    prompt=prompt_config.pre,
                    data={"question": answer.question},
                    callbacks=callbacks,
                    name="pre",
                    system_prompt=prompt_config.system,
                )
            answer.add_tokens(pre)
            pre_str = pre.text

        filtered_contexts = sorted(
            contexts,
            key=lambda x: x.score,
            reverse=True,
        )[: answer_config.answer_max_sources]
        # remove any contexts with a score of 0
        filtered_contexts = [c for c in filtered_contexts if c.score > 0]

        context_str = "\n\n".join(
            [
                f"{c.text.name}: {c.context}"
                + "".join([f"\n{k}: {v}" for k, v in (c.model_extra or {}).items()])
                + (
                    f"\nFrom {c.text.doc.citation}"
                    if answer_config.evidence_detailed_citations
                    else ""
                )
                for c in filtered_contexts
            ]
            + ([f"Extra background information: {pre_str}"] if pre_str else [])
        )

        valid_names = [c.text.name for c in filtered_contexts]
        context_str += "\n\nValid keys: " + ", ".join(valid_names)

        bib = {}
        if len(context_str) < 10:  # noqa: PLR2004
            answer_text = (
                "I cannot answer this question due to insufficient information."
            )
        else:
            with set_llm_answer_ids(answer.id):
                answer_result = await llm_model.run_prompt(
                    prompt=prompt_config.qa,
                    data={
                        "context": context_str,
                        "answer_length": answer_config.answer_length,
                        "question": answer.question,
                        "example_citation": prompt_config.EXAMPLE_CITATION,
                    },
                    callbacks=callbacks,
                    name="answer",
                    system_prompt=prompt_config.system,
                )
            answer_text = answer_result.text
            answer.add_tokens(answer_result)
        # it still happens
        if prompt_config.EXAMPLE_CITATION in answer_text:
            answer_text = answer_text.replace(prompt_config.EXAMPLE_CITATION, "")
        for c in filtered_contexts:
            name = c.text.name
            citation = c.text.doc.citation
            # do check for whole key (so we don't catch Callahan2019a with Callahan2019)
            if name_in_text(name, answer_text):
                bib[name] = citation
        bib_str = "\n\n".join(
            [f"{i+1}. ({k}): {c}" for i, (k, c) in enumerate(bib.items())]
        )

        if answer_config.answer_filter_extra_background:
            answer_text = re.sub(
                r"\([Ee]xtra [Bb]ackground [Ii]nformation\)",
                "",
                answer_text,
            )

        formatted_answer = f"Question: {answer.question}\n\n{answer_text}\n"
        if bib:
            formatted_answer += f"\nReferences\n\n{bib_str}\n"

        if prompt_config.post is not None:
            with set_llm_answer_ids(answer.id):
                post = await llm_model.run_prompt(
                    prompt=prompt_config.post,
                    data=answer.model_dump(),
                    callbacks=callbacks,
                    name="post",
                    system_prompt=prompt_config.system,
                )
            answer_text = post.text
            answer.add_tokens(post)
            formatted_answer = f"Question: {answer.question}\n\n{post}\n"
            if bib:
                formatted_answer += f"\nReferences\n\n{bib_str}\n"

        # now at end we modify, so we could have retried earlier
        answer.answer = answer_text
        answer.formatted_answer = formatted_answer
        answer.references = bib_str
        answer.contexts = contexts
        answer.context = context_str

        return answer
