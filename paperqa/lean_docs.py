from __future__ import annotations

import logging
import re

from collections.abc import Callable
from functools import partial
from typing import Any
from uuid import UUID, uuid4

from llmclient import Embeddable, EmbeddingModel, LLMModel
from pydantic import BaseModel, ConfigDict, Field

from paperqa.core import llm_parse_json, map_fxn_summary
from paperqa.prompts import CANNOT_ANSWER_PHRASE
from paperqa.types import Doc,DocDetails, DocKey, PQASession, Text, set_llm_session_ids
from paperqa.utils import gather_with_concurrency, get_loop, name_in_text
from paperqa.settings import MaybeSettings, get_settings
from paperqa.llms import (
    QdrantVectorStore,
    PromptRunner,
)
logger = logging.getLogger(__name__)


class LeanDocs(BaseModel):
    """A RAM-efficient collection of documents using Qdrant for storage."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    docs: dict[DocKey, Doc | DocDetails] = Field(default_factory=dict)
    docnames: set[str] = Field(default_factory=set)
    texts_index: QdrantVectorStore = Field(...)  # Required Qdrant instance
    name: str = Field(default="default")
    deleted_dockeys: set[DocKey] = Field(default_factory=set)

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.docs == other.docs
            and self.docnames == other.docnames
            and self.texts_index == other.texts_index
            and self.name == other.name
            # NOTE: ignoring deleted_dockeys
        )

    def _get_unique_name(self, docname: str) -> str:
        """Create a unique name given proposed name."""
        suffix = ""
        while docname + suffix in self.docnames:
            suffix = "a" if not suffix else chr(ord(suffix) + 1)
        docname += suffix
        return docname

    def clear_docs(self) -> None:
        """Clear all documents from the collection."""
        self.docs = {}
        self.docnames = set()
        self.deleted_dockeys = set()
        get_loop().run_until_complete(self.texts_index.clear())

    def delete(self, name: str | None = None, docname: str | None = None, 
               dockey: DocKey | None = None) -> None:
        """Delete a document from the collection."""
        name = docname if name is None else name

        if name is not None:
            doc = next((doc for doc in self.docs.values() if doc.docname == name), None)
            if doc is None:
                return
            self.docnames.remove(doc.docname)
            dockey = doc.dockey

        if dockey in self.docs:
            del self.docs[dockey]
            self.deleted_dockeys.add(dockey)


    async def retrieve_texts(
        self,
        query: str,
        k: int,
        settings: MaybeSettings = None,
        embedding_model: EmbeddingModel | None = None,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> list[Text]:
        """Retrieve relevant texts from Qdrant."""
        settings = get_settings(settings)
        if embedding_model is None:
            embedding_model = settings.get_embedding_model()

        matches, _ = await self.texts_index.similarity_search(
            query, k + len(self.deleted_dockeys), embedding_model
        )
        matches = [m for m in matches if m.doc.dockey not in self.deleted_dockeys]
        return matches[:k]

    def query(
        self,
        query: PQASession | str,
        settings: MaybeSettings = None,
        callbacks: list[Callable] | None = None,
        llm_model: LLMModel | None = None,
        summary_llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> PQASession:
        """Synchronous wrapper for aquery."""
        return get_loop().run_until_complete(
            self.aquery(
                query,
                settings=settings,
                callbacks=callbacks,
                llm_model=llm_model,
                summary_llm_model=summary_llm_model,
                embedding_model=embedding_model,
                partitioning_fn=partitioning_fn,
            )
        )

    def get_evidence(
        self,
        query: PQASession | str,
        exclude_text_filter: set[str] | None = None,
        settings: MaybeSettings = None,
        callbacks: list[Callable] | None = None,
        embedding_model: EmbeddingModel | None = None,
        summary_llm_model: LLMModel | None = None,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> PQASession:
        """Synchronous wrapper for aget_evidence."""
        return get_loop().run_until_complete(
            self.aget_evidence(
                query=query,
                exclude_text_filter=exclude_text_filter,
                settings=settings,
                callbacks=callbacks,
                embedding_model=embedding_model,
                summary_llm_model=summary_llm_model,
                partitioning_fn=partitioning_fn,
            )
        )

    async def aget_evidence(
        self,
        query: PQASession | str,
        exclude_text_filter: set[str] | None = None,
        settings: MaybeSettings = None,
        callbacks: list[Callable] | None = None,
        embedding_model: EmbeddingModel | None = None,
        summary_llm_model: LLMModel | None = None,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> PQASession:
        """Get evidence for a query."""
        evidence_settings = get_settings(settings)
        answer_config = evidence_settings.answer
        prompt_config = evidence_settings.prompts

        session = (
            PQASession(question=query, config_md5=evidence_settings.md5)
            if isinstance(query, str)
            else query
        )

        if not self.docs:  # Changed from docs_metadata to docs
            return session

        if embedding_model is None:
            embedding_model = evidence_settings.get_embedding_model()

        if summary_llm_model is None:
            summary_llm_model = evidence_settings.get_summary_llm()

        exclude_text_filter = exclude_text_filter or set()
        exclude_text_filter |= {c.text.name for c in session.contexts}

        _k = answer_config.evidence_k
        if exclude_text_filter:
            _k += len(exclude_text_filter)

        if answer_config.evidence_retrieval:
            matches = await self.retrieve_texts(
                session.question,
                _k,
                evidence_settings,
                embedding_model,
                partitioning_fn=partitioning_fn,
            )
        else:
            # This is a fallback that might not be memory-efficient
            # Consider implementing a streaming approach if needed
            matches = []
            async for text in self.texts_index.iterate_texts():
                matches.append(text)

        if exclude_text_filter:
            matches = [m for m in matches if m.name not in exclude_text_filter]

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

        with set_llm_session_ids(session.id):
            results = await gather_with_concurrency(
                answer_config.max_concurrent_requests,
                [
                    map_fxn_summary(
                        text=m,
                        question=session.question,
                        prompt_runner=prompt_runner,
                        extra_prompt_data={
                            "summary_length": answer_config.evidence_summary_length,
                            "citation": f"{m.name}: {m.doc.formatted_citation}",
                        },
                        parser=llm_parse_json if prompt_config.use_json else None,
                        callbacks=callbacks,
                    )
                    for m in matches
                ],
            )

        for _, llm_result in results:
            session.add_tokens(llm_result)

        session.contexts += [r for r, _ in results]
        return session

    async def aquery(
        self,
        query: PQASession | str,
        settings: MaybeSettings = None,
        callbacks: list[Callable] | None = None,
        llm_model: LLMModel | None = None,
        summary_llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> PQASession:
        """Query the document collection."""
        query_settings = get_settings(settings)
        answer_config = query_settings.answer
        prompt_config = query_settings.prompts

        if llm_model is None:
            llm_model = query_settings.get_llm()
        if summary_llm_model is None:
            summary_llm_model = query_settings.get_summary_llm()
        if embedding_model is None:
            embedding_model = query_settings.get_embedding_model()

        session = (
            PQASession(question=query, config_md5=query_settings.md5)
            if isinstance(query, str)
            else query
        )

        contexts = session.contexts
        if answer_config.get_evidence_if_no_contexts and not contexts:
            session = await self.aget_evidence(
                session,
                callbacks=callbacks,
                settings=settings,
                embedding_model=embedding_model,
                summary_llm_model=summary_llm_model,
                partitioning_fn=partitioning_fn,
            )
            contexts = session.contexts

        pre_str = None
        if prompt_config.pre is not None:
            with set_llm_session_ids(session.id):
                pre = await llm_model.run_prompt(
                    prompt=prompt_config.pre,
                    data={"question": session.question},
                    callbacks=callbacks,
                    name="pre",
                    system_prompt=prompt_config.system,
                )
            session.add_tokens(pre)
            pre_str = pre.text

        filtered_contexts = sorted(
            contexts,
            key=lambda x: (-x.score, x.text.name),
        )[: answer_config.answer_max_sources]
        filtered_contexts = [c for c in filtered_contexts if c.score > 0]

        context_inner_prompt = prompt_config.context_inner
        if (
            not answer_config.evidence_detailed_citations
            and "\nFrom {citation}" in context_inner_prompt
        ):
            context_inner_prompt = context_inner_prompt.replace("\nFrom {citation}", "")

        inner_context_strs = [
            context_inner_prompt.format(
                name=c.text.name,
                text=c.context,
                citation=c.text.doc.formatted_citation,
                **(c.model_extra or {}),
            )
            for c in filtered_contexts
        ]
        if pre_str:
            inner_context_strs.append(f"Extra background information: {pre_str}")

        context_str = prompt_config.context_outer.format(
            context_str="\n\n".join(inner_context_strs),
            valid_keys=", ".join([c.text.name for c in filtered_contexts]),
        )

        bib = {}
        if len(context_str) < 10:
            answer_text = (
                f"{CANNOT_ANSWER_PHRASE} this question due to insufficient information."
            )
        else:
            with set_llm_session_ids(session.id):
                answer_result = await llm_model.run_prompt(
                    prompt=prompt_config.qa,
                    data={
                        "context": context_str,
                        "answer_length": answer_config.answer_length,
                        "question": session.question,
                        "example_citation": prompt_config.EXAMPLE_CITATION,
                    },
                    callbacks=callbacks,
                    name="answer",
                    system_prompt=prompt_config.system,
                )
            answer_text = answer_result.text
            session.add_tokens(answer_result)

        if prompt_config.EXAMPLE_CITATION in answer_text:
            answer_text = answer_text.replace(prompt_config.EXAMPLE_CITATION, "")

        for c in filtered_contexts:
            name = c.text.name
            citation = c.text.doc.formatted_citation
            if name_in_text(name, answer_text):
                bib[name] = citation

        bib_str = "\n\n".join(
            [f"{i + 1}. ({k}): {c}" for i, (k, c) in enumerate(bib.items())]
        )

        if answer_config.answer_filter_extra_background:
            answer_text = re.sub(
                r"\([Ee]xtra [Bb]ackground [Ii]nformation\)",
                "",
                answer_text,
            )

        formatted_answer = f"Question: {session.question}\n\n{answer_text}\n"
        if bib:
            formatted_answer += f"\nReferences\n\n{bib_str}\n"

        if prompt_config.post is not None:
            with set_llm_session_ids(session.id):
                post = await llm_model.run_prompt(
                    prompt=prompt_config.post,
                    data=session.model_dump(),
                    callbacks=callbacks,
                    name="post",
                    system_prompt=prompt_config.system,
                )
            answer_text = post.text
            session.add_tokens(post)
            formatted_answer = f"Question: {session.question}\n\n{post}\n"
            if bib:
                formatted_answer += f"\nReferences\n\n{bib_str}\n"

        session.answer = answer_text
        session.formatted_answer = formatted_answer
        session.references = bib_str
        session.contexts = contexts
        session.context = context_str

        return session