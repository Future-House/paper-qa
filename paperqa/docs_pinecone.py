import asyncio
import os
import re
import sys
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Set, Union, cast , String

from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.vectorstores import FAISS, VectorStore, Pinecone
from langchain.docstore.document import Document
from pydantic import BaseModel, validator

from .docs import Docs
from .chains import get_score, make_chain
from .paths import PAPERQA_DIR
from .readers import read_doc
from .types import Answer, CallbackFactory, Context, Doc, DocKey, PromptCollection, Text
from .utils import (
    gather_with_concurrency,
    guess_is_4xx,
    maybe_is_html,
    maybe_is_pdf,
    maybe_is_text,
    md5sum,
    name_in_text,
)
import pinecone
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],  # find at app.pinecone.io
    environment=os.environ['PINECONE_ENV'],  # next to api key in console
)

class DocsPineCone(Docs):
    """A collection of documents to be used for answering questions."""
    doc_index_name: Optional[str] = None
    text_index_name: Optional[str] = None

    def __init__(self,doc_index_name="paperqa-docs", text_index_name="paperqa-text", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.doc_index_name: Optional[str] = doc_index_name
        self.text_index_name: Optional[str] = text_index_name

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
        if self.texts_index is not None:
            try:
                # TODO: Simplify - super weird
                metadatas=[{"doc_id": doc.dockey, "docname": doc.docname}]*len(texts)
                self.texts_index.add_texts( 
                    texts,metadatas=metadatas
                )
            except AttributeError:
                raise ValueError("Need a vector store that supports adding embeddings.")
        if self.doc_index is not None:
            self.doc_index.add_texts([doc.citation], metadatas=[doc.dict()])
        self.docs[doc.dockey] = doc
        self.texts += texts
        self.docnames.add(doc.docname)
        return True

    async def adoc_match(
        self, query: str, k: int = 25, get_callbacks: CallbackFactory = lambda x: None
    ) -> Set[DocKey]:
        """Return a list of dockeys that match the query."""
        if self.doc_index is None:
            if len(self.docs) == 0:
                return set()
            texts = [doc.citation for doc in self.docs.values()]
            metadatas = [d.dict() for d in self.docs.values()]
            documents_conv = [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]
            self.doc_index = Pinecone.from_documents(documents_conv, self.embeddings, index_name=self.doc_index_name)
         
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
        chain = make_chain(
            self.prompts.select, cast(BaseLanguageModel, self.llm), skip_system=True
        )
        papers = [f"{d.docname}: {d.citation}" for d in matched_docs]
        result = await chain.arun(  # type: ignore
            question=query, papers="\n".join(papers), callbacks=get_callbacks("filter")
        )
        return set([d.dockey for d in matched_docs if d.docname in result])


    def __setstate__(self, state):
        object.__setattr__(self, "__dict__", state["__dict__"])
        object.__setattr__(self, "__fields_set__", state["__fields_set__"])
        try:
            self.texts_index = Pinecone.from_existing_index(self.text_index_name, self.embeddings)
        except Exception:
            try:
                index = pinecone.Index(self.text_index_name)
                self.texts_index = Pinecone(index, embeddings.embed_query, "text")
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
            metadata=[{"doc_id": doc.dockey, "docname": doc.docname}]
            documents_conv = [Document(page_content=text, metadata=metadata) for text in raw_texts]
            self.texts_index = Pinecone.from_documents(documents_conv, self.embeddings, index_name=index_name)