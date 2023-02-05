from typing import List, Optional, Tuple, Union
import re
import gpt_index
import string
import math
from .utils import HiddenPrints
from .qaprompts import distill_chain, qa_chain
from dataclasses import dataclass


@dataclass
class Answer:
    """A class to hold the answer to a question."""

    question: str
    answer: str
    context: str
    references: str
    formatted_answer: str


def maybe_is_text(s, thresh=2.5):
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


class Docs:
    """A collection of documents to be used for answering questions."""

    def __init__(self, chunk_size_limit: int = 3000) -> None:
        self.docs = dict()
        self.chunk_size_limit = chunk_size_limit

    def add(self, path: str, citation: str, key: Optional[str] = None) -> bool:
        """Add a document to the collection."""
        if key is None:
            # get first name and year from citation
            author = re.search(r"([A-Z][a-z]+)", citation).group(1)
            year = re.search(r"(\d{4})", citation).group(1)
            key = f"{author}{year}"
        data = {"citation": citation, "key": key}
        d = gpt_index.SimpleDirectoryReader(input_files=[path]).load_data()
        # loose check to see if document was loaded
        if not maybe_is_text(d[0].text):
            return False
        with HiddenPrints():
            try:
                i = gpt_index.GPTSimpleVectorIndex(
                    d, chunk_size_limit=self.chunk_size_limit
                )
            except UnicodeEncodeError:
                return False
        data["index"] = i
        self.docs[path] = data
        return True

    # to pickle, we have to save the index as a file
    def __getstate__(self):
        state = self.__dict__.copy()
        state["docs"] = dict()
        for path in self.docs:
            self.docs[path]["index"].save_to_disk(f"{path}.json")
            # want to have a deeper copy so can delete
            state["docs"][path] = self.docs[path].copy()
            del state["docs"][path]["index"]
        return state

    def __setstate__(self, state):
        for path in state["docs"]:
            state["docs"][path][
                "index"
            ] = gpt_index.GPTSimpleVectorIndex.load_from_disk(f"{path}.json")
        self.__dict__.update(state)

    def get_evidence(self, question: str, k: int = 3, max_sources: int = 5):
        context = []
        # want to work through indices but less k
        queries = dict()
        for i in range(k):
            for doc in self.docs.values():
                index = doc["index"]
                if index not in queries:
                    query = gpt_index.indices.query.vector_store.simple.GPTSimpleVectorIndexQuery(
                        index.index_struct,
                        similarity_top_k=k,
                        embed_model=index.embed_model,
                        prompt_helper=lambda: None,  # dummy
                    )
                    queries[index] = query
                query = queries[index]
                nodes = query.get_nodes_and_similarities_for_response(question)
                if len(nodes) <= i:
                    continue
                c = (
                    doc["key"],
                    distill_chain.run(question=question, context_str=nodes[i][0].text),
                )
                if "Not applicable" not in c[1]:
                    context.append(c)
                if len(context) == max_sources:
                    break
        context_str = "\n\n".join(
            [f"{k}: {s}" for k, s in context if "Not applicable" not in s]
        )
        context_str += "\n\nValid keys: " + ", ".join([k for k, s in context])
        return context_str

    def query(self, query: str, k: int = 3, max_sources: int = 5):
        context_str = self.get_evidence(query, k=k, max_sources=max_sources)
        bib = []
        if len(context_str) < 10:
            answer = "I cannot answer this question due to insufficient information."
        else:
            answer = qa_chain.run(question=query, context_str=context_str)[1:]
        for data in self.docs.values():
            if data["key"] in answer:
                bib.append(f'{data["key"]}: {data["citation"]}')
        bib_str = "\n\n".join(bib)
        formatted_answer = f"Question: {query}\n\n{answer}\n"
        if len(bib) > 0:
            formatted_answer += f"\nReferences\n\n{bib_str}\n"
        return Answer(
            answer=answer,
            question=query,
            formatted_answer=formatted_answer,
            context=context_str,
            references=bib_str,
        )
