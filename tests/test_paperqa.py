import os
import pickle
from typing import Any
from unittest import IsolatedAsyncioTestCase

import requests
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.llms import OpenAI
from langchain.llms.fake import FakeListLLM

import paperqa
from paperqa.utils import strings_similarity


class TestHandler(AsyncCallbackHandler):
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token)


def test_maybe_is_text():
    assert paperqa.maybe_is_text(
        "This is a test. The sample conc. was 1.0 mM (at 245 ^F)"
    )
    assert not paperqa.maybe_is_text("\\C0\\C0\\B1\x00")
    # get front page of wikipedia
    r = requests.get("https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day")
    assert paperqa.maybe_is_text(r.text)

    # now force it to contain lots of weird encoding
    bad_text = r.text.encode("latin1", "ignore").decode("utf-16", "ignore")
    assert not paperqa.maybe_is_text(bad_text)


def test_docs():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get front page of wikipedia
        r = requests.get("https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day")
        f.write(r.text)
    llm = OpenAI(temperature=0.1, model_name="text-ada-001")
    docs = paperqa.Docs(llm=llm)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    assert docs.docs[0]["key"] == "Wiki2023"
    os.remove(doc_path)


class TokenTest(IsolatedAsyncioTestCase):
    async def test_token_count(self):
        doc_path = "example.txt"
        with open(doc_path, "w", encoding="utf-8") as f:
            # get wiki page about politician
            r = requests.get(
                "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)"
            )
            f.write(r.text)
        docs = paperqa.Docs()

        docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
        os.remove(doc_path)
        doc_path = "example2.txt"
        with open(doc_path, "w", encoding="utf-8") as f:
            # get wiki page about politician
            r = requests.get(
                "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)"
            )
            f.write(r.text)
            f.write("\n")  # so we don't have same hash
        docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed tomorrow")
        os.remove(doc_path)
        answer = await docs.aquery(
            "What is Frederick Bates's greatest accomplishment?",
            get_callbacks=lambda x: [TestHandler()],
        )
        assert answer.tokens > 100
        assert answer.cost > 0.001


def test_evidence():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    evidence = docs.get_evidence(
        paperqa.Answer("For which state was he a governor"), k=1, max_sources=1
    )
    print(evidence.contexts[0].context, evidence.context)
    assert "Missouri" in evidence.context
    os.remove(doc_path)


def test_query():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    docs.query("What is Frederick Bates's greatest accomplishment?")
    os.remove(doc_path)


class Test(IsolatedAsyncioTestCase):
    async def test_aquery(self):
        doc_path = "example.txt"
        with open(doc_path, "w", encoding="utf-8") as f:
            # get wiki page about politician
            r = requests.get(
                "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)"
            )
            f.write(r.text)
        docs = paperqa.Docs()
        docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
        await docs.aquery("What is Frederick Bates's greatest accomplishment?")
        os.remove(doc_path)


def test_doc_match():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    docs.doc_match("What is Frederick Bates's greatest accomplishment?")
    os.remove(doc_path)


def test_docs_pickle():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get front page of wikipedia
        r = requests.get("https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day")
        f.write(r.text)
    llm = OpenAI(temperature=0.0, model_name="text-curie-001")
    docs = paperqa.Docs(llm=llm)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now", chunk_chars=1000)
    docs_pickle = pickle.dumps(docs)
    docs2 = pickle.loads(docs_pickle)
    docs2.update_llm(llm)
    assert len(docs.docs) == len(docs2.docs)
    context1, context2 = (
        docs.get_evidence(
            paperqa.Answer("What date is flag day in Canada?"),
            k=3,
            max_sources=1,
        ).context,
        docs2.get_evidence(
            paperqa.Answer("What date is flag day in Canada?"),
            k=3,
            max_sources=1,
        ).context,
    )
    assert strings_similarity(context1, context2) > 0.75
    os.remove(doc_path)


def test_docs_pickle_no_faiss():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get front page of wikipedia
        r = requests.get("https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day")
        f.write(r.text)
    llm = OpenAI(temperature=0.0, model_name="text-curie-001")
    docs = paperqa.Docs(llm=llm)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now", chunk_chars=1000)
    docs._faiss_index = None
    docs_pickle = pickle.dumps(docs)
    docs2 = pickle.loads(docs_pickle)
    docs2.update_llm(llm)
    assert len(docs.docs) == len(docs2.docs)
    assert (
        strings_similarity(
            docs.get_evidence(
                paperqa.Answer("What date is flag day in Canada?"),
                k=3,
                max_sources=1,
            ).context,
            docs2.get_evidence(
                paperqa.Answer("What date is flag day in Canada?"),
                k=3,
                max_sources=1,
            ).context,
        )
        > 0.75
    )
    os.remove(doc_path)


def test_bad_context():
    doc_path = "example.html"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    answer = docs.query(
        "What year was Barack Obama born?",
        length_prompt="about 20 words",
    )
    assert "cannot answer" in answer.answer
    os.remove(doc_path)


def test_repeat_keys():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs(llm=OpenAI(temperature=0.0, model_name="text-ada-001"))
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    try:
        docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    except ValueError:
        pass
    assert len(docs.docs) == 1

    # now with different paths
    doc_path2 = "example2.txt"
    with open(doc_path2, "w", encoding="utf-8") as f:
        # get wiki page about politician
        f.write(r.text)
        f.write("\n")  # so we don't have same hash
    docs.add(doc_path2, "WikiMedia Foundation, 2023, Accessed now")
    assert len(docs.docs) == 2

    # check keys
    assert docs.docs[0]["key"] == "Wiki2023"
    assert docs.docs[1]["key"] == "Wiki2023a"

    os.remove(doc_path)
    os.remove(doc_path2)


def test_pdf_reader():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(tests_dir, "paper.pdf")
    docs = paperqa.Docs(llm=OpenAI(temperature=0.0, model_name="text-curie-001"))
    docs.add(doc_path, "Wellawatte et al, XAI Review, 2023")
    answer = docs.query("Are counterfactuals actionable?")
    assert "yes" in answer.answer or "Yes" in answer.answer


def test_pdf_pypdf_reader():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(tests_dir, "paper.pdf")
    splits1, _ = paperqa.readers._read_doc(
        doc_path, "foo te al", "bar", force_pypdf=True
    )
    splits2, _ = paperqa.readers._read_doc(
        doc_path, "foo te al", "bar", force_pypdf=False
    )
    assert strings_similarity(splits1[0].casefold(), splits2[0].casefold()) > 0.85


def test_prompt_length():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    docs.query("What is the name of the politician?", length_prompt="25 words")


def test_doc_preview():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs(llm=OpenAI(temperature=0.0, model_name="text-ada-001"))
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    assert len(docs.doc_previews()) == 1


def test_code():
    # load this script
    doc_path = os.path.abspath(__file__)
    docs = paperqa.Docs(llm=OpenAI(temperature=0.0, model_name="text-ada-001"))
    docs.add(doc_path, "test_paperqa.py", key="test", disable_check=True)
    assert len(docs.docs) == 1
    docs.query("What function tests the preview?")


def test_citation():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs()
    docs.add(doc_path)
    print(docs.docs[0]["metadata"][0]["citation"])
    assert (
        list(docs.docs)[0]["metadata"][0]["key"] == "Wikipedia2023"
        or list(docs.docs)[0]["metadata"][0]["key"] == "Frederick2023"
    )


def test_dockey_filter():
    """Test that we can filter evidence with dockeys"""
    doc_path = "example2.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    # add with new dockey
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(r.text)
        f.write("\n")  # so we don't have same hash
    docs.add("example.txt", "WikiMedia Foundation, 2023, Accessed now", key="test")
    answer = paperqa.Answer("What country is Bates from?")
    docs.get_evidence(answer, key_filter=["test"])


def test_dockey_delete():
    """Test that we can filter evidence with dockeys"""
    doc_path = "example2.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    # add with new dockey
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(r.text)
        f.write("\n\nBates could be from Angola")  # so we don't have same hash
    docs.add("example.txt", "WikiMedia Foundation, 2023, Accessed now", key="test")
    answer = paperqa.Answer("What country is Bates from?")
    answer = docs.get_evidence(answer, marginal_relevance=False)
    keys = set([c.key for c in answer.contexts])
    assert len(keys) == 2
    assert len(docs.docs) == 2
    assert len(docs.keys) == 2

    docs.delete("test")
    assert len(docs.docs) == 1
    assert len(docs.keys) == 1
    answer = paperqa.Answer("What country is Bates from?")
    answer = docs.get_evidence(answer, marginal_relevance=False)
    keys = set([c.key for c in answer.contexts])
    assert len(keys) == 1


def test_query_filter():
    """Test that we can filter evidence with in query"""
    doc_path = "example2.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs()
    docs.add(
        doc_path,
        "Information about Fredrick Bates, WikiMedia Foundation, 2023, Accessed now",
    )
    # add with new dockey
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(r.text)
        f.write("\n")  # so we don't have same hash
    docs.add("example.txt", "WikiMedia Foundation, 2023, Accessed now", key="test")
    docs.query("What country is Bates from?", key_filter=True)
    # the filter shouldn't trigger, so just checking that it doesn't crash


def test_nonopenai_model():
    responses = ["This is a test", "This is another test"] * 50
    model = FakeListLLM(responses=responses)
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs(llm=model)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    docs.query("What country is Bates from?")


def test_agent():
    docs = paperqa.Docs()
    answer = paperqa.run_agent(docs, "What compounds target AKT1")
    print(answer)


def test_zotera():
    from paperqa.contrib import ZoteroDB

    paperqa.Docs()
    try:
        ZoteroDB(library_type="user")  # "group" if group library
    except ValueError:
        # close enough
        return


def test_too_much_evidence():
    doc_path = "example2.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Barack_Obama")
        f.write(r.text)
    docs = paperqa.Docs(llm="gpt-3.5-turbo", summary_llm="gpt-3.5-turbo")
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    # add with new dockey
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(r.text)
        f.write("\n")  # so we don't have same hash
    docs.add(
        "example.txt",
        "WikiMedia Foundation, 2023, Accessed now",
        key="test",
        chunk_chars=4000,
    )
    docs.query("What is Barrack's greatest accomplishment?", max_sources=10, k=10)
