import os
import pickle
from io import BytesIO
from typing import Any
from unittest import IsolatedAsyncioTestCase

import requests
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.llms import OpenAI
from langchain.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate

from paperqa import Answer, Docs, PromptCollection
from paperqa.readers import read_doc
from paperqa.types import Doc
from paperqa.utils import maybe_is_html, maybe_is_text, name_in_text, strings_similarity


class TestHandler(AsyncCallbackHandler):
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token)


def test_maybe_is_text():
    assert maybe_is_text("This is a test. The sample conc. was 1.0 mM (at 245 ^F)")
    assert not maybe_is_text("\\C0\\C0\\B1\x00")
    # get front page of wikipedia
    r = requests.get("https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day")
    assert maybe_is_text(r.text)

    assert maybe_is_html(BytesIO(r.text.encode()))

    # now force it to contain lots of weird encoding
    bad_text = r.text.encode("latin1", "ignore").decode("utf-16", "ignore")
    assert not maybe_is_text(bad_text)


def test_name_in_text():
    name1 = "FooBar2022"
    name2 = "FooBar2022a"
    name3 = "FooBar20"

    text1 = "As mentioned by FooBar2022, this is a great paper"
    assert name_in_text(name1, text1)
    assert not name_in_text(name2, text1)
    assert not name_in_text(name3, text1)

    text2 = "This is great, as found by FooBar20"
    assert name_in_text(name3, text2)
    assert not name_in_text(name1, text2)
    assert not name_in_text(name2, text2)

    text3 = "Per previous work (FooBar2022, FooBar2022a), this is great"
    assert name_in_text(name1, text3)
    assert name_in_text(name2, text3)
    assert not name_in_text(name3, text3)

    text4 = "Per previous work (Foo2022, Bar2023), this is great"
    assert not name_in_text(name1, text4)
    assert not name_in_text(name2, text4)
    assert not name_in_text(name3, text4)

    text5 = "Per previous work (FooBar2022; FooBar2022a), this is great"
    assert name_in_text(name1, text5)
    assert name_in_text(name2, text5)
    assert not name_in_text(name3, text5)

    text6 = "According to FooBar2022 and Foobars, this is great"
    assert name_in_text(name1, text6)
    assert not name_in_text(name2, text6)
    assert not name_in_text(name3, text6)

    text7 = "As stated by FooBar2022.\n\nThis is great"
    assert name_in_text(name1, text7)
    assert not name_in_text(name2, text7)
    assert not name_in_text(name3, text7)


def test_docs():
    llm = OpenAI(client=None, temperature=0.1, model="text-ada-001")
    docs = Docs(llm=llm)
    docs.add_url(
        "https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    assert docs.docs["test"].docname == "Wiki2023"


def test_update_llm():
    doc = Docs()
    doc.update_llm("gpt-3.5-turbo")
    assert doc.llm == doc.summary_llm

    doc.update_llm(OpenAI(client=None, temperature=0.1, model="text-ada-001"))
    assert doc.llm == doc.summary_llm


def test_evidence():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    evidence = docs.get_evidence(
        Answer(question="For which state was he a governor"), k=1, max_sources=1
    )
    assert "Missouri" in evidence.context
    os.remove(doc_path)


def test_query():
    docs = Docs()
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    docs.query("What is Frederick Bates's greatest accomplishment?")


class Test(IsolatedAsyncioTestCase):
    async def test_aquery(self):
        docs = Docs()
        docs.add_url(
            "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
            citation="WikiMedia Foundation, 2023, Accessed now",
            dockey="test",
        )
        await docs.aquery("What is Frederick Bates's greatest accomplishment?")


class TestDocMatch(IsolatedAsyncioTestCase):
    def test_adoc_match(self):
        docs = Docs()
        docs.add_url(
            "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
            citation="WikiMedia Foundation, 2023, Accessed now",
            dockey="test",
        )
        docs.adoc_match("What is Frederick Bates's greatest accomplishment?")


def test_docs_pickle():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get front page of wikipedia
        r = requests.get("https://en.wikipedia.org/wiki/Take_Your_Dog_to_Work_Day")
        f.write(r.text)
    llm = OpenAI(client=None, temperature=0.0, model="text-curie-001")
    docs = Docs(llm=llm)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now", chunk_chars=1000)
    docs_pickle = pickle.dumps(docs)
    docs2 = pickle.loads(docs_pickle)
    docs2.update_llm(llm)
    assert llm.model_name == docs2.llm.model_name
    assert docs2.summary_llm.model_name == docs2.llm.model_name
    assert len(docs.docs) == len(docs2.docs)
    context1, context2 = (
        docs.get_evidence(
            Answer(
                question="What date is bring your dog to work in the US?",
                summary_length="about 20 words",
            ),
            k=3,
            max_sources=1,
        ).context,
        docs2.get_evidence(
            Answer(
                question="What date is bring your dog to work in the US?",
                summary_length="about 20 words",
            ),
            k=3,
            max_sources=1,
        ).context,
    )
    print(context1)
    print(context2)
    assert strings_similarity(context1, context2) > 0.75
    os.remove(doc_path)


def test_docs_pickle_no_faiss():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get front page of wikipedia
        r = requests.get("https://en.wikipedia.org/wiki/Take_Your_Dog_to_Work_Day")
        f.write(r.text)
    llm = OpenAI(client=None, temperature=0.0, model="text-curie-001")
    docs = Docs(llm=llm)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now", chunk_chars=1000)
    docs.doc_index = None
    docs.texts_index = None
    docs_pickle = pickle.dumps(docs)
    docs2 = pickle.loads(docs_pickle)
    docs2.update_llm(llm)
    assert len(docs.docs) == len(docs2.docs)
    assert (
        strings_similarity(
            docs.get_evidence(
                Answer(
                    question="What date is bring your dog to work in the US?",
                    summary_length="about 20 words",
                ),
                k=3,
                max_sources=1,
            ).context,
            docs2.get_evidence(
                Answer(
                    question="What date is bring your dog to work in the US?",
                    summary_length="about 20 words",
                ),
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
    docs = Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    answer = docs.query("What year was Barack Obama born?")
    assert "cannot answer" in answer.answer
    os.remove(doc_path)


def test_repeat_keys():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs(llm=OpenAI(client=None, temperature=0.0, model="text-ada-001"))
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
    ds = list(docs.docs.values())
    assert ds[0].docname == "Wiki2023"
    assert ds[1].docname == "Wiki2023a"

    os.remove(doc_path)
    os.remove(doc_path2)


def test_pdf_reader():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(tests_dir, "paper.pdf")
    docs = Docs(llm=OpenAI(client=None, temperature=0.0, model="text-curie-001"))
    docs.add(doc_path, "Wellawatte et al, XAI Review, 2023")
    answer = docs.query("Are counterfactuals actionable?")
    assert "yes" in answer.answer or "Yes" in answer.answer


def test_fileio_reader_pdf():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(tests_dir, "paper.pdf")
    with open(doc_path, "rb") as f:
        docs = Docs(llm=OpenAI(client=None, temperature=0.0, model="text-curie-001"))
        docs.add_file(f, "Wellawatte et al, XAI Review, 2023")
        answer = docs.query("Are counterfactuals actionable?")
        assert "yes" in answer.answer or "Yes" in answer.answer


def test_fileio_reader_txt():
    # can't use curie, because it has trouble with parsed HTML
    docs = Docs(llm=OpenAI(client=None, temperature=0.0))
    r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
    if r.status_code != 200:
        raise ValueError("Could not download wikipedia page")
    docs.add_file(
        BytesIO(r.text.encode()),
        "WikiMedia Foundation, 2023, Accessed now",
        chunk_chars=1000,
    )
    answer = docs.query("What country was Frederick Bates born in?")
    assert "Virginia" in answer.answer


def test_pdf_pypdf_reader():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(tests_dir, "paper.pdf")
    splits1 = read_doc(
        doc_path,
        Doc(docname="foo", citation="Foo et al, 2002", dockey="1"),
        force_pypdf=True,
        overlap=100,
        chunk_chars=3000,
    )
    splits2 = read_doc(
        doc_path,
        Doc(docname="foo", citation="Foo et al, 2002", dockey="1"),
        force_pypdf=False,
        overlap=100,
        chunk_chars=3000,
    )
    assert (
        strings_similarity(splits1[0].text.casefold(), splits2[0].text.casefold())
        > 0.85
    )


def test_prompt_length():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    docs.query("What is the name of the politician?")


def test_code():
    # load this script
    doc_path = os.path.abspath(__file__)
    docs = Docs(llm=OpenAI(client=None, temperature=0.0, model="text-ada-001"))
    docs.add(doc_path, "test_paperqa.py", docname="test_paperqa.py", disable_check=True)
    assert len(docs.docs) == 1
    docs.query("What function tests the preview?")


def test_citation():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs()
    docs.add(doc_path)
    assert (
        list(docs.docs.values())[0].docname == "Wikipedia2023"
        or list(docs.docs.values())[0].docname == "Frederick2023"
    )


def test_dockey_filter():
    """Test that we can filter evidence with dockeys"""
    doc_path = "example2.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    # add with new dockey
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(r.text)
        f.write("\n")  # so we don't have same hash
    docs.add("example.txt", "WikiMedia Foundation, 2023, Accessed now", dockey="test")
    answer = Answer(question="What country is Bates from?", key_filter=["test"])
    docs.get_evidence(answer)


def test_dockey_delete():
    """Test that we can filter evidence with dockeys"""
    doc_path = "example2.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    # add with new dockey
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(r.text)
        f.write("\n\nBates could be from Angola")  # so we don't have same hash
    docs.add("example.txt", "WikiMedia Foundation, 2023, Accessed now", dockey="test")
    answer = Answer(question="What country is Bates from?")
    answer = docs.get_evidence(answer, marginal_relevance=False)
    keys = set([c.text.doc.dockey for c in answer.contexts])
    assert len(keys) == 2
    assert len(docs.docs) == 2

    docs.delete(dockey="test")
    assert len(docs.docs) == 1
    answer = Answer(question="What country is Bates from?")
    answer = docs.get_evidence(answer, marginal_relevance=False)
    keys = set([c.text.doc.dockey for c in answer.contexts])
    assert len(keys) == 1


def test_query_filter():
    """Test that we can filter evidence with in query"""
    doc_path = "example2.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs()
    docs.add(
        doc_path,
        "Information about Fredrick Bates, WikiMedia Foundation, 2023, Accessed now",
    )
    # add with new dockey
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(r.text)
        f.write("\n")  # so we don't have same hash
    docs.add("example.txt", "WikiMedia Foundation, 2023, Accessed now", dockey="test")
    docs.query("What country is Bates from?", key_filter=True)
    # the filter shouldn't trigger, so just checking that it doesn't crash


def test_nonopenai_client():
    responses = ["This is a test", "This is another test"] * 50
    model = FakeListLLM(responses=responses)
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs(llm=model)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    docs.query("What country is Bates from?")


def test_zotera():
    from paperqa.contrib import ZoteroDB

    Docs()
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
    docs = Docs(llm="gpt-3.5-turbo", summary_llm="gpt-3.5-turbo")
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    # add with new dockey
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(r.text)
        f.write("\n")  # so we don't have same hash
    docs.add(
        "example.txt",
        "WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
        chunk_chars=4000,
    )
    docs.query("What is Barrack's greatest accomplishment?", max_sources=10, k=10)


def test_custom_prompts():
    my_qaprompt = PromptTemplate(
        input_variables=["question", "context"],
        template="Answer the question '{question}' "
        "using the country name alone. For example: "
        "A: United States\nA: Canada\nA: Mexico\n\n Using the context:\n\n{context}\n\nA: ",
    )

    docs = Docs(prompts=PromptCollection(qa=my_qaprompt))

    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    answer = docs.query("What country is Bates from?")
    print(answer.answer)
    assert "United States" == answer.answer


def test_pre_prompt():
    pre = PromptTemplate(
        input_variables=["question"],
        template="Provide context you have memorized "
        "that could help answer '{question}'. ",
    )

    docs = Docs(prompts=PromptCollection(pre=pre))

    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    docs.query("What country is Bates from?")


def test_post_prompt():
    post = PromptTemplate(
        input_variables=["question", "answer"],
        template="We are trying to answer the question below "
        "and have an answer provided. "
        "Please edit the answer be extremely terse, with no extra words or formatting"
        "with no extra information.\n\n"
        "Q: {question}\nA: {answer}\n\n",
    )

    docs = Docs(prompts=PromptCollection(post=post))

    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    docs.query("What country is Bates from?")
