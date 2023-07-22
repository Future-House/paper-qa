import os
import pickle
from io import BytesIO
from typing import Any
from unittest import IsolatedAsyncioTestCase

import numpy as np
import requests
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.llms import OpenAI
from langchain.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate

from paperqa import Answer, Docs, PromptCollection, Text
from paperqa.chains import get_score
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


def test_extract_score():
    sample = """
    The text describes an experiment where different cell subtypes,
    including colorectal cancer-associated fibroblasts, were treated with
    oxaliplatin for 12 days. The concentration of oxaliplatin used was the
    EC50 for each cell subtype, which was determined individually.
    The media were changed every 3 days to avoid complete cell death.
    The text does not provide information about the percentage of colorectal
    cancer-associated fibroblasts that typically survive at 2 weeks when cultured
    with oxaliplatin. (0/10)
    """
    assert get_score(sample) == 0

    sample = """
    COVID-19 vaccinations have been shown to be effective against hospitalization
    from the Omicron and Delta variants, though effectiveness may decrease over
    time. A study found that vaccine effectiveness against hospitalization peaked
    around 82-92% after a third dose but declined to 53-77% 15+ weeks after the third
    dose, depending on age group and hospitalization definition. Stricter
    definitions of hospitalization, like requiring oxygen use or ICU admission,
    showed higher and more sustained vaccine effectiveness. 8
    """

    assert get_score(sample) == 8

    sample = """
    Here is a 100-word summary of the text:
    The text discusses a phase 3 trial of a combined
    vector vaccine based on rAd26 and rAd5 vectors carrying the
    SARS-CoV-2 spike protein gene. The trial aimed to assess the efficacy,
    immunogenicity and safety of the vaccine against COVID-19 in adults.
    The study design was a randomized, double-blind, placebo-controlled trial
    done at 25 hospitals in Moscow, Russia. Eligible participants were 18 years
    or older with no history of COVID-19. The exclusion criteria ensured
    participants were healthy and had no contraindications for vaccination.
    The trial aimed to determine if the vaccine could safely and effectively
    provide protection against COVID-19. Relevance score: 8
    """

    assert get_score(sample) == 8

    sample = """
    Here is a 100-word summary of the provided text: The text details
    trial procedures for a COVID-19 vaccine, including screening
    visits, observation visits to assess vital signs, PCR testing, and
    telemedicine consultations. Participants who tested positive for
    COVID-19 during screening were excluded from the trial. During the trial
    , additional PCR tests were only done when COVID-19 symptoms were reported
    . An electronic health record platform was in place to record data from
    telemedicine consultations. The text details the screening and trial
    procedures but does not provide direct evidence regarding the
    effectiveness of COVID-19 vaccinations. Score: 3/10
    """

    assert get_score(sample) == 3

    sample = """
    Here is a 100-word summary of the text: The text discusses a
    phase 3 trial of a COVID-19 vaccine in Russia. The vaccine
    uses a heterologous prime-boost regimen, providing robust
    immune responses. The vaccine can be stored at -18°C and
    2-8°C. The study reports 91.6% efficacy against COVID-19 based on
    interim analysis of over 21,000 participants. The authors
    compare their results to other published COVID-19 vaccine
    efficacy data. They previously published safety and immunogenicity
    results from phase 1/2 trials of the same vaccine. Relevance score:
    8/10. The text provides details on the efficacy and immune response
    generated by one COVID-19 vaccine in a large phase 3 trial, which is
    relevant evidence to help answer the question regarding effectiveness
    of COVID-19 vaccinations.
    """

    assert get_score(sample) == 8

    sample = """
    Here is a 100-word summary of the text: The text discusses the safety and
    efficacy of the BNT162b2 mRNA Covid-19 vaccine. The study found that
    the vaccine was well tolerated with mostly mild to moderate side
    effects. The vaccine was found to be highly effective against Covid-19,
    with an observed vaccine efficacy of 90.5% after the second dose.
    Severe Covid-19 cases were also reduced among vaccine recipients.
    The vaccine showed an early protective effect after the first dose
    and reached full efficacy 7 days after the second dose. The favorable
    safety and efficacy results provide evidence that the BNT162b2 vaccine
    is effective against Covid-19. The text provides data on the efficacy
    and safety results from a clinical trial of the BNT162b2 Covid-19 vaccine,
    which is highly relevant to answering the question about the effectiveness
    of Covid-19 vaccinations.
    """

    assert get_score(sample) == 5


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
    doc_path = "example.html"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    evidence = docs.get_evidence(
        Answer(question="For which state was Bates a governor?"), k=1, max_sources=1
    )
    print(evidence.context)
    assert "Missouri" in evidence.context

    evidence = docs.get_evidence(
        Answer(question="For which state was Bates a governor?"),
        detailed_citations=True,
    )
    assert "Based on WikiMedia Foundation, 2023, Accessed now" in evidence.context
    os.remove(doc_path)


def test_query():
    docs = Docs()
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    docs.query("What is Frederick Bates's greatest accomplishment?")


def test_duplicate():
    docs = Docs()
    assert docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    assert (
        docs.add_url(
            "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
            citation="WikiMedia Foundation, 2023, Accessed now",
            dockey="test",
        )
        is None
    )


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
    doc_path = "example.html"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get front page of wikipedia
        r = requests.get("https://en.wikipedia.org/wiki/Take_Your_Dog_to_Work_Day")
        f.write(r.text)
    llm = OpenAI(client=None, temperature=0.0, model="text-curie-001")
    docs = Docs(llm=llm)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now", chunk_chars=1000)
    os.remove(doc_path)
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
    assert strings_similarity(context1, context2) > 0.75
    # make sure we can query
    docs.query("What date is bring your dog to work in the US?")


def test_docs_pickle_no_faiss():
    doc_path = "example.html"
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
    answer = Answer(question="What country was Bates born in?")
    answer = docs.get_evidence(answer, marginal_relevance=False)
    print(answer)
    keys = set([c.text.doc.dockey for c in answer.contexts])
    assert len(keys) == 2
    assert len(docs.docs) == 2

    docs.delete(dockey="test")
    assert len(docs.docs) == 1
    answer = Answer(question="What country was Bates born in?")
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

    doc_path = "example.html"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    answer = docs.query("What country is Frederick Bates from?")
    print(answer.answer)
    assert "United States" in answer.answer


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


def test_memory():
    # Not sure why, but gpt-3.5 cannot do this anymore.
    docs = Docs(memory=True, k=3, max_sources=1, llm="gpt-4", key_filter=False)
    docs.add_url(
        "https://en.wikipedia.org/wiki/Red_Army",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    answer1 = docs.query("When did the Soviet Union and Japan agree to a cease-fire?")
    assert answer1.memory is not None
    assert "1939" in answer1.answer
    assert "Answer" in docs.memory_model.load_memory_variables({})["memory"]
    answer2 = docs.query("When was it resolved?")
    assert "1941" in answer2.answer or "1945" in answer2.answer
    assert answer2.memory is not None
    assert "Answer" in docs.memory_model.load_memory_variables({})["memory"]
    print(answer2.answer)

    docs.clear_memory()

    answer3 = docs.query("When was it resolved?")
    assert answer3.memory is not None
    assert (
        "I cannot answer" in answer3.answer
        or "insufficient" in answer3.answer
        or "does not provide" in answer3.answer
    )


def test_add_texts():
    llm = OpenAI(client=None, temperature=0.1, model="text-ada-001")
    docs = Docs(llm=llm)
    docs.add_url(
        "https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )

    docs2 = Docs()
    texts = [Text(**dict(t)) for t in docs.texts]
    for t in texts:
        t.embeddings = None
    docs2.add_texts(texts, list(docs.docs.values())[0])

    for t1, t2 in zip(docs2.texts, docs.texts):
        assert t1.text == t2.text
        assert np.allclose(t1.embeddings, t2.embeddings, atol=1e-3)

    docs2._build_texts_index()
    # now do it again to test after text index is already built
    llm = OpenAI(client=None, temperature=0.1, model="text-ada-001")
    docs = Docs(llm=llm)
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test3",
    )

    texts = [Text(**dict(t)) for t in docs.texts]
    for t in texts:
        t.embeddings = None
    docs2.add_texts(texts, list(docs.docs.values())[0])
    assert len(docs2.docs) == 2

    docs2.query("What country was Bates Born in?")


def test_external_doc_index():
    docs = Docs()
    docs.add_url(
        "https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    evidence = docs.query(query="What is the date of flag day?", key_filter=True)
    docs2 = Docs(doc_index=docs.doc_index, texts_index=docs.texts_index)
    assert len(docs2.docs) == 0
    evidence = docs2.query("What is the date of flag day?", key_filter=True)
    assert "February 15" in evidence.context


def test_external_texts_index():
    docs = Docs(jit_texts_index=True)
    docs.add_url(
        "https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day",
        citation="Flag Day of Canada, WikiMedia Foundation, 2023, Accessed now",
    )
    answer = docs.query(query="What is the date of flag day?", key_filter=True)
    assert "February 15" in answer.answer

    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="Fredrick Bates, WikiMedia Foundation, 2023, Accessed now",
    )

    answer = docs.query(query="What is the date of flag day?", key_filter=False)
    assert "February 15" in answer.answer

    answer = docs.query(query="What is the date of flag day?", key_filter=True)
    assert "February 15" in answer.answer
