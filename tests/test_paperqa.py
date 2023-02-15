import paperqa
import requests
import os
import pickle
from paperqa.utils import strings_similarity
from langchain.llms import OpenAI


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
    assert docs.docs[doc_path]["key"] == "Wiki2023"
    os.remove(doc_path)


def test_evidence():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs(llm=OpenAI(temperature=0.1, model_name="text-curie-001"))
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    evidence, _ = docs.get_evidence(
        "For which state was he a governor", k=1, max_sources=1
    )
    assert "Missouri" in evidence
    os.remove(doc_path)


def test_query():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs(llm=OpenAI(temperature=0.1, model_name="text-ada-001"))
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    answer = docs.query("What is Frederick Bates's greatest accomplishment?")
    print(answer)
    os.remove(doc_path)


def test_docs_pickle():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get front page of wikipedia
        r = requests.get("https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day")
        f.write(r.text)
    docs = paperqa.Docs(llm=OpenAI(temperature=0.0, model_name="text-babbage-001"))
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    docs_pickle = pickle.dumps(docs)
    docs2 = pickle.loads(docs_pickle)
    assert len(docs.docs) == len(docs2.docs)
    assert (
        strings_similarity(
            docs.get_evidence("What date is flag day in Canada?", k=1, max_sources=1)[
                0
            ],
            docs2.get_evidence("What date is flag day in Canada?", k=1, max_sources=1)[
                0
            ],
        )
        > 0.75
    )
    os.remove(doc_path)


def test_bad_context():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs(llm=OpenAI(model_name="text-davinci-003"))
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    answer = docs.query("What year was Barack Obama born greatest accomplishment?")
    assert (
        answer.answer
        == "I cannot answer this question due to insufficient information."
    )
    os.remove(doc_path)


def test_repeat_keys():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs(llm=OpenAI(temperature=0.0, model_name="text-ada-003"))
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
    docs.add(doc_path2, "WikiMedia Foundation, 2023, Accessed now")
    assert len(docs.docs) == 2

    # check keys
    assert docs.docs[doc_path]["key"] == "Wiki2023"
    assert docs.docs[doc_path2]["key"] == "Wiki2023a"

    os.remove(doc_path)
    os.remove(doc_path2)


def test_pdf_reader():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(tests_dir, "paper.pdf")
    docs = paperqa.Docs(llm=OpenAI(temperature=0.0, model_name="text-curie-001"))
    docs.add(doc_path, "Wellawatte et al, XAI Review, 2023")
    answer = docs.query("Are counterfactuals actionable (yes/no)?")
    assert "yes" in answer.answer or "Yes" in answer.answer


def test_prompt_length():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = paperqa.Docs(llm=OpenAI(temperature=0.0, model_name="text-ada-001"))
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")
    docs.query("What is the name of the politician?", length_prompt="25 words")
