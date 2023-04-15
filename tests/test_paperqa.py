import paperqa
import requests
import os
import pickle
import pyperclip
from paperqa.utils import strings_similarity
from langchain.llms import OpenAI
from unittest import IsolatedAsyncioTestCase

def test_add_from_pdf_url():
    pdf_url = "https://arxiv.org/pdf/2009.09808.pdf"
    docs = paperqa.Docs()
    docs.add_pdf_from_url(pdf_url)

    assert len(docs.docs) == 1

def test_add_from_zotero_clipboard():
    # create full path from relative path
    relative_path = 'paper.pdf'
    relative_path2 = 'paper2.pdf'
    full_path = os.path.join(os.path.dirname(__file__), relative_path)
    full_path2 = os.path.join(os.path.dirname(__file__), relative_path2)
    clipboard_data = """
    @inproceedings{turquois_exploring_2016,
        location = {Carrboro North Carolina {USA}},
        title = {Exploring the Benefits of 2D Visualizations for Drum Samples Retrieval},
        isbn = {978-1-4503-3751-9},
        url = {https://dl.acm.org/doi/10.1145/2854946.2854996},
        doi = {10.1145/2854946.2854996},
        abstract = {This paper explores the potential benefits of using similarity-based two-dimensional classifications and their corresponding {GUIs}, for drum samples retrieval in a creativity-oriented context. Preliminary user studies with professional electronic music producers point up the frustration and laboriousness of finding suitable drum samples in the increasingly large libraries of sounds available, and suggest the need for alternative interfaces and approaches. To address this issue, two novel spatial visualizations (respectively organized by name and by timbre-similarity) are designed as potential alternatives to the traditional 1D list-based browsers. These visualizations are implemented and compared in a music creation task, in terms of both the exploration experience and the resulting production quality, within a system for drum kit configuration. Our study shows that spatial visualizations do improve the overall exploration experience, and reveals the potential of similarity-based arrangements for the support of creative processes.},
        eventtitle = {{CHIIR} '16: Conference on Human Information Interaction and Retrieval},
        pages = {329--332},
        booktitle = {Proceedings of the 2016 {ACM} on Conference on Human Information Interaction and Retrieval},
        publisher = {{ACM}},
        author = {Turquois, Chloé and Hermant, Martin and Gómez-Marín, Daniel and Jordà, Sergi},
        urldate = {2023-04-15},
        date = {2016-03-13},
        langid = {english},
        file = {Turquois et al. - 2016 - Exploring the Benefits of 2D Visualizations for Dr.pdf:paper.pdf:application/pdf},
    }

    @misc{bitton_assisted_2019,
        title = {Assisted Sound Sample Generation with Musical Conditioning in Adversarial Auto-Encoders},
        url = {http://arxiv.org/abs/1904.06215},
        abstract = {Deep generative neural networks have thrived in the ﬁeld of computer vision, enabling unprecedented intelligent image processes. Yet the results in audio remain less advanced and many applications are still to be investigated. Our project targets real-time sound synthesis from a reduced set of high-level parameters, including semantic controls that can be adapted to different sound libraries and speciﬁc tags. These generative variables should allow expressive modulations of target musical qualities and continuously mix into new styles.},
        number = {{arXiv}:1904.06215},
        publisher = {{arXiv}},
        author = {Bitton, Adrien and Esling, Philippe and Caillon, Antoine and Fouilleul, Martin},
        urldate = {2023-04-15},
        date = {2019-06-22},
        langid = {english},
        eprinttype = {arxiv},
        eprint = {1904.06215 [cs, eess]},
        keywords = {Computer Science - Machine Learning, Computer Science - Sound, Electrical Engineering and Systems Science - Audio and Speech Processing},
        file = {Bitton et al. - 2019 - Assisted Sound Sample Generation with Musical Cond.pdf:paper2.pdf:application/pdf},
    }
    """.replace("paper.pdf", full_path).replace("paper2.pdf", full_path2)
    # copy the clipboard data to the clipboard
    pyperclip.copy(clipboard_data)

    # initialize a docs object
    docs = paperqa.Docs()

    # add the clipboard data to the docs object
    docs.add_from_zotero_clipboard()

    assert docs.docs[full_path]["key"] == "Turquois2016"
    assert docs.docs[full_path2]["key"] == "Bitton1904"


def test_count_authors():
    authors = [x.replace(" and ", ", ") for x in ['Doe, John','Doe, John and Smith, Jane','Doe, John and Smith, Jane and Jones, Joe']]
    assert(paperqa.utils.count_authors(authors[0]) == 1)
    assert(paperqa.utils.count_authors(authors[1]) == 2)
    assert(paperqa.utils.count_authors(authors[2]) == 3)

def test_clean_citation():
    # define a messy citation with extra spaces, artifacts, and question marks
    messy_citation = '  Doe, John. ????. "An Example Article". Example Journal. 1, pp. 1–10.  .'
    new_citation = paperqa.utils.clean_citation(messy_citation)

    assert(new_citation == 'Doe, John. "An Example Article". Example Journal. 1, pp. 1–10.')

def test_zotero_clipboard_to_mla_citations():
    test_citation = """
        @article{doe2022example,
        title={An Example Article},
        author={Doe, John},
        journal={Example Journal},
        volume={1},
        pages={1--10},
        year={2022}
        }

        @article{john2022example,
        title={An Example Article},
        author={Doe, John and Smith, Jane},
        }

        @article{doe2022example,
        title={An Example Article},
        author={Doe, John and Smith, Jane and Jones, Joe},
        journal={Example Journal},
        volume={1},
        pages={1--10},
        year={2018}
        }
        """
    
    mla_citations = paperqa.utils.zotero_clipboard_to_mla_citations(test_citation)

    assert(len(mla_citations) == 3)
    assert(mla_citations[0] == '[0] Doe, John. "An Example Article." Example Journal 1, 2022, pp. 1–10.')
    assert(mla_citations[1] == '[1] Doe, John, and Jane Smith. "An Example Article."')
    assert(mla_citations[2] == '[2] Doe, John, et al. "An Example Article." Example Journal 1, 2018, pp. 1–10.')


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
    llm = OpenAI(temperature=0.0, model_name="text-babbage-001")
    docs = paperqa.Docs(llm=llm)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now", chunk_chars=1000)
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
    answer = docs.query("Are counterfactuals actionable?")
    assert "yes" in answer.answer or "Yes" in answer.answer


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
    assert (
        list(docs.docs.values())[0]["metadata"][0]["key"] == "Wikipedia2023"
        or list(docs.docs.values())[0]["metadata"][0]["key"] == "Frederick2023"
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
    docs.add("example.txt", "WikiMedia Foundation, 2023, Accessed now", key="test")
    answer = paperqa.Answer("What country is Bates from?")
    docs.get_evidence(answer, key_filter=["test"])


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
    docs.add("example.txt", "WikiMedia Foundation, 2023, Accessed now", key="test")
    answer = docs.query("What country is Bates from?", key_filter=True)
    # the filter shouldn't trigger, so just checking that it doesn't crash


def disabled_test_agent():
    docs = paperqa.Docs()
    answer = paperqa.run_agent(docs, "What compounds target AKT1")
    print(answer)

def test_zotera():
    from paperqa.contrib import ZoteroDB

    docs = paperqa.Docs()
    try:
        zotero = ZoteroDB(library_type="user")  # "group" if group library
    except ValueError:
        # close enough
        return
