import os
import pickle
from io import BytesIO
from unittest import IsolatedAsyncioTestCase

import numpy as np
import requests
from openai import AsyncOpenAI

from paperqa import (
    Answer,
    Doc,
    Docs,
    NumpyVectorStore,
    PromptCollection,
    Text,
    print_callback,
)
from paperqa.llms import (
    EmbeddingModel,
    LangchainEmbeddingModel,
    LangchainLLMModel,
    LangchainVectorStore,
    LLMModel,
    OpenAIEmbeddingModel,
    OpenAILLMModel,
    get_score,
    guess_model_type,
    is_openai_model,
)
from paperqa.readers import read_doc
from paperqa.utils import (
    get_citenames,
    maybe_is_html,
    maybe_is_text,
    name_in_text,
    strings_similarity,
    strip_citations,
)


def test_is_openai_model():
    assert is_openai_model("gpt-3.5-turbo")
    assert is_openai_model("babbage-002")
    assert is_openai_model("gpt-4-1106-preview")
    assert is_openai_model("davinci-002")
    assert not is_openai_model("llama")
    assert not is_openai_model("labgpt")
    assert not is_openai_model("mixtral-7B")


def test_guess_model_type():
    assert guess_model_type("gpt-3.5-turbo") == "chat"
    assert guess_model_type("babbage-002") == "completion"
    assert guess_model_type("gpt-4-1106-preview") == "chat"
    assert guess_model_type("gpt-3.5-turbo-instruct") == "completion"
    assert guess_model_type("davinci-002") == "completion"


def test_get_citations():
    text = (
        "Yes, COVID-19 vaccines are effective. Various studies have documented the "
        "effectiveness of COVID-19 vaccines in preventing severe disease, "
        "hospitalization, and death. The BNT162b2 vaccine has shown effectiveness "
        "ranging from 65% to -41% for the 5-11 years age group and 76% to 46% for the "
        "12-17 years age group, after the emergence of the Omicron variant in New York "
        "(Dorabawila2022EffectivenessOT). Against the Delta variant, the effectiveness "
        "of the BNT162b2 vaccine was approximately 88% after two doses "
        "(Bernal2021EffectivenessOC pg. 1-3).\n\n"
        "Vaccine effectiveness was also found to be 89% against hospitalization and "
        "91% against emergency department or urgent care clinic visits "
        "(Thompson2021EffectivenessOC pg. 3-5, Goo2031Foo pg. 3-4). In the UK "
        "vaccination program, vaccine effectiveness was approximately 56% in "
        "individuals aged ≥70 years between 28-34 days post-vaccination, increasing to "
        "approximately 58% from day 35 onwards (Marfé2021EffectivenessOC).\n\n"
        "However, it is important to note that vaccine effectiveness can decrease over "
        "time. For instance, the effectiveness of COVID-19 vaccines against severe "
        "COVID-19 declined to 64% after 121 days, compared to around 90% initially "
        "(Chemaitelly2022WaningEO, Foo2019Bar). Despite this, vaccines still provide "
        "significant protection against severe outcomes (Bar2000Foo pg 1-3; Far2000 pg 2-5)."
    )
    ref = {
        "Dorabawila2022EffectivenessOT",
        "Bernal2021EffectivenessOC pg. 1-3",
        "Thompson2021EffectivenessOC pg. 3-5",
        "Goo2031Foo pg. 3-4",
        "Marfé2021EffectivenessOC",
        "Chemaitelly2022WaningEO",
        "Foo2019Bar",
        "Bar2000Foo pg 1-3",
        "Far2000 pg 2-5",
    }
    assert get_citenames(text) == ref


def test_single_author():
    text = "This was first proposed by (Smith 1999)."
    assert strip_citations(text) == "This was first proposed by ."


def test_multiple_authors():
    text = "Recent studies (Smith et al. 1999) show that this is true."
    assert strip_citations(text) == "Recent studies  show that this is true."


def test_multiple_citations():
    text = "As discussed by several authors (Smith et al. 1999; Johnson 2001; Lee et al. 2003)."
    assert strip_citations(text) == "As discussed by several authors ."


def test_citations_with_pages():
    text = "This is shown in (Smith et al. 1999, p. 150)."
    assert strip_citations(text) == "This is shown in ."


def test_citations_without_space():
    text = "Findings by(Smith et al. 1999)were significant."
    assert strip_citations(text) == "Findings bywere significant."


def test_citations_with_commas():
    text = "The method was adopted by (Smith, 1999, 2001; Johnson, 2002)."
    assert strip_citations(text) == "The method was adopted by ."


def test_citations_with_text():
    text = "This was noted (see Smith, 1999, for a review)."
    assert strip_citations(text) == "This was noted ."


def test_no_citations():
    text = "There are no references in this text."
    assert strip_citations(text) == "There are no references in this text."


def test_malformed_citations():
    text = "This is a malformed citation (Smith 199)."
    assert strip_citations(text) == "This is a malformed citation (Smith 199)."


def test_edge_case_citations():
    text = "Edge cases like (Smith et al.1999) should be handled."
    assert strip_citations(text) == "Edge cases like  should be handled."


def test_citations_with_special_characters():
    text = "Some names have dashes (O'Neil et al. 2000; Smith-Jones 1998)."
    assert strip_citations(text) == "Some names have dashes ."


def test_citations_with_nonstandard_chars():
    text = (
        "In non-English languages, citations might look different (Müller et al. 1999)."
    )
    assert (
        strip_citations(text)
        == "In non-English languages, citations might look different ."
    )


def test_ablations():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(tests_dir, "paper.pdf")
    with open(doc_path, "rb") as f:
        docs = Docs(prompts=PromptCollection(skip_summary=True))
        docs.add_file(f, "Wellawatte et al, XAI Review, 2023")
        answer = docs.get_evidence(
            Answer(
                question="Which page is the statement 'Deep learning (DL) is advancing the boundaries of computational"
                + "chemistry because it can accurately model non-linear structure-function relationships.' on?"
            )
        )
        assert (
            answer.contexts[0].text.text == answer.contexts[0].context
        ), "summarization not ablated"
        answer = docs.get_evidence(
            Answer(
                question="Which page is the statement 'Deep learning (DL) is advancing the boundaries of computational"
                + "chemistry because it can accurately model non-linear structure-function relationships.' on?"
            ),
            disable_vector_search=True,
        )


def test_location_awareness():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(tests_dir, "paper.pdf")
    with open(doc_path, "rb") as f:
        docs = Docs()
        docs.add_file(f, "Wellawatte et al, XAI Review, 2023")
        answer = docs.get_evidence(
            Answer(
                question="Which page is the statement 'Deep learning (DL) is advancing the boundaries of computational"
                + "chemistry because it can accurately model non-linear structure-function relationships.' on?"
            ),
            detailed_citations=True,
        )
        assert "2" in answer.context or "1" in answer.context


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

    sample = """
    Introduce dynamic elements such as moving nodes or edges to create a sense of activity within
    the network. 2. Add more nodes and connections to make the network
    appear more complex and interconnected. 3. Incorporate both red and
    green colors into the network, as the current screenshot only shows
    green lines. 4. Vary the thickness of the lines to add depth and
    visual interest. 5. Implement different shades of red and green to
    create a gradient effect for a more visually appealing experience.
    6. Consider adding a background color or pattern to enhance the
    contrast and make the network stand out. 7. Introduce interactive
    elements that allow users to manipulate the network, such as
    dragging nodes or zooming in/out. 8. Use animation effects like
    pulsing or changing colors to highlight certain parts of the network
    or to show activity. 9. Add labels or markers to provide information
      about the nodes or connections, if relevant to the purpose of the
        network visualization. 10. Consider the use of algorithms that
        organize the network in a visually appealing manner, such as
        force-directed placement or hierarchical layouts. 3/10 """

    assert get_score(sample) == 3

    sample = (
        "The text mentions a work by Shozo Yokoyama titled "
        '"Evolution of Dim-Light and Color Vision Pigments". '
        "This work, published in the Annual Review of Genomics and "
        "Human Genetics, discusses the evolution of human color vision. "
        "However, the text does not provide specific details or findings "
        "from Yokoyama's work. \n"
        "Relevance Score: 7"
    )

    assert get_score(sample) == 7

    sample = (
        "The evolution of human color vision is "
        "closely tied to theories about the nature "
        "of light, dating back to the 17th to 19th "
        "centuries. Initially, there was no clear distinction "
        "between the properties of light, the eye and retina, "
        "and color percepts. Major figures in science attempted "
        "to resolve these issues, with physicists leading most "
        "advances in color science into the 20th century. Prior "
        "to Newton, colors were viewed as stages between black "
        "and white. Newton was the first to describe colors in "
        "a modern sense, using prisms to disperse light into "
        "a spectrum of colors. He demonstrated that each color "
        "band could not be further divided and that different "
        "colors had different refrangibility. \n"
        "Relevance Score: 9.5"
    )

    assert get_score(sample) == 9


class TestChains(IsolatedAsyncioTestCase):
    async def test_chain_completion(self):
        client = AsyncOpenAI()
        llm = OpenAILLMModel(config=dict(model="babbage-002", temperature=0.2))
        call = llm.make_chain(
            client,
            "The {animal} says",
            skip_system=True,
        )
        outputs = []

        def accum(x):
            outputs.append(x)

        completion = await call(dict(animal="duck"), callbacks=[accum])  # type: ignore[call-arg]
        assert completion.seconds_to_first_token > 0
        assert completion.prompt_count > 0
        assert completion.completion_count > 0
        assert str(completion) == "".join(outputs)

        completion = await call(dict(animal="duck"))  # type: ignore[call-arg]
        assert completion.seconds_to_first_token == 0
        assert completion.seconds_to_last_token > 0

    async def test_chain_chat(self):
        client = AsyncOpenAI()
        llm = OpenAILLMModel(
            config=dict(temperature=0, model="gpt-3.5-turbo", max_tokens=56)
        )
        call = llm.make_chain(
            client,
            "The {animal} says",
            skip_system=True,
        )
        outputs = []

        def accum(x):
            outputs.append(x)

        completion = await call(dict(animal="duck"), callbacks=[accum])  # type: ignore[call-arg]
        assert completion.seconds_to_first_token > 0
        assert completion.prompt_count > 0
        assert completion.completion_count > 0
        assert str(completion) == "".join(outputs)

        completion = await call(dict(animal="duck"))  # type: ignore[call-arg]
        assert completion.seconds_to_first_token == 0
        assert completion.seconds_to_last_token > 0

        # check with mixed callbacks
        async def ac(x):
            pass

        completion = await call(dict(animal="duck"), callbacks=[accum, ac])  # type: ignore[call-arg]


def test_docs():
    docs = Docs(llm="babbage-002")
    docs.add_url(
        "https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    assert docs.docs["test"].docname == "Wiki2023"
    assert docs.llm == "babbage-002"
    assert docs.summary_llm == "babbage-002"


def test_evidence():
    doc_path = "example.html"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
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


def test_json_evidence():
    doc_path = "example.html"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    summary_llm = OpenAILLMModel(
        config=dict(
            model="gpt-3.5-turbo-1106",
            response_format=dict(type="json_object"),
            temperature=0.0,
        )
    )
    docs = Docs(
        prompts=PromptCollection(json_summary=True),
        summary_llm_model=summary_llm,
        llm_result_callback=print_callback,
    )
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
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


def test_custom_json_props():
    doc_path = "example.html"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    summary_llm = OpenAILLMModel(
        config=dict(
            model="gpt-3.5-turbo-0125",
            response_format=dict(type="json_object"),
            temperature=0.0,
        )
    )
    my_prompts = PromptCollection(
        json_summary=True,
        summary_json_system="Provide a summary of the excerpt that could help answer the question based on the excerpt.  "
        "The excerpt may be irrelevant. Do not directly answer the question - only summarize relevant information. "
        "Respond with the following JSON format:\n\n"
        '{{\n"summary": "...",\n"person_name": "...",\n"relevance_score": "..."}}\n\n'
        "where `summary` is relevant information from text - "
        "about 100 words words, `person_name` specifies the person discussed in "
        "the excerpt (may be different than query), and `relevance_score` is "
        "the relevance of `summary` to answer the question (integer out of 10).",
    )
    docs = Docs(
        prompts=my_prompts,
        summary_llm_model=summary_llm,
        llm_result_callback=print_callback,
    )
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
    evidence = docs.get_evidence(
        Answer(question="For which state was Bates a governor?"), k=1, max_sources=1
    )
    assert "person_name" in evidence.contexts[0].model_extra
    assert "person_name: " in evidence.context
    print(evidence.context)
    answer = docs.query("What is Frederick Bates's greatest accomplishment?")
    assert "person_name" in answer.context
    os.remove(doc_path)


def test_query():
    docs = Docs()
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    docs.query("What is Frederick Bates's greatest accomplishment?")


def test_answer_attributes():
    docs = Docs()
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    answer = docs.query("What is Frederick Bates's greatest accomplishment?")
    used_citations = answer.used_contexts
    assert len(used_citations) > 0
    assert len(used_citations) < len(answer.contexts)
    assert (
        answer.get_citation(list(used_citations)[0])
        == "WikiMedia Foundation, 2023, Accessed now"
    )

    # make sure it is serialized correctly
    js = answer.model_dump_json()
    assert "used_contexts" in js

    # make sure it can be deserialized
    answer2 = Answer.model_validate_json(js)
    assert answer2.used_contexts == used_citations


def test_llmresult_callbacks():
    my_results = []

    async def my_callback(result):
        my_results.append(result)

    docs = Docs(llm_result_callback=my_callback)
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    docs.query("What is Frederick Bates's greatest accomplishment?")
    assert any([x.name == "answer" for x in my_results])
    assert len(my_results) > 1


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


def test_custom_embedding():
    class MyEmbeds(EmbeddingModel):
        name: str = "my_embed"

        async def embed_documents(self, client, texts):
            return [[1, 2, 3] for _ in texts]

    docs = Docs(
        docs_index=NumpyVectorStore(embedding_model=MyEmbeds()),
        texts_index=NumpyVectorStore(embedding_model=MyEmbeds()),
        embedding_client=None,
    )
    assert docs._embedding_client is None
    assert docs.embedding == "my_embed"
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    assert docs.docs["test"].embedding == [1, 2, 3]


def test_sentence_transformer_embedding():
    from paperqa import SentenceTransformerEmbeddingModel

    docs = Docs(embedding="sentence-transformers")
    assert docs._embedding_client is None
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    assert any(docs.docs["test"].embedding)  # type: ignore[arg-type]

    docs = Docs(
        texts_index=NumpyVectorStore(
            embedding_model=SentenceTransformerEmbeddingModel()
        ),
        doc_index=NumpyVectorStore(embedding_model=SentenceTransformerEmbeddingModel()),
        embedding_client=None,
    )
    assert docs._embedding_client is None
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    assert any(docs.docs["test"].embedding)  # type: ignore[arg-type]


def test_custom_llm():
    class MyLLM(LLMModel):
        name: str = "myllm"

        async def acomplete(self, client, prompt):
            assert client is None
            return "Echo"

    docs = Docs(llm_model=MyLLM(), client=None)
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    evidence = docs.get_evidence(Answer(question="Echo"))
    assert "Echo" in evidence.context


def test_custom_llm_stream():
    class MyLLM(LLMModel):
        name: str = "myllm"

        async def acomplete_iter(self, client, prompt):
            assert client is None
            yield "Echo"

    docs = Docs(llm_model=MyLLM(), client=None)
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    evidence = docs.get_evidence(
        Answer(question="Echo"), get_callbacks=lambda x: [lambda y: print(y, end="")]
    )
    assert "Echo" in evidence.context


def test_langchain_llm():
    from langchain_openai import ChatOpenAI, OpenAI

    docs = Docs(llm="langchain", client=ChatOpenAI(model="gpt-3.5-turbo"))
    assert type(docs.llm_model) == LangchainLLMModel
    assert type(docs.summary_llm_model) == LangchainLLMModel
    assert docs.llm == "gpt-3.5-turbo"
    assert docs.summary_llm == "gpt-3.5-turbo"
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    assert docs._client is not None
    assert type(docs.llm_model) == LangchainLLMModel
    assert docs.summary_llm_model == docs.llm_model

    docs.get_evidence(
        Answer(question="What is Frederick Bates's greatest accomplishment?"),
        get_callbacks=lambda x: [lambda y: print(y, end="")],
    )

    assert docs.llm_model.llm_type == "chat"

    # trying without callbacks (different codepath)
    docs.get_evidence(
        Answer(question="What is Frederick Bates's greatest accomplishment?")
    )

    # now completion

    docs = Docs(llm_model=LangchainLLMModel(), client=OpenAI(model="babbage-002"))
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    docs.get_evidence(
        Answer(question="What is Frederick Bates's greatest accomplishment?"),
        get_callbacks=lambda x: [lambda y: print(y, end="")],
    )

    assert docs.summary_llm_model.llm_type == "completion"  # type: ignore[union-attr]

    # trying without callbacks (different codepath)
    docs.get_evidence(
        Answer(question="What is Frederick Bates's greatest accomplishment?")
    )

    # now make sure we can pickle it
    docs_pickle = pickle.dumps(docs)
    docs2 = pickle.loads(docs_pickle)
    assert docs2._client is None
    assert docs2.llm == "babbage-002"
    docs2.set_client(OpenAI(model="babbage-002"))
    assert docs2.summary_llm == "babbage-002"
    docs2.get_evidence(
        Answer(question="What is Frederick Bates's greatest accomplishment?"),
        get_callbacks=lambda x: [lambda y: print(y)],
    )


def test_langchain_embeddings():
    from langchain_openai import OpenAIEmbeddings

    docs = Docs(
        texts_index=NumpyVectorStore(embedding_model=LangchainEmbeddingModel()),
        docs_index=NumpyVectorStore(embedding_model=LangchainEmbeddingModel()),
        embedding_client=OpenAIEmbeddings(),
    )
    assert docs._embedding_client is not None

    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )
    docs = Docs(embedding="langchain", embedding_client=OpenAIEmbeddings())
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )


class TestVectorStore(IsolatedAsyncioTestCase):
    async def test_langchain_vector_store(self):
        from langchain_community.vectorstores.faiss import FAISS
        from langchain_openai import OpenAIEmbeddings

        some_texts = [
            Text(
                embedding=OpenAIEmbeddings().embed_query("test"),
                text="this is a test",
                name="test",
                doc=Doc(docname="test", citation="test", dockey="test"),
            )
        ]

        # checks on builder
        try:
            index = LangchainVectorStore()
            index.add_texts_and_embeddings(some_texts)
            raise "Failed to check for builder"  # type: ignore[misc]
        except ValueError:
            pass

        try:
            index = LangchainVectorStore(store_builder=lambda x: None)
            raise "Failed to count arguments"  # type: ignore[misc]
        except ValueError:
            pass

        try:
            index = LangchainVectorStore(store_builder="foo")
            raise "Failed to check if builder is callable"  # type: ignore[misc]
        except ValueError:
            pass

        # now with real builder
        index = LangchainVectorStore(
            store_builder=lambda x, y: FAISS.from_embeddings(x, OpenAIEmbeddings(), y)
        )
        assert index._store is None
        index.add_texts_and_embeddings(some_texts)
        assert index._store is not None
        # check search returns Text obj
        data, score = await index.similarity_search(None, "test", k=1)  # type: ignore[unreachable]
        print(data)
        assert type(data[0]) == Text

        # now try with convenience
        index = LangchainVectorStore(cls=FAISS, embedding_model=OpenAIEmbeddings())
        assert index._store is None
        index.add_texts_and_embeddings(some_texts)
        assert index._store is not None

        docs = Docs(
            texts_index=LangchainVectorStore(
                cls=FAISS, embedding_model=OpenAIEmbeddings()
            )
        )
        assert docs._embedding_client is not None  # from docs_index default

        await docs.aadd_url(
            "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
            citation="WikiMedia Foundation, 2023, Accessed now",
            dockey="test",
        )
        # should be embedded

        # now try with JIT
        docs = Docs(texts_index=index, jit_texts_index=True)
        await docs.aadd_url(
            "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
            citation="WikiMedia Foundation, 2023, Accessed now",
            dockey="test",
        )
        # should get cleared and rebuilt here
        ev = await docs.aget_evidence(
            answer=Answer(question="What is Frederick Bates's greatest accomplishment?")
        )
        assert len(ev.context) > 0
        # now with dockkey filter
        await docs.aget_evidence(
            answer=Answer(
                question="What is Frederick Bates's greatest accomplishment?",
                dockey_filter=["test"],
            )
        )

        # make sure we can pickle it
        docs_pickle = pickle.dumps(docs)
        pickle.loads(docs_pickle)

        # will not work at this point - have to reset index


class Test(IsolatedAsyncioTestCase):
    async def test_aquery(self):
        docs = Docs()
        await docs.aadd_url(
            "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
            citation="WikiMedia Foundation, 2023, Accessed now",
            dockey="test",
        )
        await docs.aquery("What is Frederick Bates's greatest accomplishment?")


class TestDocMatch(IsolatedAsyncioTestCase):
    async def test_adoc_match(self):
        docs = Docs()
        await docs.aadd_url(
            "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
            citation="WikiMedia Foundation, 2023, Accessed now",
            dockey="test",
        )
        sources = await docs.adoc_match(
            "What is Frederick Bates's greatest accomplishment?"
        )
        assert len(sources) > 0
        sources = await docs.adoc_match(
            "What is Frederick Bates's greatest accomplishment?"
        )
        assert len(sources) > 0


def test_docs_pickle():
    doc_path = "example.html"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get front page of wikipedia
        r = requests.get("https://en.wikipedia.org/wiki/Take_Your_Dog_to_Work_Day")
        f.write(r.text)
    docs = Docs(
        llm_model=OpenAILLMModel(config=dict(temperature=0.0, model="gpt-3.5-turbo"))
    )
    assert docs._client is not None
    old_config = docs.llm_model.config
    old_sconfig = docs.summary_llm_model.config  # type: ignore[union-attr]
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now", chunk_chars=1000)  # type: ignore[arg-type]
    os.remove(doc_path)
    docs_pickle = pickle.dumps(docs)
    docs2 = pickle.loads(docs_pickle)
    # make sure it fails if we haven't set client
    try:
        docs2.query("What date is bring your dog to work in the US?")
    except ValueError:
        pass
    docs2.set_client()
    assert docs2._client is not None
    assert docs2.llm_model.config == old_config
    assert docs2.summary_llm_model.config == old_sconfig
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

    # make sure we can embed documents
    docs2.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
    )


def test_bad_context():
    doc_path = "example.html"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
    answer = docs.query("What is the radius of Jupyter?")
    assert "cannot answer" in answer.answer
    os.remove(doc_path)


def test_repeat_keys():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs(
        llm_model=OpenAILLMModel(config=dict(temperature=0.0, model="babbage-002"))
    )
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
    try:
        docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
    except ValueError:
        pass
    assert len(docs.docs) == 1

    # now with different paths
    doc_path2 = "example2.txt"
    with open(doc_path2, "w", encoding="utf-8") as f:
        # get wiki page about politician
        f.write(r.text)
        f.write("\n")  # so we don't have same hash
    docs.add(doc_path2, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
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
    docs = Docs(llm_model=OpenAILLMModel(config=dict(temperature=0.0, model="gpt-4")))
    docs.add(doc_path, "Wellawatte et al, XAI Review, 2023")  # type: ignore[arg-type]
    answer = docs.query("Are counterfactuals actionable? [yes/no]")
    assert "yes" in answer.answer or "Yes" in answer.answer


def test_fileio_reader_pdf():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(tests_dir, "paper.pdf")
    with open(doc_path, "rb") as f:
        docs = Docs()
        docs.add_file(f, "Wellawatte et al, XAI Review, 2023")
        answer = docs.query("Are counterfactuals actionable?[yes/no]")
        assert "yes" in answer.answer or "Yes" in answer.answer


def test_fileio_reader_txt():
    # can't use curie, because it has trouble with parsed HTML
    docs = Docs()
    r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
    if r.status_code != 200:
        raise ValueError("Could not download wikipedia page")
    docs.add_file(
        BytesIO(r.text.encode()),
        "WikiMedia Foundation, 2023, Accessed now",
        chunk_chars=1000,
    )
    answer = docs.query("What country was Frederick Bates born in?")
    assert "United States" in answer.answer


def test_pdf_pypdf_reader():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(tests_dir, "paper.pdf")
    splits1 = read_doc(
        doc_path,  # type: ignore[arg-type]
        Doc(docname="foo", citation="Foo et al, 2002", dockey="1"),
        force_pypdf=True,
        overlap=100,
        chunk_chars=3000,
    )
    splits2 = read_doc(
        doc_path,  # type: ignore[arg-type]
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
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
    docs.query("What is the name of the politician?")


def test_code():
    # load this script
    doc_path = os.path.abspath(__file__)
    docs = Docs(
        llm_model=OpenAILLMModel(config=dict(temperature=0.0, model="babbage-002"))
    )
    docs.add(doc_path, "test_paperqa.py", docname="test_paperqa.py", disable_check=True)  # type: ignore[arg-type]
    assert len(docs.docs) == 1
    docs.query("What function tests the preview?")


def test_citation():
    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs()
    docs.add(doc_path)  # type: ignore[arg-type]
    assert (
        list(docs.docs.values())[0].docname == "Wikipedia2024"
        or list(docs.docs.values())[0].docname == "Frederick2024"
        or list(docs.docs.values())[0].docname == "Wikipedia"
        or list(docs.docs.values())[0].docname == "Frederick"
    )


def test_dockey_filter():
    """Test that we can filter evidence with dockeys"""
    doc_path = "example2.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
    # add with new dockey
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(r.text)
        f.write("\n")  # so we don't have same hash
    docs.add("example.txt", "WikiMedia Foundation, 2023, Accessed now", dockey="test")  # type: ignore[arg-type]
    answer = Answer(question="What country is Bates from?", dockey_filter=["test"])
    docs.get_evidence(answer)


def test_dockey_delete():
    """Test that we can filter evidence with dockeys"""
    doc_path = "example2.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs = Docs()
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
    # add with new dockey
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(r.text)
        f.write("\n\nBates could be from Angola")  # so we don't have same hash
    docs.add("example.txt", "WikiMedia Foundation, 2023, Accessed now", docname="test")  # type: ignore[arg-type]
    answer = Answer(question="What country was Bates born in?")
    answer = docs.get_evidence(
        answer, max_sources=25, k=30
    )  # we just have a lot so we get both docs
    keys = set([c.text.doc.dockey for c in answer.contexts])
    assert len(keys) == 2
    assert len(docs.docs) == 2

    docs.delete(docname="test")
    answer = Answer(question="What country was Bates born in?")
    assert len(docs.docs) == 1
    assert len(docs.deleted_dockeys) == 1
    answer = docs.get_evidence(answer, max_sources=25, k=30)
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
        doc_path,  # type: ignore[arg-type]
        "Information about Fredrick Bates, WikiMedia Foundation, 2023, Accessed now",
    )
    # add with new dockey
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(r.text)
        f.write("\n")  # so we don't have same hash
    docs.add("example.txt", "WikiMedia Foundation, 2023, Accessed now", dockey="test")  # type: ignore[arg-type]
    docs.query("What country is Bates from?", key_filter=True)
    # the filter shouldn't trigger, so just checking that it doesn't crash


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
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
    # add with new dockey
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(r.text)
        f.write("\n")  # so we don't have same hash
    docs.add(
        "example.txt",  # type: ignore[arg-type]
        "WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
        chunk_chars=4000,
    )
    docs.query("What is Barrack's greatest accomplishment?", max_sources=10, k=10)


def test_custom_prompts():
    my_qaprompt = (
        "Answer the question '{question}' "
        "using the country name alone. For example: "
        "A: United States\nA: Canada\nA: Mexico\n\n Using the context:\n\n{context}\n\nA: "
    )

    docs = Docs(prompts=PromptCollection(qa=my_qaprompt))

    doc_path = "example.html"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
    answer = docs.query("What country is Frederick Bates from?")
    assert "United States" in answer.answer


def test_pre_prompt():
    pre = "Provide context you have memorized " "that could help answer '{question}'. "

    docs = Docs(prompts=PromptCollection(pre=pre))

    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
    docs.query("What country is Bates from?")


def test_post_prompt():
    post = (
        "We are trying to answer the question below "
        "and have an answer provided. "
        "Please edit the answer be extremely terse, with no extra words or formatting"
        "with no extra information.\n\n"
        "Q: {question}\nA: {answer}\n\n"
    )

    docs = Docs(prompts=PromptCollection(post=post))

    doc_path = "example.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        # get wiki page about politician
        r = requests.get("https://en.wikipedia.org/wiki/Frederick_Bates_(politician)")
        f.write(r.text)
    docs.add(doc_path, "WikiMedia Foundation, 2023, Accessed now")  # type: ignore[arg-type]
    docs.query("What country is Bates from?")

    docs = Docs(
        prompts=PromptCollection(
            system="Answer all questions with as few words as possible"
        )
    )
    docs.query("What country is Bates from?")


def disabled_test_memory():
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
    answer2 = docs.query("When was the conflict resolved?")
    assert "1941" in answer2.answer or "1945" in answer2.answer
    assert answer2.memory is not None
    assert "Answer" in docs.memory_model.load_memory_variables({})["memory"]
    print(answer2.answer)

    docs.clear_memory()

    answer3 = docs.query("When was the conflict resolved?")
    assert answer3.memory is not None
    assert (
        "I cannot answer" in answer3.answer
        or "insufficient" in answer3.answer
        or "does not provide" in answer3.answer
        or "ambiguous" in answer3.answer
    )


def test_add_texts():
    llm_config = dict(temperature=0.1, model="babbage-02")
    docs = Docs(llm_model=OpenAILLMModel(config=llm_config))
    docs.add_url(
        "https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test",
    )

    docs2 = Docs()
    texts = [Text(**dict(t)) for t in docs.texts]
    for t in texts:
        t.embedding = None
    docs2.add_texts(texts, list(docs.docs.values())[0])

    for t1, t2 in zip(docs2.texts, docs.texts):
        assert t1.text == t2.text
        assert np.allclose(t1.embedding, t2.embedding, atol=1e-3)

    docs2._build_texts_index()
    # now do it again to test after text index is already built
    llm_config = dict(temperature=0.1, model="babbage-02")
    docs = Docs(llm_model=OpenAILLMModel(config=llm_config))
    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="WikiMedia Foundation, 2023, Accessed now",
        dockey="test3",
    )

    texts = [Text(**dict(t)) for t in docs.texts]
    for t in texts:
        t.embedding = None
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
    docs2 = Docs(docs_index=docs.docs_index, texts_index=docs.texts_index)
    assert len(docs2.docs) == 0
    evidence = docs2.query("What is the date of flag day?", key_filter=True)
    assert "February 15" in evidence.context


def test_embedding_name_consistency():
    docs = Docs()
    assert docs.embedding == "text-embedding-ada-002"
    assert docs.texts_index.embedding_model.name == "text-embedding-ada-002"
    docs = Docs(embedding="langchain")
    assert docs.embedding == "langchain"
    assert docs.texts_index.embedding_model.name == "langchain"
    assert type(docs.texts_index.embedding_model) == LangchainEmbeddingModel
    docs = Docs(embedding="foo")
    assert docs.embedding == "foo"
    assert type(docs.texts_index.embedding_model) == OpenAIEmbeddingModel
    docs = Docs(
        texts_index=NumpyVectorStore(embedding_model=OpenAIEmbeddingModel(name="test"))
    )
    assert docs.embedding == "test"


def test_external_texts_index():
    docs = Docs(jit_texts_index=True)
    docs.add_url(
        "https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day",
        citation="Flag Day of Canada, WikiMedia Foundation, 2023, Accessed now",
    )
    answer = docs.query(query="On which date is flag day annually observed?")
    print(answer.model_dump())
    assert "February 15" in answer.answer

    docs.add_url(
        "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)",
        citation="Fredrick Bates, WikiMedia Foundation, 2023, Accessed now",
    )

    answer = docs.query(
        query="On which date is flag day annually observed?", key_filter=True
    )
    assert "February 15" in answer.answer
