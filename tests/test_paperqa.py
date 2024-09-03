from __future__ import annotations

import contextlib
import os
import pickle
import tempfile
import textwrap
from io import BytesIO
from pathlib import Path
from typing import cast, no_type_check

import numpy as np
import pytest
import requests
from openai import AsyncOpenAI

from paperqa import (
    Answer,
    Doc,
    Docs,
    NumpyVectorStore,
    Settings,
    Text,
    get_settings,
    print_callback,
)
from paperqa.clients import CrossrefProvider
from paperqa.core import llm_parse_json
from paperqa.llms import (
    AnthropicLLMModel,
    EmbeddingModel,
    HybridEmbeddingModel,
    LangchainEmbeddingModel,
    LangchainLLMModel,
    LangchainVectorStore,
    LLMModel,
    OpenAIEmbeddingModel,
    OpenAILLMModel,
    SparseEmbeddingModel,
    VoyageAIEmbeddingModel,
    guess_model_type,
    is_openai_model,
)
from paperqa.readers import read_doc
from paperqa.utils import (
    get_citenames,
    get_score,
    maybe_is_html,
    maybe_is_text,
    name_in_text,
    strings_similarity,
    strip_citations,
)


def test_is_openai_model():
    assert is_openai_model("gpt-4o-mini")
    assert is_openai_model("babbage-002")
    assert is_openai_model("gpt-4-1106-preview")
    assert is_openai_model("davinci-002")
    assert is_openai_model("ft:gpt-3.5-turbo-0125:my-org::ABC123")
    assert not is_openai_model("llama")
    assert not is_openai_model("labgpt")
    assert not is_openai_model("mixtral-7B")
    os.environ["ANYSCALE_API_KEY"] = "abc123"
    os.environ["ANYSCALE_BASE_URL"] = "https://example.com"
    assert is_openai_model("meta-llama/Meta-Llama-3-70B-Instruct")
    assert is_openai_model("mistralai/Mixtral-8x22B-Instruct-v0.1")
    os.environ.pop("ANYSCALE_API_KEY")
    os.environ.pop("ANYSCALE_BASE_URL")
    assert not is_openai_model("meta-llama/Meta-Llama-3-70B-Instruct")
    assert not is_openai_model("mistralai/Mixtral-8x22B-Instruct-v0.1")


def test_guess_model_type():
    assert guess_model_type("gpt-3.5-turbo") == "chat"
    assert guess_model_type("babbage-002") == "completion"
    assert guess_model_type("gpt-4-1106-preview") == "chat"
    assert guess_model_type("gpt-3.5-turbo-instruct") == "completion"
    assert guess_model_type("davinci-002") == "completion"
    os.environ["ANYSCALE_API_KEY"] = "abc123"
    os.environ["ANYSCALE_BASE_URL"] = "https://example.com"
    assert guess_model_type("meta-llama/Meta-Llama-3-70B-Instruct") == "chat"
    assert guess_model_type("mistralai/Mixtral-8x22B-Instruct-v0.1") == "chat"
    os.environ.pop("ANYSCALE_API_KEY")
    os.environ.pop("ANYSCALE_BASE_URL")


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
    settings = Settings()
    settings.prompts.skip_summary = True
    settings.answer.evidence_retrieval = False
    with open(doc_path, "rb") as f:
        docs = Docs()
        docs.add_file(f, "Wellawatte et al, XAI Review, 2023")
        contexts = docs.get_evidence(
            "Which page is the statement 'Deep learning (DL) is advancing the boundaries of computational"
            + " chemistry because it can accurately model non-linear structure-function relationships.' on?",
            settings=settings,
        )
        assert contexts[0].text.text == contexts[0].context, "summarization not ablated"

        assert len(contexts) == len(docs.texts), "evidence retrieval not ablated"
