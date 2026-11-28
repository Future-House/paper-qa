from __future__ import annotations

import pytest

from paperqa.contrib.retraction import (
    LANDMARK_RETRACTED_DOIS,
    RetractionFilter,
    is_retracted_doi,
)
from paperqa.docs import Docs
from paperqa.settings import Settings
from paperqa.types import DocDetails, Text

WAKEFIELD_DOI = "10.1016/S0140-6736(97)11096-0"
SURGISPHERE_DOI = "10.1016/S0140-6736(20)31180-6"
VALID_DOI = "10.1038/s41586-020-2649-2"


def test_is_retracted_doi_landmarks() -> None:
    assert WAKEFIELD_DOI.lower() in LANDMARK_RETRACTED_DOIS
    assert SURGISPHERE_DOI.lower() in LANDMARK_RETRACTED_DOIS

    # Wakefield 1998 Lancet
    assert is_retracted_doi(WAKEFIELD_DOI)
    assert is_retracted_doi(WAKEFIELD_DOI.lower())
    assert is_retracted_doi(f"https://doi.org/{WAKEFIELD_DOI}")
    assert is_retracted_doi(f"http://dx.doi.org/{WAKEFIELD_DOI}")

    # Surgisphere 2020 Lancet COVID-19
    assert is_retracted_doi(SURGISPHERE_DOI)
    assert is_retracted_doi(f"https://doi.org/{SURGISPHERE_DOI}")

    # Non-retracted valid DOI
    assert not is_retracted_doi(VALID_DOI)


def test_is_retracted_doi_custom_and_hooks() -> None:
    custom_doi = "10.1234/custom-retracted-paper"
    assert not is_retracted_doi(custom_doi)
    assert is_retracted_doi(custom_doi, custom_dois={custom_doi})

    # Callable hook
    hook_called = False

    def custom_hook(doi: str) -> bool:
        nonlocal hook_called
        hook_called = True
        return "flagged" in doi

    assert is_retracted_doi("10.5555/flagged-paper", hook=custom_hook)
    assert hook_called
    assert not is_retracted_doi("10.5555/clean-paper", hook=custom_hook)


def test_retraction_filter_class() -> None:
    rf = RetractionFilter(action="drop")
    doc_retracted = DocDetails(
        docname="wakefield1998",
        dockey="wakefield1998",
        citation="Wakefield et al. Lancet 1998",
        doi=WAKEFIELD_DOI,
    )
    doc_clean = DocDetails(
        docname="clean2020",
        dockey="clean2020",
        citation="Clean Author et al. 2020",
        doi=VALID_DOI,
    )

    text_retracted = Text(
        text="Wakefield claim text.", name="wakefield1998-0", doc=doc_retracted
    )
    text_clean = Text(text="Clean study text.", name="clean2020-0", doc=doc_clean)

    assert rf.is_retracted(WAKEFIELD_DOI)
    assert rf.is_retracted(doc_retracted)
    assert rf.is_retracted(text_retracted)
    assert not rf.is_retracted(VALID_DOI)
    assert not rf.is_retracted(doc_clean)
    assert not rf.is_retracted(text_clean)

    # Filter with action="drop"
    dropped = rf.filter_texts([text_retracted, text_clean], action="drop")
    assert len(dropped) == 1
    assert dropped[0].name == text_clean.name

    # Filter with action="warn"
    warned = rf.filter_texts([text_retracted, text_clean], action="warn")
    assert len(warned) == 2
    warned_retracted = next(t for t in warned if t.name == text_retracted.name)
    assert "[RETRACTED PAPER WARNING]" in warned_retracted.text
    assert getattr(warned_retracted, "is_retracted", False) is True


@pytest.mark.asyncio
async def test_evidence_retrieval_retraction_action_drop() -> None:
    doc_retracted = DocDetails(
        docname="wakefield1998",
        dockey="wakefield1998",
        citation="Wakefield et al. Lancet 1998",
        doi=WAKEFIELD_DOI,
    )
    doc_clean = DocDetails(
        docname="clean2020",
        dockey="clean2020",
        citation="Clean Author et al. 2020",
        doi=VALID_DOI,
    )
    text_retracted = Text(
        text="Discredited claim about vaccines and autism.",
        name="wakefield1998-0",
        doc=doc_retracted,
    )
    text_clean = Text(
        text="Rigorous evidence that MMR vaccine does not cause autism.",
        name="clean2020-0",
        doc=doc_clean,
    )

    docs = Docs(
        docs={
            doc_retracted.dockey: doc_retracted,
            doc_clean.dockey: doc_clean,
        },
        texts=[text_retracted, text_clean],
        docnames={doc_retracted.docname, doc_clean.docname},
    )

    settings = Settings()
    settings.answer.evidence_skip_summary = True
    settings.answer.evidence_retrieval = False
    settings.evidence_retraction_filter = True
    settings.retraction_action = "drop"

    session = await docs.aget_evidence(
        "Does MMR vaccine cause autism?", settings=settings
    )
    assert len(session.contexts) == 1
    assert session.contexts[0].text.name == text_clean.name
    assert all(c.text.name != text_retracted.name for c in session.contexts)


@pytest.mark.asyncio
async def test_evidence_retrieval_retraction_action_warn() -> None:
    doc_retracted = DocDetails(
        docname="surgisphere2020",
        dockey="surgisphere2020",
        citation="Mehra et al. Lancet 2020",
        doi=SURGISPHERE_DOI,
    )
    doc_clean = DocDetails(
        docname="clean2020",
        dockey="clean2020",
        citation="Clean Author et al. 2020",
        doi=VALID_DOI,
    )
    text_retracted = Text(
        text="Surgisphere fabricated hospital data.",
        name="surgisphere2020-0",
        doc=doc_retracted,
    )
    text_clean = Text(
        text="Properly verified clinical data.",
        name="clean2020-0",
        doc=doc_clean,
    )

    docs = Docs(
        docs={
            doc_retracted.dockey: doc_retracted,
            doc_clean.dockey: doc_clean,
        },
        texts=[text_retracted, text_clean],
        docnames={doc_retracted.docname, doc_clean.docname},
    )

    settings = Settings()
    settings.answer.evidence_skip_summary = True
    settings.answer.evidence_retrieval = False
    settings.evidence_retraction_filter = True
    settings.retraction_action = "warn"

    session = await docs.aget_evidence("Clinical outcomes question", settings=settings)
    assert len(session.contexts) == 2

    retracted_context = next(
        c for c in session.contexts if c.text.name == text_retracted.name
    )
    assert "[RETRACTED PAPER WARNING]" in retracted_context.context
    assert "[RETRACTED PAPER WARNING]" in retracted_context.text.text
    assert getattr(retracted_context, "is_retracted", False) is True
    assert (
        getattr(retracted_context, "retraction_warning", None)
        == "[RETRACTED PAPER WARNING]"
    )


@pytest.mark.asyncio
async def test_evidence_retrieval_retraction_filter_disabled_by_default() -> None:
    doc_retracted = DocDetails(
        docname="wakefield1998",
        dockey="wakefield1998",
        citation="Wakefield et al. Lancet 1998",
        doi=WAKEFIELD_DOI,
    )
    text_retracted = Text(
        text="Discredited claim about vaccines and autism.",
        name="wakefield1998-0",
        doc=doc_retracted,
    )

    docs = Docs(
        docs={doc_retracted.dockey: doc_retracted},
        texts=[text_retracted],
        docnames={doc_retracted.docname},
    )

    settings = Settings()
    settings.answer.evidence_skip_summary = True
    settings.answer.evidence_retrieval = False

    session = await docs.aget_evidence(
        "Does MMR vaccine cause autism?", settings=settings
    )
    assert len(session.contexts) == 1
    assert "[RETRACTED PAPER WARNING]" not in session.contexts[0].context
    assert getattr(session.contexts[0], "is_retracted", None) is None
