from paperqa.utils import citation_to_docname


def test_citation_to_docname_acronym_title() -> None:
    citation = "CD47/SIRPα axis: bridging innate and adaptive immunity, 2022"
    assert citation_to_docname(citation) == "CD472022"


def test_citation_to_docname_non_text_fallback_is_deterministic() -> None:
    citation = "___, n.d."
    first = citation_to_docname(citation)
    second = citation_to_docname(citation)

    assert first == second
    assert first.startswith("Doc")
