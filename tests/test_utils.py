from paperqa.utils import citation_to_docname


def test_citation_to_docname_acronym_title() -> None:
    citation = "CD47/SIRP\u03b1 axis: bridging innate and adaptive immunity, 2022"
    assert citation_to_docname(citation) == "CD472022"


def test_citation_to_docname_non_text_fallback_is_deterministic() -> None:
    # Contrived edge-case input chosen to guarantee the final fallback branch:
    # - no TitleCase token (e.g., "Smith")
    # - no acronym token (e.g., "CD47")
    # This models malformed/placeholder citation text from extraction failures.
    citation = "___, n.d."
    first = citation_to_docname(citation)
    second = citation_to_docname(citation)

    assert first == second, (
        "Expected deterministic fallback: identical malformed input should yield the "
        "same docname each call (regression guard against random/UUID-based suffixes)."
    )
    assert first.startswith("Doc"), (
        "Expected malformed citations to use the explicit final fallback format "
        "'Doc<hash8>' (for example, Docc7acc74a)."
    )
