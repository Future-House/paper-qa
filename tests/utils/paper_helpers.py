import re

def compare_formatted_citations(expected: str, actual: str) -> bool:
    """
    Compares two formatted citation strings; ignoring the citation_count value.

    :param expected: The expected formatted citation string.
    :param actual: The actual formatted citation string.
    :return: True if the citations match except for the citation count, False otherwise.
    """
    # https://regex101.com/r/lCN8ET/1
    citation_pattern = r"(This article has )\d+( citations?)"

    # between group 1 and 2, replace with the character "n"
    expected_cleaned = re.sub(citation_pattern, r"\1n\2", expected).strip()
    actual_cleaned = re.sub(citation_pattern, r"\1n\2", actual).strip()

    return expected_cleaned == actual_cleaned
