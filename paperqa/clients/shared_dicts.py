BIBTEX_MAPPING: dict[str, str] = {
    "journal-article": "article",
    "journal-issue": "misc",  # No direct equivalent, so 'misc' is used
    "journal-volume": "misc",  # No direct equivalent, so 'misc' is used
    "journal": "misc",  # No direct equivalent, so 'misc' is used
    "proceedings-article": "inproceedings",
    "proceedings": "proceedings",
    "dataset": "misc",  # No direct equivalent, so 'misc' is used
    "component": "misc",  # No direct equivalent, so 'misc' is used
    "report": "techreport",
    "report-series": "techreport",  # 'series' implies multiple tech reports, but each is still a 'techreport'
    "standard": "misc",  # No direct equivalent, so 'misc' is used
    "standard-series": "misc",  # No direct equivalent, so 'misc' is used
    "edited-book": "book",  # Edited books are considered books in BibTeX
    "monograph": "book",  # Monographs are considered books in BibTeX
    "reference-book": "book",  # Reference books are considered books in BibTeX
    "book": "book",
    "book-series": "book",  # Series of books can be considered as 'book' in BibTeX
    "book-set": "book",  # Set of books can be considered as 'book' in BibTeX
    "book-chapter": "inbook",
    "book-section": "inbook",  # Sections in books can be considered as 'inbook'
    "book-part": "inbook",  # Parts of books can be considered as 'inbook'
    "book-track": "inbook",  # Tracks in books can be considered as 'inbook'
    "reference-entry": "inbook",  # Entries in reference books can be considered as 'inbook'
    "dissertation": "phdthesis",  # Dissertations are usually PhD thesis
    "posted-content": "misc",  # No direct equivalent, so 'misc' is used
    "peer-review": "misc",  # No direct equivalent, so 'misc' is used
    "other": "article",  # Assume an article if we don't know the type
}
