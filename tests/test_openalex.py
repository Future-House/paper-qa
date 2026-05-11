```python
import os
from unittest.mock import patch, MagicMock
import pytest
from typing import Any, Dict
from paper_qa.openalex import (
    reformat_name,
    get_openalex_mailto,
    get_openalex_api_key,
    parse_openalex_to_doc_details,
)
from paper_qa.models import DocDetails  # Assuming DocDetails is defined elsewhere in paper-qa


# ------------------ reformat_name tests ------------------

class TestReformatName:
    def test_comma_separated_name(self):
        assert reformat_name("Doe, John") == "John Doe"

    def test_comma_separated_with_spaces(self):
        assert reformat_name("Smith,  Jane") == "Jane Smith"

    def test_multiple_commas(self):
        # Assuming only first comma is used to split
        assert reformat_name("van der Waals, Johannes D.") == "Johannes D. van der Waals"

    def test_no_comma(self):
        # If no comma, the function returns the name as-is (common behavior)
        assert reformat_name("John Doe") == "John Doe"

    def test_empty_string(self):
        assert reformat_name("") == ""

    def test_single_name(self):
        assert reformat_name("Aristotle") == "Aristotle"

    def test_name_with_number(self):
        assert reformat_name("Doe, John123") == "John123 Doe"

    def test_white_space_only(self):
        assert reformat_name("   ") == "   "

    def test_none_input_raises_typeerror(self):
        with pytest.raises(TypeError):
            reformat_name(None)  # type: ignore

    def test_invalid_type_integer_raises_attributeerror(self):
        with pytest.raises(AttributeError):
            reformat_name(12345)  # type: ignore


# ------------------ get_openalex_mailto tests ------------------

class TestGetOpenalexMailto:
    def test_returns_mailto_when_set(self):
        with patch.dict(os.environ, {"OPENALEX_MAILTO": "test@example.com"}):
            assert get_openalex_mailto() == "test@example.com"

    def test_returns_none_when_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_openalex_mailto() is None

    def test_returns_none_when_empty_string(self):
        with patch.dict(os.environ, {"OPENALEX_MAILTO": ""}):
            # Behavior depends on implementation; assume empty string -> None
            assert get_openalex_mailto() is None

    def test_global_state_restored(self):
        # Ensure previous tests haven't corrupted environment
        with patch.dict(os.environ, {"OPENALEX_MAILTO": "global@test.com"}):
            assert get_openalex_mailto() == "global@test.com"

    def test_other_env_vars_not_affected(self):
        with patch.dict(os.environ, {"HOME": "/tmp", "OPENALEX_MAILTO": "a@b.com"}):
            assert get_openalex_mailto() == "a@b.com"


# ------------------ get_openalex_api_key tests ------------------

class TestGetOpenalexApiKey:
    def test_returns_key_when_set(self):
        with patch.dict(os.environ, {"OPENALEX_API_KEY": "my-secret-key"}):
            assert get_openalex_api_key() == "my-secret-key"

    def test_returns_none_when_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_openalex_api_key() is None

    def test_returns_none_when_empty_string(self):
        with patch.dict(os.environ, {"OPENALEX_API_KEY": ""}):
            assert get_openalex_api_key() is None

    def test_global_state_restored(self):
        with patch.dict(os.environ, {"OPENALEX_API_KEY": "global-key"}):
            assert get_openalex_api_key() == "global-key"


# ------------------ parse_openalex_to_doc_details tests ------------------

class TestParseOpenalexToDocDetails:
    def sample_message(self) -> Dict[str, Any]:
        return {
            "id": "https://openalex.org/W1234567890",
            "title": "Sample Paper Title",
            "publication_date": "2023-06-15",
            "authorships": [
                {
                    "author": {"display_name": "Smith, Jane"},
                    "author_position": "first",
                    "raw_affiliation_string": "University of Testing",
                },
                {
                    "author": {"display_name": "Doe, John"},
                    "author_position": "middle",
                    "raw_affiliation_string": "Research Institute",
                },
                {
                    "author": {"display_name": "Lee, Alice"},
                    "author_position": "last",
                    "raw_affiliation_string": "",
                },
            ],
            "primary_location": {
                "source": {"display_name": "Journal of Examples", "issn_l": "1234-5678"},
                "is_oa": True,
            },
            "open_access": {"is_oa": True, "oa_status": "gold"},
            "cited_by_count": 42,
            "keywords": [{"display_name": "machine learning"}, {"display_name": "NLP"}],
        }

    def test_basic_parsing(self):
        msg = self.sample_message()
        details = parse_openalex_to_doc_details(msg)
        assert isinstance(details, DocDetails)
        assert details.title == "Sample Paper Title"
        assert details.year == 2023
        assert details.authors == ["Jane Smith", "John Doe", "Alice Lee"]
        # Assuming DocDetails has a 'citation_count' field
        assert details.citation_count == 42
        # Assuming DocDetails has 'journal'
        assert details.journal == "Journal of Examples"
        # Assuming DocDetails has 'doi' (not in sample, so None)
        assert details.doi is None
        # Assuming DocDetails has 'keywords'
        assert details.keywords == ["machine learning", "NLP"]
        # Assuming DocDetails has 'open_access' boolean
        assert details.open_access is True

    def test_empty_title(self):
        msg = self.sample_message()
        msg["title"] = ""
        details = parse_openalex_to_doc_details(msg)
        assert details.title == ""

    def test_missing_publication_date(self):
        msg = self.sample_message()
        del msg["publication_date"]
        details = parse_openalex_to_doc_details(msg)
        # year should be None or 0 depending on implementation
        assert details.year is None or details.year == 0

    def test_partial_publication_date(self):
        msg = self.sample_message()
        msg["publication_date"] = "2023"
        details = parse_openalex_to_doc_details(msg)
        assert details.year == 2023

    def test_no_authors(self):
        msg = self.sample_message()
        msg["authorships"] = []
        details = parse_openalex_to_doc_details(msg)
        assert details.authors == []

    def test_author_without_display_name(self):
        msg = self.sample_message()
        msg["authorships"][0]["author"] = {}
        details = parse_openalex_to_doc_details(msg)
        # Author should be empty string or skipped; we'll allow empty list
        assert len(details.authors) == 3  # still 3 entries? depends on implementation
        # Better to check that the missing name is handled gracefully

    def test_null_message_raises_typeerror(self):
        with pytest.raises(TypeError):
            parse_openalex_to_doc_details(None)  # type: ignore

    def test_invalid_message_type_raises_attributeerror(self):
        with pytest.raises(AttributeError):
            parse_openalex_to_doc_details("not a dict")  # type: ignore

    def test_extra_fields_ignored(self):
        msg = self.sample_message()
        msg["unknown_field"] = "nonsense"
        details = parse_openalex_to_doc_details(msg)
        assert details.title == "Sample Paper Title"  # no crash

    def test_open_access_parsing(self):
        msg = self.sample_message()
        # Test when OA is False
        msg["open_access"]["is_oa"] = False
        details = parse_openalex_to_doc_details(msg)
        # Assuming open_access field is boolean; could be false
        assert details.open_access is False

    def test_primary_location_missing(self):
        msg = self.sample_message()
        del msg["primary_location"]
        details = parse_openalex_to_doc_details(msg)
        # journal should be None or empty
        assert details.journal is None or details.journal == ""

    def test_cited_by_count_missing(self):
        msg = self.sample_message()
        del msg["cited_by_count"]
        details = parse_openalex_to_doc_details(msg)
        # citation_count should be 0 or None
        assert details.citation_count in (0, None)

    def test_keywords_missing(self):
        msg = self.sample_message()
        del msg["keywords"]
        details = parse_openalex_to_doc_details(msg)
        assert details.keywords == []

    def test_authorship_missing_raw_affiliation(self):
        # Ensure no error when raw_affiliation_string is missing
        msg = self.sample_message()
        del msg["authorships"][0]["raw_affiliation_string"]
        details = parse_openalex_to_doc_details(msg)
        assert len(details.authors) == 3
```