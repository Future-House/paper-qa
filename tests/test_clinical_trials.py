# pylint: disable=redefined-outer-name
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import ClientResponseError, ClientSession

from paperqa import Docs, Settings
from paperqa.sources.clinical_trials import (
    add_clinical_trials_to_docs,
    api_get_clinical_trial,
    api_search_clinical_trials,
    format_to_doc_details,
    parse_clinical_trial,
)

SAMPLE_TRIAL_DATA = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT12345678",
            "briefTitle": "Test Clinical Trial",
        },
        "sponsorCollaboratorsModule": {
            "responsibleParty": {"investigatorFullName": "Dr. John Doe"},
            "leadSponsor": {"name": "Test Organization"},
        },
        "statusModule": {"startDateStruct": {"date": "2023-01"}},
    }
}


@pytest.fixture
def mock_bucket_client():
    with patch("app.clinical_trials.GCS_BUCKET_CLINICAL_TRIALS_CLIENT") as mock_client:
        yield mock_client


@pytest.fixture
def mock_session():
    return AsyncMock(spec=ClientSession)


@pytest.mark.asyncio
async def test_api_search_clinical_trials_success(mock_session):
    mock_response = AsyncMock(status=200)
    mock_response.raise_for_status = Mock()
    mock_response.text.return_value = json.dumps({"studies": [SAMPLE_TRIAL_DATA]})
    mock_response.json.return_value = {"studies": [SAMPLE_TRIAL_DATA]}
    mock_session.get.return_value.__aenter__.return_value = mock_response

    result = await api_search_clinical_trials("test query", mock_session)

    assert result == {"studies": [SAMPLE_TRIAL_DATA]}
    mock_session.get.assert_called_once()
    mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_api_get_clinical_trial_success(mock_session):
    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = SAMPLE_TRIAL_DATA
    mock_session.get.return_value.__aenter__.return_value = mock_response

    result = await api_get_clinical_trial("NCT12345678", mock_session)

    assert result == SAMPLE_TRIAL_DATA
    mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_api_get_clinical_trial_not_found(mock_session):
    mock_session.get.side_effect = ClientResponseError(
        request_info=Mock(), history=Mock(), status=404
    )

    result = await api_get_clinical_trial("NCT12345678", mock_session)

    assert result is None, "Should be robust to missing trials"


def test_format_to_doc_details():
    result = format_to_doc_details(SAMPLE_TRIAL_DATA)

    assert result.title == "Test Clinical Trial"
    assert result.authors == ["Dr. John Doe"]
    assert result.year == 2023
    assert "Dr. John Doe" in result.citation
    assert "Test Clinical Trial" in result.citation
    assert "Test Organization" in result.citation
    assert "2023" in result.citation
    assert "NCT12345678" in result.citation


@pytest.mark.asyncio
async def test_add_clinical_trials_to_docs():
    mock_session = AsyncMock(spec=ClientSession)
    mock_docs = Mock(spec=Docs)
    mock_docs.aadd_texts = AsyncMock()
    mock_docs.texts = []

    mock_response = AsyncMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "studies": [
            {"protocolSection": {"identificationModule": {"nctId": "NCT12345678"}}}
        ]
    }
    mock_session.get.return_value.__aenter__.return_value = mock_response

    with patch(
        "paperqa.sources.clinical_trials.search_retrieve_clinical_trials",
        return_value=([SAMPLE_TRIAL_DATA], 1),
    ):
        await add_clinical_trials_to_docs(
            "test query", mock_docs, Settings(), session=mock_session
        )

        assert (
            mock_docs.aadd_texts.call_count == 2
        ), "One for the metadata and one for the trial"
        call_args = mock_docs.aadd_texts.call_args[1]
        assert "doc" in call_args
        assert isinstance(call_args["doc"].citation, str)


def test_parse_clinical_trial():
    # Test data with all fields including detailed description
    complete_trial_data = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT12345678",
                "briefTitle": "Sample Trial",
                "organization": {"fullName": "Test Hospital"},
            },
            "statusModule": {
                "overallStatus": "Recruiting",
                "startDateStruct": {"date": "2023-01"},
                "completionDateStruct": {"date": "2024-12"},
            },
            "descriptionModule": {
                "briefSummary": "This is a brief summary",
                "detailedDescription": "This is a detailed description",
            },
            "designModule": {
                "studyType": "Interventional",
                "phases": ["Phase 1", "Phase 2"],
                "enrollmentInfo": {"count": 100},
            },
            "eligibilityModule": {"eligibilityCriteria": "Must be 18 or older"},
        }
    }

    # Test data without detailed description
    minimal_trial_data = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT87654321",
                "briefTitle": "Basic Trial",
            },
            "statusModule": {},
            "descriptionModule": {"briefSummary": "Brief summary only"},
            "designModule": {"phases": []},
            "eligibilityModule": {},
        }
    }

    # Test complete data
    result_complete = parse_clinical_trial(complete_trial_data)

    # Verify all sections are present
    assert "CLINICAL TRIAL INFORMATION" in result_complete
    assert "NCT Number: NCT12345678" in result_complete
    assert "Organization: Test Hospital" in result_complete
    assert "Overall Status: Recruiting" in result_complete
    assert "Start Date: 2023-01" in result_complete
    assert "Completion Date: 2024-12" in result_complete
    assert "This is a brief summary" in result_complete
    assert "This is a detailed description" in result_complete
    assert "Study Type: Interventional" in result_complete
    assert "Phase: Phase 1, Phase 2" in result_complete
    assert "Enrollment: 100 participants" in result_complete
    assert "Must be 18 or older" in result_complete

    # Verify section order (detailed description should come after brief summary)
    brief_pos = result_complete.find("This is a brief summary")
    detailed_pos = result_complete.find("This is a detailed description")
    assert (
        brief_pos < detailed_pos
    ), "Detailed description should come after brief summary"

    # Test minimal data
    result_minimal = parse_clinical_trial(minimal_trial_data)

    # Verify default values for missing fields
    assert "NCT Number: NCT87654321" in result_minimal
    assert "Organization: Not provided" in result_minimal
    assert "Start Date: Not provided" in result_minimal
    assert "Phase: " in result_minimal  # Empty phases list results in empty string
    assert (
        "DETAILED DESCRIPTION" not in result_minimal
    ), "Detailed description section should not be present"

    # Verify newlines and formatting
    assert result_complete.count("\n") > result_minimal.count(
        "\n"
    ), "Complete result should have more lines due to detailed description"
    assert "=" * 25 in result_complete, "Section separators should be present"
