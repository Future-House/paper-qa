import json
import logging
import ssl
from contextlib import suppress
from datetime import datetime
from typing import Any

import httpx
import httpx_aiohttp
from lmi.utils import gather_with_concurrency
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_incrementing,
)

from paperqa.docs import Docs
from paperqa.settings import Settings
from paperqa.types import DocDetails, Embeddable, Text

logger = logging.getLogger(__name__)


CLINICAL_TRIALS_BASE = "clinicaltrials.gov"
CLINICAL_TRIALS_URL = f"https://{CLINICAL_TRIALS_BASE}"
STUDIES_API_URL = CLINICAL_TRIALS_URL + "/api/v2/studies"
SEARCH_API_FIELDS = "NCTId,OfficialTitle"
SEARCH_PAGE_SIZE = 1000
TRIAL_API_FIELDS = "protocolSection,derivedSection"
DOWNLOAD_CONCURRENCY = 20
TRIAL_CHAR_TRUNCATION_SIZE = 28_000  # stay under 8k tokens for embeddings context limit
MALFORMATTED_QUERY_STATUS: int = 400


@retry(
    stop=stop_after_attempt(3),
    wait=wait_incrementing(0.1, 0.1),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def api_search_clinical_trials(query: str, client: httpx.AsyncClient) -> dict:
    response = await client.get(
        STUDIES_API_URL,
        params={
            "query.term": query,
            "fields": SEARCH_API_FIELDS,
            "pageSize": SEARCH_PAGE_SIZE,
            "countTotal": "true",
            "sort": "@relevance",
        },
    )
    if response.status_code == MALFORMATTED_QUERY_STATUS:
        # the 400s from clinicaltrials.gov are not JSON, here's an example text:
        # > Error parsing query in Other terms:
        # > Allowed values for enum field `protocolSection.statusModule.overallStatus`
        # > are `ACTIVE_NOT_RECRUITING`, `COMPLETED`, `ENROLLING_BY_INVITATION`,
        # > `NOT_YET_RECRUITING`, `RECRUITING`, `SUSPENDED`, `TERMINATED`, `WITHDRAWN`,
        # > `AVAILABLE`, `NO_LONGER_AVAILABLE`, `TEMPORARILY_NOT_AVAILABLE`,
        # > `APPROVED_FOR_MARKETING`, `WITHHELD`, `UNKNOWN`
        raise httpx.HTTPStatusError(
            message=response.text, request=response.request, response=response
        )
    response.raise_for_status()
    return response.json()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_incrementing(0.1, 0.1),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def api_get_clinical_trial(nct_id: str, client: httpx.AsyncClient) -> dict | None:
    with suppress(httpx.HTTPStatusError):
        response = await client.get(
            f"{STUDIES_API_URL}/{nct_id}", params={"fields": TRIAL_API_FIELDS}
        )
        response.raise_for_status()
        return response.json()
    return None


async def search_retrieve_clinical_trials(
    query: str,
    client: httpx.AsyncClient,
    limit: int = 10,
    offset: int = 0,
) -> tuple[list[dict], int]:

    search_results = await api_search_clinical_trials(query, client=client)
    return (
        [
            trial
            for trial in await gather_with_concurrency(
                DOWNLOAD_CONCURRENCY,
                [
                    api_get_clinical_trial(
                        result["protocolSection"]["identificationModule"]["nctId"],
                        client,
                    )
                    for result in search_results.get("studies", [])[
                        offset : offset + limit
                    ]
                ],
            )
            if trial
        ],
        search_results.get("totalCount", 0),
    )


def format_to_doc_details(trial_data: dict) -> DocDetails:
    """
    Format clinical trial data into ICMJE citation style.

    Args:
        trial_data (dict): Clinical trial data in ClinicalTrials.gov JSON format
    """
    protocol = trial_data.get("protocolSection", {})

    investigator = (
        protocol.get("sponsorCollaboratorsModule", {})
        .get("responsibleParty", {})
        .get("investigatorFullName", "")
    )
    title = protocol.get("identificationModule", {}).get("briefTitle", "")
    organization = (
        protocol.get("sponsorCollaboratorsModule", {})
        .get("leadSponsor", {})
        .get("name", "")
    )
    start_date = (
        protocol.get("statusModule", {}).get("startDateStruct", {}).get("date", "")
    )
    nct_id = protocol.get("identificationModule", {}).get("nctId", "")

    # Extract year from date (assuming YYYY-MM format)
    year = start_date.split("-")[0] if start_date else ""

    citation_parts = []

    if investigator:
        citation_parts.append(f"{investigator}.")

    if title:
        citation_parts.append(f" {title}.")

    if organization:
        citation_parts.append(f" {organization}.")

    if year:
        citation_parts.append(f" {year}.")

    if nct_id:
        citation_parts.append(f" ClinicalTrials.gov Identifier: {nct_id}")

    citation = "".join(citation_parts)

    return DocDetails(
        title=title,
        docname=nct_id,
        dockey=nct_id,
        authors=[investigator],
        year=year or None,
        citation=citation,
        other={"client_source": [CLINICAL_TRIALS_BASE]},
        fields_to_overwrite_from_metadata=set(),
    )


def parse_clinical_trial(json_data: dict[str, Any]) -> str:
    """Convert clinical trial JSON data into human readable format."""
    protocol = json_data.get("protocolSection", {})
    # Get different sections
    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    description = protocol.get("descriptionModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    design = protocol.get("designModule", {})

    # Build all sections at once
    sections = [
        # Title and Basic Information
        "CLINICAL TRIAL INFORMATION",
        "=" * 25,
        f"NCT Number: {identification.get('nctId', 'Not provided')}",
        f"Title: {identification.get('briefTitle', 'Not provided')}",
        (
            "Organization:"
            f" {identification.get('organization', {}).get('fullName', 'Not provided')}"
        ),
        # Status Information
        "\nSTUDY STATUS",
        "=" * 13,
        f"Overall Status: {status.get('overallStatus', 'Not provided')}",
        f"Start Date: {status.get('startDateStruct', {}).get('date', 'Not provided')}",
        (
            "Completion Date:"
            f" {status.get('completionDateStruct', {}).get('date', 'Not provided')}"
        ),
        # Study Description
        "\nSTUDY DESCRIPTION",
        "=" * 17,
        description.get("briefSummary", "Not provided"),
        # Study Design
        "\nSTUDY DESIGN",
        "=" * 13,
        f"Study Type: {design.get('studyType', 'Not provided')}",
        f"Phase: {', '.join(design.get('phases', ['Not provided']))}",
        (
            "Enrollment:"
            f" {design.get('enrollmentInfo', {}).get('count', 'Not provided')} participants"
        ),
        # Eligibility
        "\nELIGIBILITY CRITERIA",
        "=" * 19,
        eligibility.get("eligibilityCriteria", "Not provided"),
    ]

    # Add detailed description if available
    if description.get("detailedDescription"):
        detailed_section = [
            "\nDETAILED DESCRIPTION",
            "=" * 20,
            description.get("detailedDescription", "Not provided"),
        ]
        # Insert detailed description after brief summary
        sections[13:13] = detailed_section

    # Format the final text
    return "\n".join(sections)


async def add_clinical_trials_to_docs(
    query: str,
    docs: Docs,
    settings: Settings,
    limit: int = 10,
    offset: int = 0,
    client: httpx.AsyncClient | None = None,
) -> tuple[int, int, str | None]:
    """Add clinical trials to the docs state.

    Args:
        query: Query to search for.
        docs: Docs state to add the trials to.
        settings: Query settings.
        limit: Number of trials to add.
        offset: Offset for the search results.
        client: Async HTTP client for any requests.

    Returns:
        tuple[int, int, str | None]:
            Total number of trials found, number of trials added, and error message if any.
    """
    ssl_context = ssl.create_default_context()
    # clinicaltrials.gov throws 403's in GitHub Actions if TLS 1.3 is used with httpx
    ssl_context.maximum_version = ssl.TLSVersion.TLSv1_2
    # Cookies are not needed
    _client = (
        httpx_aiohttp.HttpxAiohttpClient(timeout=10.0, verify=ssl_context)
        if client is None
        else client
    )

    logger.info(f"Querying clinical trials for: {query}.")

    try:
        trials, total_result_count = await search_retrieve_clinical_trials(
            query, _client, limit, offset
        )
    except Exception as e:
        logger.warning(f"Failed to retrieve clinical trials for query: {query}.")
        # close client if it was ephemeral
        if client is None:
            await _client.aclose()  # TODO: move to context manager
        return (0, 0, str(e))

    logger.info(f"Successfully found {len(trials)} trials.")

    initial_docs_size = len(docs.texts)

    for trial in trials:
        trial_text = (
            parse_clinical_trial(trial)
            if settings.parsing.use_human_readable_clinical_trials
            else json.dumps(trial)
        )
        doc_details = format_to_doc_details(trial)
        # always uses full object, no chunking for clinical trials
        # for embedding model context windows, we truncate at TRIAL_CHAR_TRUNCATION_SIZE
        await docs.aadd_texts(
            texts=[
                Text(
                    text=trial_text[:TRIAL_CHAR_TRUNCATION_SIZE],
                    name=doc_details.docname,
                    doc=doc_details,
                )
            ],
            doc=doc_details,
            settings=settings,
        )
    logger.info(f"Added {len(docs.texts) - initial_docs_size} trials to docs state.")

    # we add a final context stub representing the metadata surrounding this search
    # it can be used to answer questions about the search results
    meta_details = DocDetails(
        title="Clinical Trials Search Result",
        docname=f"Clinical Trial Search: {query}",
        dockey=f"Clinical Trial Search: {query}",
        authors=["PaperQA"],
        year=datetime.now().year,
        citation=f"Clinical Trials Search via ClinicalTrials.gov: {query}",
        other={"client_source": [CLINICAL_TRIALS_BASE]},
        fields_to_overwrite_from_metadata=set(),
    )

    await docs.aadd_texts(
        texts=[
            Text(
                text=(
                    f"After querying the ClinicalTrials.gov API for '{query}',"
                    f" {total_result_count} trials were found."
                ),
                name=meta_details.docname,
                doc=meta_details,
            )
        ],
        doc=meta_details,
        settings=settings,
    )

    # close client if it was ephemeral
    if client is None:
        await _client.aclose()

    return (total_result_count, len(docs.texts) - initial_docs_size, None)


def partition_clinical_trials_by_source(text: Embeddable) -> int:
    if (
        hasattr(text, "doc")
        and isinstance(text.doc, DocDetails)
        and CLINICAL_TRIALS_BASE in text.doc.other.get("client_source", [])
    ):
        return 1
    return 0
