from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import aiohttp
import pytest

import paperqa
from paperqa.clients import (
    ALL_CLIENTS,
    CrossrefProvider,
    DocMetadataClient,
    SemanticScholarProvider,
)
from paperqa.clients.client_models import MetadataPostProcessor, MetadataProvider
from paperqa.clients.journal_quality import JournalQualityPostProcessor
from paperqa.clients.retractions import RetractionDataPostProcessor


@pytest.mark.vcr
@pytest.mark.parametrize(
    "paper_attributes",
    [
        {
            "title": (
                "Effect of native oxide layers on copper thin-film "
                "tensile properties: A reactive molecular dynamics study"
            ),
            "source": ["semantic_scholar", "crossref"],
            "key": "skarlinski2015effectofnative",
            "doi": "10.1063/1.4938384",
            "doc_id": "c217ec9289696c3c",
            "journal": "Journal of Applied Physics",
            "authors": ["Michael D. Skarlinski", "David J. Quesnel"],
            "formatted_citation": (
                "Michael D. Skarlinski and David J. Quesnel. Effect of native oxide"
                " layers on copper thin-film tensile properties: a reactive molecular"
                " dynamics study. Journal of Applied Physics, 118:235306, Dec 2015."
                " URL: https://doi.org/10.1063/1.4938384, doi:10.1063/1.4938384. This"
                " article has 8 citations and is from a peer-reviewed journal."
            ),
            "is_oa": False,
        },
        {
            "title": (
                "PaperQA: Retrieval-Augmented Generative Agent for Scientific Research"
            ),
            "source": ["semantic_scholar"],
            "key": "lala2023paperqaretrievalaugmentedgenerative",
            "doi": "10.48550/arxiv.2312.07559",
            "doc_id": "bb985e0e3265d678",
            "journal": "ArXiv",
            "authors": [
                "Jakub L'ala",
                "Odhran O'Donoghue",
                "Aleksandar Shtedritski",
                "Sam Cox",
                "Samuel G. Rodriques",
                "Andrew D. White",
            ],
            "formatted_citation": (
                "Jakub L'ala, Odhran O'Donoghue, Aleksandar Shtedritski, Sam Cox,"
                " Samuel G. Rodriques, and Andrew D. White. Paperqa:"
                " retrieval-augmented generative agent for scientific research. ArXiv,"
                " Dec 2023. URL: https://doi.org/10.48550/arxiv.2312.07559,"
                " doi:10.48550/arxiv.2312.07559. This article has 25 citations."
            ),
            "is_oa": None,
        },
        {
            "title": "Augmenting large language models with chemistry tools",
            "source": ["semantic_scholar", "crossref"],
            "key": "bran2024augmentinglargelanguage",
            "doi": "10.1038/s42256-024-00832-8",
            "doc_id": "0f650d59b0a2ba5a",
            "journal": "Nature Machine Intelligence",
            "authors": [
                "Andres M. Bran",
                "Sam Cox",
                "Oliver Schilter",
                "Carlo Baldassari",
                "Andrew D. White",
                "Philippe Schwaller",
            ],
            "formatted_citation": (
                "Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D."
                " White, and Philippe Schwaller. Augmenting large language models with"
                " chemistry tools. Nature Machine Intelligence, 6:525-535, May 2024."
                " URL: https://doi.org/10.1038/s42256-024-00832-8,"
                " doi:10.1038/s42256-024-00832-8. This article has 196 citations and is"
                " from a domain leading peer-reviewed journal."
            ),
            "is_oa": True,
        },
    ],
)
@pytest.mark.asyncio
async def test_title_search(paper_attributes: dict[str, str]) -> None:
    async with aiohttp.ClientSession() as session:
        client_list = [
            client for client in ALL_CLIENTS if client != RetractionDataPostProcessor
        ]
        client = DocMetadataClient(
            session,
            clients=cast(
                Collection[
                    type[MetadataPostProcessor[Any]] | type[MetadataProvider[Any]]
                ],
                client_list,
            ),
        )
        details = await client.query(title=paper_attributes["title"])
        assert set(details.other["client_source"]) == set(  # type: ignore[union-attr]
            paper_attributes["source"]
        ), "Should have the correct source"
        for key, value in paper_attributes.items():
            if key not in {"is_oa", "source"}:
                assert getattr(details, key) == value, f"Should have the correct {key}"
            elif key == "is_oa":
                assert (
                    details.other.get("is_oa") == value  # type: ignore[union-attr]
                ), "Open access data should match"


@pytest.mark.vcr
@pytest.mark.parametrize(
    "paper_attributes",
    [
        {
            "title": (
                "High-throughput screening of human genetic variants by pooled prime"
                " editing"
            ),
            "source": ["semantic_scholar", "crossref"],
            "key": "herger2024highthroughputscreeningof",
            "doi": "10.1101/2024.04.01.587366",
            "doc_id": "8e7669b50f31c52b",
            "journal": "bioRxiv",
            "authors": [
                "Michael Herger",
                "Christina M. Kajba",
                "Megan Buckley",
                "Ana Cunha",
                "Molly Strom",
                "Gregory M. Findlay",
            ],
            "formatted_citation": (
                "Michael Herger, Christina M. Kajba, Megan Buckley, Ana Cunha, Molly"
                " Strom, and Gregory M. Findlay. High-throughput screening of human"
                " genetic variants by pooled prime editing. bioRxiv, Apr 2024. URL:"
                " https://doi.org/10.1101/2024.04.01.587366,"
                " doi:10.1101/2024.04.01.587366. This article has 1 citations."
            ),
            "is_oa": True,
        },
        {
            "title": (
                "An essential role of active site arginine residue in iodide binding"
                " and histidine residue in electron transfer for iodide oxidation by"
                " horseradish peroxidase"
            ),
            "source": ["semantic_scholar", "crossref"],
            "key": "adak2001anessentialrole",
            "doi": "10.1023/a:1007154515475",
            "doc_id": "3012c6676b658a27",
            "journal": "Molecular and Cellular Biochemistry",
            "authors": [
                "Subrata Adak",
                "Debashis Bandyopadhyay",
                "Uday Bandyopadhyay",
                "Ranajit K. Banerjee",
            ],
            "formatted_citation": (
                "Subrata Adak, Debashis Bandyopadhyay, Uday Bandyopadhyay, and Ranajit"
                " K. Banerjee. An essential role of active site arginine residue in"
                " iodide binding and histidine residue in electron transfer for iodide"
                " oxidation by horseradish peroxidase. Molecular and Cellular"
                " Biochemistry, 218:1-11, Feb 2001. URL:"
                " https://doi.org/10.1023/a:1007154515475, doi:10.1023/a:1007154515475."
                " This article has 7 citations and is from a peer-reviewed journal."
            ),
            "is_oa": False,
        },
        {
            "title": "Convalescent-anti-sars-cov-2-plasma/immune-globulin",
            "source": ["semantic_scholar", "crossref"],
            "key": "unknownauthors2023convalescentantisarscov2plasmaimmuneglobulin",
            "doi": "10.1007/s40278-023-41815-2",
            "doc_id": "c2a60b772778732c",
            "journal": "Reactions Weekly",
            "authors": [],
            "formatted_citation": (
                "Unknown author(s)."
                " Convalescent-anti-sars-cov-2-plasma/immune-globulin. Reactions"
                " Weekly, 1962:145-145, Jun 2023. URL:"
                " https://doi.org/10.1007/s40278-023-41815-2,"
                " doi:10.1007/s40278-023-41815-2. This article has 0 citations and is"
                " from a peer-reviewed journal."
            ),
            "is_oa": False,
        },
    ],
)
@pytest.mark.asyncio
async def test_doi_search(paper_attributes: dict[str, str]) -> None:
    async with aiohttp.ClientSession() as session:
        client_list = [
            client for client in ALL_CLIENTS if client != RetractionDataPostProcessor
        ]
        client = DocMetadataClient(
            session,
            clients=cast(
                Collection[
                    type[MetadataPostProcessor[Any]] | type[MetadataProvider[Any]]
                ],
                client_list,
            ),
        )
        details = await client.query(doi=paper_attributes["doi"])
        assert set(details.other["client_source"]) == set(  # type: ignore[union-attr]
            paper_attributes["source"]
        ), "Should have the correct source"
        for key, value in paper_attributes.items():
            if key not in {"is_oa", "source"}:
                assert getattr(details, key) == value, f"Should have the correct {key}"
            elif key == "is_oa":
                assert (
                    details.other.get("is_oa") == value  # type: ignore[union-attr]
                ), "Open access data should match"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_bulk_doi_search() -> None:
    dois = [
        "10.1063/1.4938384",
        "10.48550/arxiv.2312.07559",
        "10.1038/s42256-024-00832-8",
        "10.1101/2024.04.01.587366",
        "10.1023/a:1007154515475",
        "10.1007/s40278-023-41815-2",
    ]
    async with aiohttp.ClientSession() as session:
        client = DocMetadataClient(session)
        details = await client.bulk_query([{"doi": doi} for doi in dois])
        assert len(details) == 6, "Should return 6 results"
        assert all(d for d in details), "All results should be non-None"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_bulk_title_search() -> None:
    titles = [
        (
            "Effect of native oxide layers on copper thin-film tensile properties: A"
            " reactive molecular dynamics study"
        ),
        "PaperQA: Retrieval-Augmented Generative Agent for Scientific Research",
        "Augmenting large language models with chemistry tools",
        "High-throughput screening of human genetic variants by pooled prime editing",
        (
            "An essential role of active site arginine residue in iodide binding and"
            " histidine residue in electron transfer for iodide oxidation by"
            " horseradish peroxidase"
        ),
        "Convalescent-anti-sars-cov-2-plasma/immune-globulin",
    ]
    async with aiohttp.ClientSession() as session:
        client = DocMetadataClient(session)
        details = await client.bulk_query([{"title": title} for title in titles])
        assert len(details) == 6, "Should return 6 results"
        assert all(d for d in details), "All results should be non-None"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_bad_titles() -> None:
    async with aiohttp.ClientSession() as session:
        client = DocMetadataClient(session)
        details = await client.query(title="askldjrq3rjaw938h")
        assert not details, "Should return None for bad title"
        details = await client.query(
            title=(
                "Effect of native oxide layers on copper thin-film tensile properties:"
                " A study"
            )
        )
        assert details, "Should find a similar title"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_bad_dois() -> None:
    async with aiohttp.ClientSession() as session:
        client = DocMetadataClient(session)
        details = await client.query(title="abs12032jsdafn")
        assert not details, "Should return None for bad doi"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_minimal_fields_filtering() -> None:
    async with aiohttp.ClientSession() as session:
        client = DocMetadataClient(session)
        details = await client.query(
            title="Augmenting large language models with chemistry tools",
            fields=["title", "doi"],
        )
        assert not details.journal, "Journal should not be populated"  # type: ignore[union-attr]
        assert not details.year, "Year should not be populated"  # type: ignore[union-attr]
        assert not details.authors, "Authors should not be populated"  # type: ignore[union-attr]
        assert details.citation == (  # type: ignore[union-attr]
            "Unknown author(s). Augmenting large language models with chemistry tools."
            " Unknown journal, Unknown year. URL:"
            " https://doi.org/10.1038/s42256-024-00832-8,"
            " doi:10.1038/s42256-024-00832-8."
        ), "Citation should be populated"
        assert set(details.other["client_source"]) == {  # type: ignore[union-attr]
            "semantic_scholar",
            "crossref",
        }, "Should be from two sources"
        assert not details.source_quality, "No source quality data should exist"  # type: ignore[union-attr]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_s2_only_fields_filtering() -> None:
    async with aiohttp.ClientSession() as session:
        # now get with authors just from one source
        s2_client = DocMetadataClient(session, clients=[SemanticScholarProvider])
        s2_details = await s2_client.query(
            title="Augmenting large language models with chemistry tools",
            fields=["title", "doi", "authors"],
        )
        assert s2_details.authors, "Authors should be populated"  # type: ignore[union-attr]
        assert set(s2_details.other["client_source"]) == {"semantic_scholar"}  # type: ignore[union-attr]
        assert s2_details.citation == (  # type: ignore[union-attr]
            "Andrés M Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D."
            " White, and P. Schwaller. Augmenting large language models with chemistry"
            " tools. Unknown journal, Unknown year. URL:"
            " https://doi.org/10.1038/s42256-024-00832-8,"
            " doi:10.1038/s42256-024-00832-8."
        ), "Citation should be populated"
        assert not s2_details.source_quality, "No source quality data should exist"  # type: ignore[union-attr]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_crossref_journalquality_fields_filtering() -> None:
    async with aiohttp.ClientSession() as session:
        crossref_client = DocMetadataClient(
            session,
            clients=cast(
                Collection[
                    type[MetadataPostProcessor[Any]] | type[MetadataProvider[Any]]
                ],
                [CrossrefProvider, JournalQualityPostProcessor],
            ),
        )
        crossref_details = await crossref_client.query(
            title="Augmenting large language models with chemistry tools",
            fields=["title", "doi", "authors", "journal"],
        )
        assert crossref_details, "Failed to query crossref"
        assert set(crossref_details.other["client_source"]) == {
            "crossref"
        }, "Should be from only crossref"
        assert crossref_details.source_quality == 2, "Should have source quality data"
        assert (
            crossref_details.citation
            == "Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D."
            " White, and Philippe Schwaller. Augmenting large language models with"
            " chemistry tools. Nature Machine Intelligence, Unknown year. URL:"
            " https://doi.org/10.1038/s42256-024-00832-8,"
            " doi:10.1038/s42256-024-00832-8."
        ), "Citation should be populated"

    async with aiohttp.ClientSession() as session:
        crossref_client = DocMetadataClient(
            session,
            clients=cast(
                Collection[
                    type[MetadataPostProcessor[Any]] | type[MetadataProvider[Any]]
                ],
                [CrossrefProvider, JournalQualityPostProcessor],
            ),
        )
        nejm_crossref_details = await crossref_client.query(
            title=(
                "Beta-Blocker Interruption or Continuation after Myocardial Infarction"  # codespell:ignore
            ),
            fields=["title", "doi", "authors", "journal"],
        )

        assert nejm_crossref_details.source_quality == 3, "Should have source quality data"  # type: ignore[union-attr]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_author_matching() -> None:
    async with aiohttp.ClientSession() as session:
        crossref_client = DocMetadataClient(session, clients=[CrossrefProvider])
        s2_client = DocMetadataClient(session, clients=[SemanticScholarProvider])
        crossref_details_bad_author = await crossref_client.query(
            title="Augmenting large language models with chemistry tools",
            authors=["Jack NoScience"],
            fields=["title", "doi", "authors"],
        )

        s2_details_bad_author = await s2_client.query(
            title="Augmenting large language models with chemistry tools",
            authors=["Jack NoScience"],
            fields=["title", "doi", "authors"],
        )

        s2_details_w_author = await s2_client.query(
            title="Augmenting large language models with chemistry tools",
            authors=["Andres M. Bran", "Sam Cox"],
            fields=["title", "doi", "authors"],
        )

        assert not crossref_details_bad_author, "Should return None for bad author"
        assert not s2_details_bad_author, "Should return None for bad author"
        assert s2_details_w_author, "Should return results for good author"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_odd_client_requests() -> None:
    # try querying using an authors match, but not requesting authors back
    async with aiohttp.ClientSession() as session:
        client = DocMetadataClient(session)
        details = await client.query(
            title="Augmenting large language models with chemistry tools",
            authors=["Andres M. Bran", "Sam Cox"],
            fields=["title", "doi"],
        )
        assert (
            details.authors  # type: ignore[union-attr]
        ), "Should return correct author results"

    # try querying using a title, asking for no DOI back
    async with aiohttp.ClientSession() as session:
        client = DocMetadataClient(session)
        details = await client.query(
            title="Augmenting large language models with chemistry tools",
            fields=["title"],
        )
        assert (
            details.doi  # type: ignore[union-attr]
        ), "Should return a doi even though we don't ask for it"

    # try querying using a title, asking for no title back
    async with aiohttp.ClientSession() as session:
        client = DocMetadataClient(session)
        details = await client.query(
            title="Augmenting large language models with chemistry tools",
            fields=["doi"],
        )
        assert (
            details.title  # type: ignore[union-attr]
        ), "Should return a title even though we don't ask for it"

    async with aiohttp.ClientSession() as session:
        client = DocMetadataClient(session)
        details = await client.query(
            doi="10.1007/s40278-023-41815-2",
            fields=["doi", "title", "gibberish-field", "no-field"],
        )
        assert (
            details.title  # type: ignore[union-attr]
        ), "Should return title even though we asked for some bad fields"


@pytest.mark.asyncio
@patch.object(paperqa.clients.crossref, "CROSSREF_API_REQUEST_TIMEOUT", 0.05)
@patch.object(
    paperqa.clients.semantic_scholar, "SEMANTIC_SCHOLAR_API_REQUEST_TIMEOUT", 0.05
)
async def test_ensure_robust_to_timeouts() -> None:
    async with aiohttp.ClientSession() as session:
        client = DocMetadataClient(session)
        details = await client.query(
            doi="10.1007/s40278-023-41815-2",
            fields=["doi", "title"],
        )
    assert details is None, "Should return None for timeout"


@pytest.mark.asyncio
async def test_bad_init() -> None:
    with pytest.raises(
        ValueError, match="At least one MetadataProvider must be provided."
    ):
        client = DocMetadataClient(clients=[])  # noqa: F841


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ensure_sequential_run(caplog, reset_log_levels) -> None:  # noqa: ARG001
    caplog.set_level(logging.DEBUG)
    # were using a DOI that is NOT in crossref, but running the crossref client first
    # we will ensure that both are run sequentially

    async with aiohttp.ClientSession() as session:
        client = DocMetadataClient(
            session=session,
            clients=cast(
                Sequence[
                    Collection[
                        type[MetadataPostProcessor[Any]] | type[MetadataProvider[Any]]
                    ]
                ],
                [[CrossrefProvider], [SemanticScholarProvider]],
            ),
        )
        details = await client.query(
            doi="10.48550/arxiv.2312.07559",
            fields=["doi", "title"],
        )
        assert details, "Should find the right DOI in the second client"
        record_indices = {"crossref": -1, "semantic_scholar": -1}
        for n, record in enumerate(caplog.records):
            if "CrossrefProvider" in record.msg:
                record_indices["crossref"] = n
            if "SemanticScholarProvider" in record.msg:
                record_indices["semantic_scholar"] = n
        assert (
            record_indices["crossref"] < record_indices["semantic_scholar"]
        ), "Crossref should run first"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ensure_sequential_run_early_stop(
    caplog, reset_log_levels  # noqa: ARG001
) -> None:
    caplog.set_level(logging.DEBUG)
    # now we should stop after hitting s2
    async with aiohttp.ClientSession() as session:
        client = DocMetadataClient(
            session=session,
            clients=cast(
                Sequence[
                    Collection[
                        type[MetadataPostProcessor[Any]] | type[MetadataProvider[Any]]
                    ]
                ],
                [[SemanticScholarProvider], [CrossrefProvider]],
            ),
        )
        details = await client.query(
            doi="10.48550/arxiv.2312.07559",
            fields=["doi", "title"],
        )
        assert details, "Should find the right DOI in the second client"
        record_indices = {"crossref": -1, "semantic_scholar": -1, "early_stop": -1}
        for n, record in enumerate(caplog.records):
            if "CrossrefProvider" in record.msg:
                record_indices["crossref"] = n
            if "SemanticScholarProvider" in record.msg:
                record_indices["semantic_scholar"] = n
            if "stopping early." in record.msg:
                record_indices["early_stop"] = n
        assert (
            record_indices["crossref"] == -1
        ), "Crossref should be index -1 i.e. not found"
        assert (
            record_indices["semantic_scholar"] != -1
        ), "Semantic Scholar should be found"
        assert record_indices["early_stop"] != -1, "We should stop early."


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_crossref_retraction_status(stub_data_dir: Path) -> None:
    async with aiohttp.ClientSession() as session:
        retract_processor = RetractionDataPostProcessor(
            f"{stub_data_dir}/stub_retractions.csv"
        )
        crossref_client = DocMetadataClient(
            session,
            clients=cast(
                Collection[
                    type[MetadataPostProcessor[Any]] | type[MetadataProvider[Any]]
                ],
                [CrossrefProvider, retract_processor],
            ),
        )
        crossref_details = await crossref_client.query(
            title="The Dilemma and Countermeasures of Music Education under the Background of Big Data",
            fields=["title", "doi", "authors", "journal"],
        )
        assert crossref_details
        assert (
            "**RETRACTED ARTICLE** Citation: Jiaye Han."
            in crossref_details.formatted_citation
        )
        assert crossref_details.is_retracted is True, "Should be retracted"
