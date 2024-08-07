
from typing import Any

import aiohttp

from crossref import get_paper_details_from_crossref
from paperqa.clients.exceptions import DOINotFoundError
from semantic_scholar import get_paper_details_from_s2

from paperqa.types import PaperDetails

class PaperMetadataClient():
    def __init__(self, session: aiohttp.ClientSession) -> None:
        self.session = session
    
    # TODO: pared down query for just DOI / links / title

    # TODO: we should probably allow for author searches too
    def query(self, doi: str | None = None, title: str | None = None) -> PaperDetails:
        # TODO: proper validation here
        if title is doi is None:
            raise ValueError("Either title or DOI must be provided.")
        
        try:
            return get_paper_details_from_crossref(doi=doi, title=title, session=self.session)
        except DOINotFoundError:
            return get_paper_details_from_s2(doi=doi, title=title, session=self.session)
    
    
# this is what take incomplete paperdetails data from gscholar and fills it in
async def parse_google_scholar_metadata(  # noqa: C901
    paper: dict[str, Any], session: ClientSession | None = None
) -> dict[str, Any]:
    """Parse pre-processed paper metadata from Google Scholar into a richer format."""
    if session is None:
        session = await CACHED_METADATA_SESSION.get_session()

    doi: str | None = (paper.get("externalIds") or {}).get("DOI")
    citation: str | None = None
    if doi:
        try:
            missing_replacements = {
                "author": "Unknown authors",
                "year": str(paper.get("year", "Unknown year")),
                "title": paper.get("title", "Unknown title"),
            }
            bibtex = await doi_to_bibtex(
                doi, session, missing_replacements=missing_replacements
            )
            citation = format_bibtex(
                bibtex, clean=False, missing_replacements=missing_replacements
            )
        except DOINotFoundError:
            doi = None
        except CitationConversionError:
            citation = None
    if (not doi or not citation) and "inline_links" in paper:
        # get citation by following link
        # SLOW SLOW Using SerpAPI for this
        async with session.get(
            paper["inline_links"]["serpapi_cite_link"],
            params={"api_key": os.environ["SERPAPI_API_KEY"]},
        ) as r:
            # we raise here, because something really is wrong.
            r.raise_for_status()
            data = await r.json()
        citation = next(c["snippet"] for c in data["citations"] if c["title"] == "MLA")
        bibtex_link = next(c["link"] for c in data["links"] if c["name"] == "BibTeX")
        async with session.get(bibtex_link) as r:
            try:
                r.raise_for_status()
            except ClientResponseError as exc:
                # we may have a 443 - link expired
                msg = (
                    "Google scholar blocked"
                    if r.status == 443
                    else "Unexpected failure to follow"
                )
                raise RuntimeError(
                    f"{msg} bibtex link {bibtex_link} for paper {paper}."
                ) from exc
            bibtex = await r.text()
            if not bibtex.strip().startswith("@"):
                raise RuntimeError(
                    f"Google scholar ip block bibtex link {bibtex_link} for paper {paper}."
                )
    # will have bibtex by now
    key = bibtex.split("{")[1].split(",")[0]

    if not citation:
        raise RuntimeError(
            f"Exhausted all options for citation retrieval for {paper!r}"
        )
    # TODO: PaperDetails object??
    if "citationCount" not in paper:
        raise RuntimeError("citationCount not in paper metadata")

    if "year" not in paper:
        raise RuntimeError("year not in paper metadata")

    return {
        "citation": citation,
        "key": key,
        "bibtex": bibtex,
        "year": paper["year"],
        "url": paper.get("link"),
        "paper_id": paper["paper_id"],
        "doi": paper["externalIds"].get("DOI"),
        "citationCount": paper["citationCount"],
        "title": paper["title"],
    }