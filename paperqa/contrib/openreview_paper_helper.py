import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from litellm import completion

from paperqa import Docs, Settings

try:
    import openreview
except ImportError as e:
    raise ImportError(
        "openreview-py requires the 'zotero' extra for 'pyzotero'. Please: `pip install paper-qa[openreview]`."
    ) from e
logger = logging.getLogger(__name__)


class OpenReviewPaperHelper:
    def __init__(
        self,
        settings: Settings,
        venue_id: str | None = "ICLR.cc/2025/Conference",
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.settings = settings
        Path(settings.paper_directory).mkdir(parents=True, exist_ok=True)

        self.client = openreview.api.OpenReviewClient(
            baseurl="https://api2.openreview.net",
            username=username or os.getenv("OPENREVIEW_USERNAME"),
            password=password or os.getenv("OPENREVIEW_PASSWORD"),
        )
        self.venue_id = venue_id

    def get_venues(self) -> list[str]:
        """Get list of available venues."""
        return self.client.get_group(id="venues").members

    def get_submissions(self) -> list[Any]:
        """Get all submissions for the current venue."""
        logger.info(f"Fetching submissions for venue {self.venue_id}")
        return self.client.get_all_notes(content={"venueid": self.venue_id})

    def create_submission_string(self, submissions: list[Any]) -> str:
        """Creates a string containing the id, title, and abstract of all submissions."""
        submission_info_string = ""
        for submission in submissions:
            paper = {
                "submission_id": submission.id,
                "title": submission.content["title"]["value"],
                "abstract": submission.content["abstract"]["value"],
            }
            submission_info_string += f"{paper}\n"
        return submission_info_string

    def fetch_relevant_papers(self, question: str) -> dict[str, Any]:
        """Get relevant papers for a given question using LLM."""
        submissions = self.get_submissions()
        submission_string = self.create_submission_string(submissions)

        if len(submission_string) > self.settings.parsing.chunk_size:
            chunks = [
                submission_string[i : i + self.settings.parsing.chunk_size]
                for i in range(0, len(submission_string), self.settings.parsing.chunk_size)
            ]
        else:
            chunks = [submission_string]
        relevant_papers = []
        for chunk in chunks:
            logger.info(f"Fetching relevant papers for question: {question}")
            relevant_papers += self._get_relevant_papers_chunk(question, chunk)
        subs = [s for s in submissions if s.id in set(relevant_papers)]
        self.download_papers(subs)
        return {sub.id: sub for sub in subs}

    def _get_relevant_papers_chunk(self, question: str, chunk: str) -> list[Any]:
        prompt = (
            chunk
            + "You are the helper model that aims to get up to 20 most relevant papers for the user's question. "
            + "User's question:\n"
        )

        schema = {
            "type": "object",
            "properties": {
                "suggested_papers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "submission_id": {"type": "string"},
                            "explanation": {"type": "string"},
                        },
                        "required": ["submission_id", "explanation"],
                    },
                },
                "reasoning_step_by_step": {"type": "string"},
            },
            "required": ["suggested_papers", "reasoning_step_by_step"],
        }
        response = completion(
            model=self.settings.llm,
            messages=[{"role": "user", "content": prompt + question}],
            response_format={"type": "json_object", "response_schema": schema},
            verbose=self.settings.verbosity,
        )

        content = json.loads(response.choices[0].message.content)
        return [p["submission_id"] for p in content["suggested_papers"]]

    def download_papers(self, submissions: list[Any]) -> None:
        """Download PDFs for given submissions."""
        downloaded_papers = Path(self.settings.paper_directory).rglob("*.pdf")
        downloaded_ids = [p.stem for p in downloaded_papers]
        logger.info("Downloading PDFs for relevant papers.")
        for submission in submissions:
            if submission.id not in downloaded_ids:
                self._download_pdf(submission)

    def _download_pdf(self, submission: Any) -> bool:
        """Download a single PDF."""
        pdf_link = f"https://openreview.net/{submission.content['pdf']['value']}"
        response = httpx.get(pdf_link)
        SUCCESS_CODE = 200
        if response.status_code == SUCCESS_CODE:
            with open(f"{self.settings.paper_directory}/{submission.id}.pdf", "wb") as f:
                f.write(response.content)
            return True
        logger.warning(f"Failed to download the PDF. Status code: {response.status_code}")
        return False

    async def aadd_docs(self, subs: dict[str, Any] | None = None) -> Docs:
        docs = Docs()
        for doc_path in Path(self.settings.paper_directory).rglob("*.pdf"):
            sub = subs.get(doc_path.stem) if subs is not None else None
            if sub:
                await docs.aadd(
                    doc_path,
                    settings=self.settings,
                    citation=sub.content["_bibtex"]["value"],
                    title=sub.content["title"]["value"],
                    doi="None",
                    authors=sub.content["authors"]["value"],
                )
            else:
                await docs.aadd(doc_path, settings=self.settings)
        return docs
