# This file gets PDF files from the user's Zotero library
import os
from typing import Union
from pathlib import Path

from pyzotero import zotero

from ..docs import CACHE_PATH

StrPath = Union[str, Path]


class QAZotero(zotero.Zotero):
    def __init__(
        self, *, library_type: str = "user", library_id=None, api_key=None, storage=None, **kwargs
    ):
        if library_id is None:
            if "ZOTERO_USER_ID" not in os.environ:
                raise ValueError(
                    "ZOTERO_USER_ID not set. Please navigate to"
                    " https://www.zotero.org/settings/keys and get your user ID"
                    " from the text 'Your userID for use in API calls is [XXXXXX]'."
                    " Then, set the environment variable ZOTERO_USER_ID to this value."
                )
            else:
                library_id = os.environ["ZOTERO_USER_ID"]

        if api_key is None:
            if "ZOTERO_API_KEY" not in os.environ:
                raise ValueError(
                    "ZOTERO_API_KEY not set. Please navigate to"
                    " https://www.zotero.org/settings/keys and create a new API key"
                    " with access to your library."
                    " Then, set the environment variable ZOTERO_API_KEY to this value."
                )
            else:
                api_key = os.environ["ZOTERO_API_KEY"]

        if storage is None:
            storage = CACHE_PATH.parent / "zotero"
        
        self.storage = storage

        super().__init__(
            library_type=library_type, library_id=library_id, api_key=api_key, **kwargs
        )

    def get_pdf(self, item: dict) -> Union[Path, None]:
        """Gets a filename for a given Zotero key for a PDF.

        If the PDF is not found locally, the PDF will be downloaded to a local file at the correct key.
        If no PDF exists for the file, None is returned.

        Parameters
        ----------
        item : dict
            An item from `pyzotero`. Should have a `key` field, and also have an entry
            `links->attachment->attachmentType == application/pdf`.
        """
        if type(item) != dict:
            raise TypeError("Pass the full item of the paper. The item must be a dict.")

        pdf_key = _extract_pdf_key(item)

        if pdf_key is None:
            return None

        pdf_path = self.storage / (pdf_key + ".pdf")

        if not pdf_path.exists():
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            self.dump(pdf_key, pdf_path)

        return pdf_path
        

def _extract_pdf_key(item: dict) -> str:
    """Extract the PDF key from a Zotero item."""

    if "links" not in item:
        return None

    if "attachment" not in item["links"]:
        return None

    attachments = item["links"]["attachment"]

    if type(attachments) != dict:
        # Find first attachment with attachmentType == application/pdf:
        for attachment in attachments:
            # TODO: This assumes there's only one PDF attachment.
            if attachment["attachmentType"] == "application/pdf":
                break
    else:
        attachment = attachments

    if "attachmentType" not in attachment:
        return None

    if attachment["attachmentType"] != "application/pdf":
        return None

    return attachment["href"].split("/")[-1]
