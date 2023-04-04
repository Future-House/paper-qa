# This file gets PDF files from the user's Zotero library
import os
from typing import Union
from pathlib import Path

from pyzotero import zotero

from ..docs import CACHE_PATH
from .. import Docs

StrPath = Union[str, Path]


class QAZotero(zotero.Zotero):
    """An extension of pyzotero.zotero.Zotero to interface with paperqa.
    
    This class automatically reads in your `ZOTERO_USER_ID` and `ZOTERO_API_KEY`
    from your environment variables. If you do not have these, see 
    step 2 of https://github.com/urschrei/pyzotero#quickstart.
    
    This class will download PDFs from your Zotero library and store them in
    `~/.paperqa/zotero` by default. To use this class, call the `gen_paperdb`
    method, which returns a `paperqa.Docs` object.
    """
    def __init__(
        self,
        *,
        library_type: str = "user",
        library_id=None,
        api_key=None,
        storage=None,
        **kwargs,
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

    def gen_paperdb(
        self,
        q=None,
        qmode=None,
        since=None,
        tag=None,
        sort=None,
        direction=None,
        limit=None,
        start=None,
    ):
        """Given a search query, this converts the Zotero library to a `paperqa.docs.Docs` object.

        This will download all PDFs in the query.
        For information on parameters, see
        https://pyzotero.readthedocs.io/en/latest/?badge=latest#zotero.Zotero.add_parameters
        For extra information on the query, see
        https://www.zotero.org/support/dev/web_api/v3/basics#search_syntax.

        Parameters
        ----------
        q : str, optional
            Quick search query. Searches only titles and creator fields by default.
            Control with `qmode`.
        qmode : str, optional
            Quick search mode. One of `titleCreatorYear` or `everything`.
        since : int, optional
            Only return objects modified after the specified library version.
        tag : str, optional
            Tag search. Can use `AND` or `OR` to combine tags.
        sort : str, optional
            The name of the field to sort by. One of dateAdded, dateModified,
            title, creator, itemType, date, publisher, publicationTitle,
            journalAbbreviation, language, accessDate, libraryCatalog, callNumber,
            rights, addedBy, numItems (tags).
        direction : str, optional
            asc or desc.
        limit : int, optional
            The maximum number of items to return. Default is 25. You may use the `start`
            parameter to continue where you left off.
        start : int, optional
            The index of the first item to return. Default is 0.
        """
        query_kwargs = {}
        if q is not None:
            query_kwargs["q"] = q
        if qmode is not None:
            query_kwargs["qmode"] = qmode
        if since is not None:
            query_kwargs["since"] = since
        if tag is not None:
            query_kwargs["tag"] = tag
        if sort is not None:
            query_kwargs["sort"] = sort
        if direction is not None:
            query_kwargs["direction"] = direction

        # (We deal with start and limit manually)
        if limit is None:
            limit = 25

        if start is None:
            start = 0

        max_limit = 100

        items = []
        pdfs = []
        citations = []
        keys = []
        num_remaining = limit - len(items)

        while num_remaining > 0:
            _items = self.top(
                **query_kwargs, limit=min(max_limit, num_remaining), start=start
            )
            _pdfs = [self.get_pdf(item) for item in _items]

            # Filter:
            new_items = []
            for (item, pdf) in zip(_items, _pdfs):
                no_pdf = item is None or pdf is None
                is_duplicate = pdf in pdfs

                if no_pdf or is_duplicate:
                    continue

                new_items.append(item)
                items.append(item)
                pdfs.append(pdf)


            citations.extend([_get_citation_key(item) for item in new_items])

            num_remaining = limit - len(items)
            start += len(new_items)

        docs = Docs()

        for i in range(len(items)):
            docs.add(path=pdfs[i], citation=citations[i], key=items[i]["key"])

        return docs


def _get_citation_key(item: dict) -> str:
    if (
        "data" not in item
        or "creators" not in item["data"]
        or len(item["data"]["creators"]) == 0
        or "lastName" not in item["data"]["creators"][0]
        or "title" not in item["data"]
        or "date" not in item["data"]
    ):
        return item["key"]

    last_name = item["data"]["creators"][0]["lastName"]
    short_title = "".join(item["data"]["title"].split(" ")[:3])
    year = item["data"]["date"]

    return f"{last_name}{short_title}{year}_{item['key']}"


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
