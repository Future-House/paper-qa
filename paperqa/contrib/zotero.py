# This file gets PDF files from the user's Zotero library
import logging
import os
from pathlib import Path
from typing import List, Optional, Union, cast

from pydantic import BaseModel

try:
    from pyzotero import zotero
except ImportError:
    raise ImportError("Please install pyzotero: `pip install pyzotero`")
from ..paths import PAPERQA_DIR
from ..types import StrPath
from ..utils import count_pdf_pages


class ZoteroPaper(BaseModel):
    """A paper from Zotero.

    Attributes
    ----------
    key : str
        The citation key.
    title : str
        The title of the item.
    pdf : Path
        The path to the PDF for the item (pass to `paperqa.Docs`)
    num_pages : int
        The number of pages in the PDF.
    zotero_key : str
        The Zotero key for the item.
    details : dict
        The full item details from Zotero.
    """

    key: str
    title: str
    pdf: Path
    num_pages: int
    zotero_key: str
    details: dict

    def __str__(self) -> str:
        """Return the title of the paper."""
        return (
            f'ZoteroPaper(\n    key = "{self.key}",\n'
            f'title = "{self.title}",\n    pdf = "{self.pdf}",\n    '
            f'num_pages = {self.num_pages},\n    zotero_key = "{self.zotero_key}",\n    details = ...\n)'
        )


class ZoteroDB(zotero.Zotero):
    """An extension of pyzotero.zotero.Zotero to interface with paperqa.

    This class automatically reads in your `ZOTERO_USER_ID` and `ZOTERO_API_KEY`
    from your environment variables. If you do not have these, see
    step 2 of https://github.com/urschrei/pyzotero#quickstart.

    This class will download PDFs from your Zotero library and store them in
    `~/.paperqa/zotero` by default. To use this class, call the `iterate`
    method, which returns a `paperqa.Docs` object.
    """

    def __init__(
        self,
        *,
        library_type: str = "user",
        library_id: Optional[str] = None,
        api_key: Optional[str] = None,
        storage: Optional[StrPath] = None,
        **kwargs,
    ):
        self.logger = logging.getLogger("ZoteroDB")

        if library_id is None:
            self.logger.info("Attempting to get ZOTERO_USER_ID from `os.environ`...")
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
            self.logger.info("Attempting to get ZOTERO_API_KEY from `os.environ`...")
            if "ZOTERO_API_KEY" not in os.environ:
                raise ValueError(
                    "ZOTERO_API_KEY not set. Please navigate to"
                    " https://www.zotero.org/settings/keys and create a new API key"
                    " with access to your library."
                    " Then, set the environment variable ZOTERO_API_KEY to this value."
                )
            else:
                api_key = os.environ["ZOTERO_API_KEY"]

        self.logger.info(f"Using library ID: {library_id} with type: {library_type}.")

        if storage is None:
            storage = PAPERQA_DIR / "zotero"

        self.logger.info(f"Using cache location: {storage}")
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

        pdf_path: Path = Path(self.storage / (pdf_key + ".pdf"))  # type: ignore

        if not pdf_path.exists():
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"|  Downloading PDF for: {_get_citation_key(item)}")
            self.dump(pdf_key, pdf_path)

        return pdf_path

    def iterate(
        self,
        limit: int = 25,
        start: int = 0,
        q: Optional[str] = None,
        qmode: Optional[str] = None,
        since: Optional[str] = None,
        tag: Optional[str] = None,
        sort: Optional[str] = None,
        direction: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """Given a search query, this will lazily iterate over papers in a Zotero library, downloading PDFs as needed.

        This will download all PDFs in the query.
        For information on parameters, see
        https://pyzotero.readthedocs.io/en/latest/?badge=latest#zotero.Zotero.add_parameters
        For extra information on the query, see
        https://www.zotero.org/support/dev/web_api/v3/basics#search_syntax.

        For each item, it will return a `ZoteroPaper` object, which has the following fields:

            - `pdf`: The path to the PDF for the item (pass to `paperqa.Docs`)
            - `key`: The citation key.
            - `title`: The title of the item.
            - `details`: The full item details from Zotero.

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

        if collection_name is not None and len(query_kwargs) > 0:
            raise ValueError(
                "You cannot specify a `collection_name` and search query simultaneously!"
            )

        max_limit = 100

        items: List = []
        pdfs: List[Path] = []
        i = 0
        actual_i = 0
        num_remaining = limit

        collection_id = None
        if collection_name:
            collection_id = self._get_collection_id(
                collection_name
            )  # raise error if not found

        while num_remaining > 0:
            cur_limit = min(max_limit, num_remaining)
            self.logger.info(f"Downloading new batch of up to {cur_limit} papers.")

            if collection_id:
                _items = self._sliced_collection_items(
                    collection_id, limit=cur_limit, start=i
                )
            else:
                _items = self.top(**query_kwargs, limit=cur_limit, start=i)

            if len(_items) == 0:
                break
            i += cur_limit
            self.logger.info("Downloading PDFs.")
            _pdfs = [self.get_pdf(item) for item in _items]

            # Filter:
            for item, pdf in zip(_items, _pdfs):
                no_pdf = item is None or pdf is None
                is_duplicate = pdf in pdfs

                if no_pdf or is_duplicate:
                    continue
                pdf = cast(Path, pdf)
                title = item["data"]["title"] if "title" in item["data"] else ""
                if len(items) >= start:
                    yield ZoteroPaper(
                        key=_get_citation_key(item),
                        title=title,
                        pdf=pdf,
                        num_pages=count_pdf_pages(pdf),
                        details=item,
                        zotero_key=item["key"],
                    )
                    actual_i += 1

                items.append(item)
                pdfs.append(pdf)

            num_remaining = limit - actual_i

        self.logger.info("Finished downloading papers. Now creating Docs object.")

    def _sliced_collection_items(self, collection_id, limit, start):
        items = self.collection_items(collection_id)
        items = items[start:]
        if len(items) < limit:
            return items
        return items[:limit]

    def _get_collection_id(self, collection_name: str) -> str:
        """Get the collection id for a given collection name
            Raises ValueError if collection not found
        Args:
            collection_name (str): The name of the collection

        Returns:
            str: collection id
        """
        # specfic collection
        collections = self.collections()
        collection_id = ""

        for collection in collections:
            name = collection["data"]["name"]
            if name == collection_name:
                collection_id = collection["data"]["key"]
                break

        if collection_id:
            coll_items = self.collection_items(collection_id)
            self.logger.info(
                f"Collection '{collection_name}' found: {len(coll_items)} items"
            )

        else:
            raise ValueError(f"Collection '{collection_name}' not found")
        return collection_id


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
    date = item["data"]["date"]

    # Delete non-alphanumeric characters:
    short_title = "".join([c for c in short_title if c.isalnum()])
    last_name = "".join([c for c in last_name if c.isalnum()])
    date = "".join([c for c in date if c.isalnum()])

    return f"{last_name}_{short_title}_{date}_{item['key']}".replace(" ", "")


def _extract_pdf_key(item: dict) -> Union[str, None]:
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
