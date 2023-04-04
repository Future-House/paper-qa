# This file gets PDF files from the user's Zotero library
import os
import warnings
from typing import Optional, Union, List
from pathlib import Path
from functools import lru_cache

from pyzotero import zotero

StrPath = Union[str, Path]


def default_library(library_type: str = "user") -> zotero.Zotero:
    """Returns a Zotero library object for the user's default library."""

    if "ZOTERO_USER_ID" not in os.environ:
        raise ValueError(
            "ZOTERO_USER_ID not set. Please navigate to"
            " https://www.zotero.org/settings/keys and get your user ID"
            " from the text 'Your userID for use in API calls is [XXXXXX]'."
            " Then, set the environment variable ZOTERO_USER_ID to this value."
        )

    if "ZOTERO_API_KEY" not in os.environ:
        raise ValueError(
            "ZOTERO_API_KEY not set. Please navigate to"
            " https://www.zotero.org/settings/keys and create a new API key"
            " with access to your library."
            " Then, set the environment variable ZOTERO_API_KEY to this value."
        )

    return zotero.Zotero(
        os.environ["ZOTERO_USER_ID"],
        library_type,
        os.environ["ZOTERO_API_KEY"],
    )


@lru_cache(maxsize=1)
def cached_walk(root: StrPath) -> List:
    return list(os.walk(root))


def get_pdf(item: dict, *, storage: Optional[StrPath] = None) -> Union[Path, None]:
    """Gets a PDF filename for a given Zotero key.

    If storage is None, or if the PDF is not found locally, the PDF will be downloaded to a local file.
    """
    key = item["key"]
    has_pdf = (
        "links" in item
        and "attachment" in item["links"]
        and (
            (
                "attachmentType" in item["links"]["attachment"]
                and item["links"]["attachment"]["attachmentType"] == "application/pdf"
            )
            or any(
                [
                    "attachmentType" in l and l["attachmentType"] == "application/pdf"
                    for l in item["links"]["attachment"]
                ]
            )
        )
    )

    if not has_pdf:
        return None

    # Replace with home directory:
    if storage is not None:
        raise NotImplementedError("Storage not implemented yet, due to inconsistencies in keys between local and remote.")
        # storage = Path(os.path.expanduser(storage))
        # if has_pdf:
        #     with open("test.txt", "w") as f:
        #         for root, dirs, files in cached_walk(storage):
        #             root = Path(root)
        #             for dir in dirs:
        #                 print(dir, file=f)
        #                 if dir == key:
        #                     print("HERE")
        #                     # Find all PDFs in this directory:
        #                     pdfs = [
        #                         root / dir / f
        #                         for f in os.listdir(root / dir)
        #                         if f.endswith(".pdf")
        #                     ]
        #                     if len(pdfs) > 0:
        #                         if len(pdfs) > 1:
        #                             warnings.warn(
        #                                 f"Found multiple PDFs for item {key}. Picking the first."
        #                             )
        #                         return pdfs[0]

    # We did not find the PDF locally. Thus, we need to download it.
