import math
import string
import re
import asyncio
import tempfile
import mimetypes

import requests
import pypdf

from .types import StrPath


def maybe_is_text(s, thresh=2.5):
    if len(s) == 0:
        return False
    # Calculate the entropy of the string
    entropy = 0
    for c in string.printable:
        p = s.count(c) / len(s)
        if p > 0:
            entropy += -p * math.log2(p)

    # Check if the entropy is within a reasonable range for text
    if entropy > thresh:
        return True
    return False


def maybe_is_code(s):
    if len(s) == 0:
        return False
    # Check if the string contains a lot of non-ascii characters
    if len([c for c in s if ord(c) > 128]) / len(s) > 0.1:
        return True
    return False


def strings_similarity(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return 0
    # break the strings into words
    s1 = set(s1.split())
    s2 = set(s2.split())
    # return the similarity ratio
    return len(s1.intersection(s2)) / len(s1.union(s2))


def maybe_is_truncated(s):
    punct = [".", "!", "?", '"']
    if s[-1] in punct:
        return False
    return True


def maybe_is_html(s):
    if len(s) == 0:
        return False
    # check for html tags
    if "<body" in s or "<html" in s or "<div" in s:
        return True


def count_pdf_pages(file_path: StrPath) -> int:
    with open(file_path, "rb") as pdf_file:
        pdf_reader = pypdf.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
    return num_pages


def md5sum(file_path: StrPath) -> str:
    import hashlib

    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


async def gather_with_concurrency(n, *coros):
    # https://stackoverflow.com/a/61478547/2392535
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


def guess_is_4xx(msg: str) -> bool:
    if re.search(r"4\d\d", msg):
        return True
    return False

def is_url(string: str) -> bool:
    """
    Check if a string is a URL.
    """
    # match web URLs
    url_pattern = r'^(?:(?:https?|ftp)://)?[\w\-]+(\.[\w\-]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?$'
    return re.match(url_pattern, string) is not None

def download_file(url: str) -> str:
    """
    Download a file from a URL.
    """
    response = requests.get(url)
    if response.status_code == 200:
        file_extension = mimetypes.guess_extension(response.headers.get('content-type'))
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    else:
        # raise error if the status code is not 200
        raise requests.exceptions.HTTPError(f"Error downloading file from {url}. Status code: {response.status_code}")
