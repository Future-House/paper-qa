import math
import string
import re
import asyncio

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
