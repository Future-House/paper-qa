import asyncio
import math
import re
import string
from typing import List
from functools import wraps

import pypdf

from .types import StrPath

def make_sync(f):
    @wraps(f)
    def with_sync(*args, **kwargs):
         # special case for jupyter notebooks
        if "get_ipython" in globals() or "google.colab" in sys.modules:
            import nest_asyncio

            nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(f(*args, **kwargs))
    return with_sync


def name_in_text(name: str, text: str) -> bool:
    sname = name.strip()
    pattern = r"\b({0})\b(?!\w)".format(re.escape(sname))
    if re.search(pattern, text):
        return True
    return False


def maybe_is_text(s: str, thresh: float = 2.5) -> bool:
    if len(s) == 0:
        return False
    # Calculate the entropy of the string
    entropy = 0.0
    for c in string.printable:
        p = s.count(c) / len(s)
        if p > 0:
            entropy += -p * math.log2(p)

    # Check if the entropy is within a reasonable range for text
    if entropy > thresh:
        return True
    return False


def strings_similarity(s1: str, s2: str) -> float:
    if len(s1) == 0 or len(s2) == 0:
        return 0
    # break the strings into words
    ss1 = set(s1.split())
    ss2 = set(s2.split())
    # return the similarity ratio
    return len(ss1.intersection(ss2)) / len(ss1.union(ss2))


def count_pdf_pages(file_path: StrPath) -> int:
    with open(file_path, "rb") as pdf_file:
        pdf_reader = pypdf.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
    return num_pages


def md5sum(file_path: StrPath) -> str:
    import hashlib

    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


async def gather_with_concurrency(n: int, *coros: List) -> List:
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
