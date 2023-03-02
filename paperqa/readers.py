from .utils import maybe_is_code
from html2text import html2text

from langchain.text_splitter import TokenTextSplitter

TextSplitter = TokenTextSplitter


def parse_pdf(path, citation, key, chunk_chars=2000, overlap=50):
    import pypdf

    pdfFileObj = open(path, "rb")
    pdfReader = pypdf.PdfReader(pdfFileObj)
    splits = []
    split = ""
    pages = []
    metadatas = []
    for i, page in enumerate(pdfReader.pages):
        split += page.extract_text()
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            splits.append(split[:chunk_chars])
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            metadatas.append(
                dict(
                    citation=citation,
                    dockey=key,
                    key=f"{key} pages {pg}",
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
    if len(split) > overlap:
        splits.append(split[:chunk_chars])
        pg = "-".join([pages[0], pages[-1]])
        metadatas.append(
            dict(
                citation=citation,
                dockey=key,
                key=f"{key} pages {pg}",
            )
        )
    pdfFileObj.close()
    return splits, metadatas


def parse_txt(path, citation, key, chunk_chars=2000, overlap=50, html=False):

    try:
        with open(path) as f:
            doc = f.read()
    except UnicodeDecodeError as e:
        with open(path, encoding="utf-8", errors="ignore") as f:
            doc = f.read()
    if html:
        doc = html2text(doc)
    # yo, no idea why but the texts are not split correctly
    text_splitter = TextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap)
    texts = text_splitter.split_text(doc)
    return texts, [dict(citation=citation, dockey=key, key=key)] * len(texts)


def parse_code_txt(path, citation, key, chunk_chars=2000, overlap=50):
    """Parse a document into chunks, based on line numbers (for code)."""

    splits = []
    split = ""
    metadatas = []
    last_line = 0

    with open(path) as f:
        for i, line in enumerate(f):
            split += line
            if len(split) > chunk_chars:
                splits.append(split[:chunk_chars])
                metadatas.append(
                    dict(
                        citation=citation,
                        dockey=key,
                        key=f"{key} lines {last_line}-{i}",
                    )
                )
                split = split[chunk_chars - overlap :]
                last_line = i
    if len(split) > overlap:
        splits.append(split[:chunk_chars])
        metadatas.append(
            dict(
                citation=citation,
                dockey=key,
                key=f"{key} lines {last_line}-{i}",
            )
        )
    return splits, metadatas


def read_doc(path, citation, key, chunk_chars=3000, overlap=100, disable_check=False):
    """Parse a document into chunks."""
    if path.endswith(".pdf"):
        return parse_pdf(path, citation, key, chunk_chars, overlap)
    elif path.endswith(".txt"):
        return parse_txt(path, citation, key, chunk_chars, overlap)
    elif path.endswith(".html"):
        return parse_txt(path, citation, key, chunk_chars, overlap, html=True)
    else:
        return parse_code_txt(path, citation, key, chunk_chars, overlap)
