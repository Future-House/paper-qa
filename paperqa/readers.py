from langchain.text_splitter import RecursiveCharacterTextSplitter

TextSplitter = RecursiveCharacterTextSplitter


def parse_pdf(path, citation, key, chunk_chars=4000, overlap=50):
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
        if len(split) > chunk_chars:
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
            split = str(splits[chunk_chars:overlap])
            pages = [str(i + 1)]
    pdfFileObj.close()
    return splits, metadatas


def parse_txt(path, citation, key, chunk_chars=4000, overlap=50):

    try:
        with open(path) as f:
            doc = f.read()
    except UnicodeDecodeError as e:
        with open(path, encoding="utf-8", errors="ignore") as f:
            doc = f.read()
    # yo, no idea why but the texts are not split correctly
    text_splitter = TextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap)
    texts = text_splitter.split_text(doc)
    return texts, [dict(citation=citation, dockey=key, key=key)] * len(texts)


def read_doc(path, citation, key, chunk_chars=4000, overlap=50, disable_check=False):
    """Parse a document into chunks."""
    if path.endswith(".pdf"):
        return parse_pdf(path, citation, key, chunk_chars, overlap)
    elif path.endswith(".txt"):
        return parse_txt(path, citation, key, chunk_chars, overlap)
    else:
        raise ValueError(f"Unknown file type: {path} (expected .pdf or .txt).")
