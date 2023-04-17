import math
import string
import PyPDF4
import aiohttp
import re

from .types import StrPath

def clean_citation(citation):
    citation = citation.replace("None", "")
    citation = re.sub(r"\s+", " ", citation)
    citation = re.sub(r", \?{4}, pp. \?{2}–\?{2}\.", "", citation)
    citation = re.sub(r"\?{4}", "", citation)
    citation = citation.replace(". .", ".").strip()
    citation = citation.replace('..', '.').replace('"  .','" ').strip()
    return citation

def get_first_non_none(match):
    for group in match.groups():
        if group is not None:
            return group

def zotero_clipboard_to_mla_citations(bibtex_clipboard_string):
    """
    Convert a Zotero clipboard string to a list of MLA citations.
    """
    bibtex_list = bibtex_clipboard_string.split("\n\n") # split at double linebreaks
    mla_citations = []

    for i, bibtex in enumerate(bibtex_list):
        mla_citations.append(f'[{i}] ' + bibtex_to_mla(bibtex))

    return mla_citations

# define a function that can take in a string of authors such as 'Doe, John, Smith, Jane' and output the number of authors
def count_authors(authors):
    if authors:
        return len(authors.split(", "))//2
    else:
        return 0

# define a function that can take in a fields['author'] string and output the authors in the correct format, for example take 'Doe, John and Smith, Jane' and output 'Doe, John, and Jane Smith'
def format_two_authors_mla(authors):
    if authors:
        authors = authors.split(", ")
        tmp = authors[1].split('and')
        return authors[0] + f", {tmp[0].strip()}, and " + authors[2] + ' ' + tmp[1].strip()
    else:
        return "Unknown"

# define a function that can take in an authors object such as 'Doe, John, Smith, Jane, Jones, Joe' and output the first author only followed by et al, for example 'Doe, John, et al.'
def format_first_author_mla(authors):
    if authors:
        authors = authors.split(", ")
        return authors[0] + f", {authors[1]}, et al."
    else:
        return "Unknown"

def bibtex_to_mla(bibtex_entry):
    """
    Convert a BibTeX entry to a citation string.

    Args:
        bibtex_entry (str): BibTeX entry
    
    Returns:
        str: Citation string (MLA format)
    """
    # Define patterns for extracting the required fields
    patterns = {
        'author': r"author\s*=\s*{(.*?)}",
        'title': r"title\s*=\s*{(.*?)}",
        'booktitle': (r"booktitle\s*=\s*{(.*?)}|journal\s*=\s*{(.*?)}"),
        'year': r"year\s*=\s*{(\d{4})}", 
        'pages': r"pages\s*=\s*{(\d+)--(\d+)}",
        'volume': r"volume\s*=\s*{(\d+)}",
        'doi': r"doi\s*=\s*{(\S+)}",
        'url': r"url\s*=\s*{(\S+)}"
    }    

    # Extract fields using regex
    fields = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, bibtex_entry)
        if match:
            fields[key] = match.groups() if key == 'pages' else get_first_non_none(match)
        else:
            fields[key] = None        

    if fields['author']:
        authors = fields['author'].replace(" and ", ", ")
        num_authors = count_authors(fields['author'])
        # if the authors string more than 2 commas...
        if num_authors > 2:
            authors = format_first_author_mla(authors)
        elif num_authors == 2:
            authors = format_two_authors_mla(fields['author'])
        else:
            authors = authors
    else:
        authors = 'Unknown'

    if fields['pages']:
        start_page, end_page = fields['pages']
    else:
        start_page, end_page = '??', '??'

    if not fields['year']:
        fields['year'] = '????'

    # Format citation
    citation = f'{authors}. "{fields["title"]}." {fields["booktitle"]} {fields["volume"]}, {fields["year"]}, pp. {start_page}–{end_page}. {fields["url"]}.'    
    
    # clean up the citation
    citation = clean_citation(citation)

    return citation
    
async def download_pdf(url, output_file='temp.pdf'): # TODO: figure out how to download pdf's faster in python
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                with open(output_file, 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
            else:
                print(f"Error: Unable to download the PDF file. HTTP status: {response.status}")


def start_asking(docs):
    """
    Start asking questions in the CLI.

    Args:
        docs (Docs): paperqa.docs.Docs object
    """
    while True:
        try:
            # Ask for query
            query = input("Enter query: ")
            answer = docs.query(query)
            print(answer.formatted_answer)
        except KeyboardInterrupt:
            break  

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
