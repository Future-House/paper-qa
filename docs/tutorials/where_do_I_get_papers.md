# Where to get papers

## OpenReview

You can use papers from [https://openreview.net/](https://openreview.net/) as your database!
Here's a helper that fetches a list of all papers from a selected conference (like ICLR, ICML, NeurIPS), queries this list to find relevant papers using LLM, and downloads those relevant papers to a local directory which can be used with paper-qa on the next step. Install `openreview-py` with

```bash
pip install paper-qa[openreview]
```

and get your username and password from the website. You can put them into `.env` file under `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` variables, or pass them in the code directly.

```python
from paperqa import Settings
from paperqa.contrib.openreview_paper_helper import OpenReviewPaperHelper

# these settings require gemini api key you can get from https://aistudio.google.com/
# import os; os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
# 1Mil context window helps to suggest papers. These settings are not required, but useful for an initial setup.
settings = Settings.from_name("openreview")
helper = OpenReviewPaperHelper(settings, venue_id="ICLR.cc/2025/Conference")
# if you don't know venue_id you can find it via
# helper.get_venues()

# Now we can query LLM to select relevant papers and download PDFs
question = "What is the progress on brain activity research?"

submissions = helper.fetch_relevant_papers(question)

# There's also a function that saves tokens by using openreview metadata for citations
docs = await helper.aadd_docs(submissions)

# Now you can continue asking like in the [main tutorial](../../README.md)
session = docs.query(question, settings=settings)
print(session.answer)
```

## Zotero

_It's been a while since we've tested this - so let us know if it runs into issues!_

If you use [Zotero](https://www.zotero.org/) to organize your personal bibliography,
you can use the `paperqa.contrib.ZoteroDB` to query papers from your library,
which relies on [pyzotero](https://github.com/urschrei/pyzotero).

Install `pyzotero` via the `zotero` extra for this feature:

```bash
pip install paper-qa[zotero]
```

First, note that PaperQA2 parses the PDFs of papers to store in the database,
so all relevant papers should have PDFs stored inside your database.
You can get Zotero to automatically do this by highlighting the references
you wish to retrieve, right clicking, and selecting _"Find Available PDFs"_.
You can also manually drag-and-drop PDFs onto each reference.

To download papers, you need to get an API key for your account.

1. Get your library ID, and set it as the environment variable `ZOTERO_USER_ID`.
   - For personal libraries, this ID is given [here](https://www.zotero.org/settings/keys) at the part "_Your userID for use in API calls is XXXXXX_".
   - For group libraries, go to your group page `https://www.zotero.org/groups/groupname`, and hover over the settings link. The ID is the integer after /groups/. (_h/t pyzotero!_)
2. Create a new API key [here](https://www.zotero.org/settings/keys/new) and set it as the environment variable `ZOTERO_API_KEY`.
   - The key will need read access to the library.

With this, we can download papers from our library and add them to PaperQA2:

```python
from paperqa import Docs
from paperqa.contrib import ZoteroDB

docs = Docs()
zotero = ZoteroDB(library_type="user")  # "group" if group library

for item in zotero.iterate(limit=20):
    if item.num_pages > 30:
        continue  # skip long papers
    docs.add(item.pdf, docname=item.key)
```

which will download the first 20 papers in your Zotero database and add
them to the `Docs` object.

We can also do specific queries of our Zotero library and iterate over the results:

```python
for item in zotero.iterate(
    q="large language models",
    qmode="everything",
    sort="date",
    direction="desc",
    limit=100,
):
    print("Adding", item.title)
    docs.add(item.pdf, docname=item.key)
```

You can read more about the search syntax by typing `zotero.iterate?` in IPython.

## Paper Scraper

If you want to search for papers outside of your own collection, I've found an unrelated project called [paper-scraper](https://github.com/blackadad/paper-scraper) that looks
like it might help. But beware, this project looks like it uses some scraping tools that may violate publisher's rights or be in a gray area of legality.

```python
from paperqa import Docs

keyword_search = "bispecific antibody manufacture"
papers = paperscraper.search_papers(keyword_search)
docs = Docs()
for path, data in papers.items():
    try:
        docs.add(path)
    except ValueError as e:
        # sometimes this happens if PDFs aren't downloaded or readable
        print("Could not read", path, e)
session = docs.query(
    "What manufacturing challenges are unique to bispecific antibodies?"
)
print(session)
```
