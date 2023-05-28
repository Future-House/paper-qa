# Paper QA

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/whitead/paper-qa)
[![tests](https://github.com/whitead/paper-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/whitead/paper-qa)
[![PyPI version](https://badge.fury.io/py/paper-qa.svg)](https://badge.fury.io/py/paper-qa)

This is a minimal package for doing question and answering from
PDFs or text files (which can be raw HTML). It strives to give very good answers, with no hallucinations, by grounding responses with in-text citations. It uses [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) with a vector DB called [FAISS](https://github.com/facebookresearch/faiss) to embed and search documents. [langchain](https://github.com/hwchase17/langchain) helps
generate answers.

It uses the process shown below:

```
embed docs into vectors -> embed query into vector -> search for top k passages in docs

create summary of each passage relevant to query -> put summaries into prompt -> generate answer
```

<img src="https://user-images.githubusercontent.com/908389/230854097-8fa96768-c694-45c0-bb04-3a7386facef3.jpeg" width="600" alt="Process of vector search, refinement, and answer with context">

## Output Example

Question: How can carbon nanotubes be manufactured at a large scale?

Carbon nanotubes can be manufactured at a large scale using the electric-arc technique (Journet6644). This technique involves creating an arc between two electrodes in a reactor under a helium atmosphere and using a mixture of a metallic catalyst and graphite powder in the anode. Yields of 80% of entangled carbon filaments can be achieved, which consist of smaller aligned SWNTs self-organized into bundle-like crystallites (Journet6644). Additionally, carbon nanotubes can be synthesized and self-assembled using various methods such as DNA-mediated self-assembly, nanoparticle-assisted alignment, chemical self-assembly, and electro-addressed functionalization (Tulevski2007). These methods have been used to fabricate large-area nanostructured arrays, high-density integration, and freestanding networks (Tulevski2007). 98% semiconducting CNT network solution can also be used and is separated from metallic nanotubes using a density gradient ultracentrifugation approach (Chen2014). The substrate is incubated in the solution and then rinsed with deionized water and dried with N2 air gun, leaving a uniform carbon network (Chen2014).

### References

Journet6644: Journet, Catherine, et al. "Large-scale production of single-walled carbon nanotubes by the electric-arc technique." nature 388.6644 (1997): 756-758.

Tulevski2007: Tulevski, George S., et al. "Chemically assisted directed assembly of carbon nanotubes for the fabrication of large-scale device arrays." Journal of the American Chemical Society 129.39 (2007): 11964-11968.

Chen2014: Chen, Haitian, et al. "Large-scale complementary macroelectronics using hybrid integration of carbon nanotubes and IGZO thin-film transistors." Nature communications 5.1 (2014): 4097.


## Hugging Face Demo

[Hugging Face Demo](https://huggingface.co/spaces/whitead/paper-qa)

## Install

Install with pip:

```bash
pip install paper-qa
```

## Usage

Make sure you have set your OPENAI_API_KEY environment variable to your [openai api key](https://platform.openai.com/account/api-keys)

To use paper-qa, you need to have a list of paths (valid extensions include: .pdf, .txt) and a list of citations (strings) that correspond to the paths. You can then use the `Docs` class to add the documents and then query them. If you don't have citations, `Docs` will try to guess them from the first page of your docs.

```python

from paperqa import Docs

# get a list of paths

docs = Docs()
for d in my_docs:
    docs.add(d)

answer = docs.query("What manufacturing challenges are unique to bispecific antibodies?")
print(answer.formatted_answer)
```

The answer object has the following attributes: `formatted_answer`, `answer` (answer alone), `question`, `context` (the summaries of passages found for answer), `references` (the docs from which the passages came), and `passages` which contain the raw text of the passages as a dictionary.

### Choosing Model

By default, it uses a hybrid of `gpt-3.5-turbo` and `gpt-4`. If you don't have gpt-4 access or would like to save money, you can adjust:

```py
docs = Docs(llm='gpt-3.5-turbo')
```

or you can use any other model available in [langchain](https://github.com/hwchase17/langchain):

```py
from langchain.llms import Anthropic, OpenAIChat
model = OpenAIChat(model='gpt-4')
summary_model = Anthropic(model="claude-instant-v1-100k", anthropic_api_key="my-api-key")
docs = Docs(llm=model, summary_llm=summary_model)
```


#### Locally Hosted

You can also use any other models (or embeddings) available in [langchain](https://github.com/hwchase17/langchain). Here's an example of using `llama.cpp` to have locally hosted paper-qa:

```py
from paperqa import Docs
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import LlamaCppEmbeddings

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./ggml-model-q4_0.bin", callback_manager=callback_manager
)
embeddings = LlamaCppEmbeddings(model_path="./ggml-model-q4_0.bin")

docs = Docs(llm=llm, embeddings=embeddings)

keyword_search = 'bispecific antibody manufacture'
papers = paperscraper.search_papers(keyword_search, limit=2)
for path,data in papers.items():
    try:
        docs.add(path,chunk_chars=500)
    except ValueError as e:
        print('Could not read', path, e)

answer = docs.query("What manufacturing challenges are unique to bispecific antibodies?")
print(answer)
```

### Adjusting number of sources

You can adjust the numbers of sources (passages of text) to reduce token usage or add more context. `k` refers to the top k most relevant and diverse (may from different sources) passages. Each passage is sent to the LLM to summarize, or determine if it is irrelevant. After this step, a limit of `max_sources` is applied so that the final answer can fit into the LLM context window. Thus, `k` > `max_sources`  and `max_sources` is the number of sources used in the final answer.

```python
docs.query("What manufacturing challenges are unique to bispecific antibodies?", k = 5, max_sources = 2)
```

### Using Code or HTML

You do not need to use papers -- you can use code or raw HTML. Note that this tool is focused on answering questions, so it won't do well at writing code. One note is that the tool cannot infer citations from code, so you will need to provide them yourself.

```python

import glob

source_files = glob.glob('**/*.js')

docs = Docs()
for f in source_files:
    # this assumes the file names are unique in code
    docs.add(f, citation='File ' + os.path.name(f), key=os.path.name(f))
answer = docs.query("Where is the search bar in the header defined?")
print(answer)
```

## Notebooks

If you want to use this in an jupyter notebook or colab, you need to run the following command:

```python
import nest_asyncio
nest_asyncio.apply()
```

Also - if you know how to make this automated, please let me know!

## Agents (experimental)

You can try to automate the collection of papers and assessment of correctness of papers using an agent. This is experimental and requires installation of [paper-scraper](https://github.com/blackadad/paper-scraper).

```python

docs = paperqa.Docs()
answer = paperqa.run_agent(docs, 'What compounds target AKT1')
print(answer)
```

## Where do I get papers?

Well that's a really good question! It's probably best to just download PDFs of papers you think will help answer your question and start from there.

### Zotero

If you use [Zotero](https://www.zotero.org/) to organize your personal bibliography,
you can use the `paperqa.contrib.ZoteroDB` to query papers from your library,
which relies on [pyzotero](https://github.com/urschrei/pyzotero).

Install `pyzotero` to use this feature:

```bash
pip install pyzotero
```

First, note that `paperqa` parses the PDFs of papers to store in the database,
so all relevant papers should have PDFs stored inside your database.
You can get Zotero to automatically do this by highlighting the references
you wish to retrieve, right clicking, and selecting *"Find Available PDFs"*.
You can also manually drag-and-drop PDFs onto each reference.

To download papers, you need to get an API key for your account.

1. Get your library ID, and set it as the environment variable `ZOTERO_USER_ID`.
    - For personal libraries, this ID is given [here](https://www.zotero.org/settings/keys) at the part "*Your userID for use in API calls is XXXXXX*".
    - For group libraries, go to your group page `https://www.zotero.org/groups/groupname`, and hover over the settings link. The ID is the integer after /groups/. (*h/t pyzotero!*)
2. Create a new API key [here](https://www.zotero.org/settings/keys/new) and set it as the environment variable `ZOTERO_API_KEY`.
    - The key will need read access to the library.

With this, we can download papers from our library and add them to `paperqa`:

```py
from paperqa.contrib import ZoteroDB

docs = paperqa.Docs()
zotero = ZoteroDB(library_type="user")  # "group" if group library

for item in zotero.iterate(limit=20):
    if item.num_pages > 30:
        continue  # skip long papers
    docs.add(item.pdf, key=item.key)
```

which will download the first 20 papers in your Zotero database and add
them to the `Docs` object.

We can also do specific queries of our Zotero library and iterate over the results:

```py
for item in zotero.iterate(
        q="large language models",
        qmode="everything",
        sort="date",
        direction="desc",
        limit=100,
):
    print("Adding", item.title)
    docs.add(item.pdf, key=item.key)
```

You can read more about the search syntax by typing `zotero.iterate?` in IPython.

### Paper Scraper

If you want to search for papers outside of your own collection, I've found an unrelated project called [paper-scraper](https://github.com/blackadad/paper-scraper) that looks
like it might help. But beware, this project looks like it uses some scraping tools that may violate publisher's rights or be in a gray area of legality.

```py
keyword_search = 'bispecific antibody manufacture'
papers = paperscraper.search_papers(keyword_search)
docs = paperqa.Docs()
for path,data in papers.items():
    try:
        docs.add(path)
    except ValueError as e:
        # sometimes this happens if PDFs aren't downloaded or readable
        print('Could not read', path, e)
answer = docs.query("What manufacturing challenges are unique to bispecific antibodies?")
print(answer)
```

## FAQ

### How is this different from LlamaIndex?

It's not that different! This is similar to the tree response method in LlamaIndex. I just have included some prompts I find useful, readers that give page numbers/line numbers, and am focused on one task - answering technical questions with cited sources.

### How is this different from LangChain?

It's not! We use langchain to abstract the LLMS, and the process is very similar to the `map_reduce` chain in LangChain.

### Caching

This code will cache responses from LLMS by default in `$HOME/.paperqa/llm_cache.db`. Delete this file to clear the cache.

### Can I use different LLMs?

Yes, you can use any LLMs from [langchain](https://langchain.readthedocs.io/) by passing the `llm` argument to the `Docs` class. You can use different LLMs for summarization and for question answering too.

### Where do the documents come from?

You can provide your own. I use some of my own code to pull papers from Google Scholar. This code is not included because it may enable people to violate Google's terms of service and publisher's terms of service.

### Can I save or load?

The `Docs` class can be pickled and unpickled. This is useful if you want to save the embeddings of the documents and then load them later.

```python
import pickle

# save
with open("my_docs.pkl", "wb") as f:
    pickle.dump(docs, f)

# load
with open("my_docs.pkl", "rb") as f:
    docs = pickle.load(f)
```

### PDF Reading Options

By default [PyPDF](https://pypi.org/project/pypdf/) is used since it's pure python and easy to install. For faster PDF reading, paper-qa will detect and use [PymuPDF (fitz)](https://pymupdf.readthedocs.io/en/latest/):

```sh
pip install pymupdf
```

### Callbacks

TODO
