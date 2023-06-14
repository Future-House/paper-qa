# Paper QA- [Paper QA](#paper-qa)
- [Paper QA- Paper QA](#paper-qa--paper-qa)
  - [Output Example](#output-example)
    - [References](#references)
  - [Hugging Face Demo](#hugging-face-demo)
  - [Install](#install)
  - [Usage](#usage)
    - [Adding Documents](#adding-documents)
    - [Choosing Model](#choosing-model)
    - [Adjusting number of sources](#adjusting-number-of-sources)
    - [Using Code or HTML](#using-code-or-html)
  - [Version 3 Changes](#version-3-changes)
    - [New Features](#new-features)
    - [Naming](#naming)
    - [Breaking Changes](#breaking-changes)
  - [Notebooks](#notebooks)
  - [Where do I get papers?](#where-do-i-get-papers)
    - [Zotero](#zotero)
    - [Paper Scraper](#paper-scraper)
  - [PDF Reading Options](#pdf-reading-options)
  - [Typewriter View](#typewriter-view)
  - [LLM/Embedding Caching](#llmembedding-caching)
    - [Caching Embeddings](#caching-embeddings)
  - [Customizing Prompts](#customizing-prompts)
    - [Pre and Post Prompts](#pre-and-post-prompts)
  - [FAQ](#faq)
    - [How is this different from LlamaIndex?](#how-is-this-different-from-llamaindex)
    - [How is this different from LangChain?](#how-is-this-different-from-langchain)
    - [Can I use different LLMs?](#can-i-use-different-llms)
    - [Where do the documents come from?](#where-do-the-documents-come-from)
    - [Can I save or load?](#can-i-save-or-load)


[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/whitead/paper-qa)
[![tests](https://github.com/whitead/paper-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/whitead/paper-qa)
[![PyPI version](https://badge.fury.io/py/paper-qa.svg)](https://badge.fury.io/py/paper-qa)

This is a minimal package for doing question and answering from
PDFs or text files (which can be raw HTML). It strives to give very good answers, with no hallucinations, by grounding responses with in-text citations.

By default, it uses [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) with a vector DB called [FAISS](https://github.com/facebookresearch/faiss) to embed and search documents. However, via [langchain](https://github.com/hwchase17/langchain) you can use open-source models or embeddings (see details below).

PaperQA uses the process shown below:

1. embed docs into vectors
2. embed query into vector
3. search for top k passages in docs
4. create summary of each passage relevant to query
5. put summaries into prompt
6. generate answer with prompt

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

### Adding Documents

`add` will add from paths. You can also use `add_file` (expects a file object) or `add_url` to work with other sources.

### Choosing Model

By default, it uses a hybrid of `gpt-3.5-turbo` and `gpt-4`. If you don't have gpt-4 access or would like to save money, you can adjust:

```py
docs = Docs(llm='gpt-3.5-turbo')
```

or you can use any other model available in [langchain](https://github.com/hwchase17/langchain):

```py
from langchain.chat_models import ChatAnthropic, ChatOpenAI
model = ChatOpenAI(model='gpt-4')
summary_model = ChatAnthropic(model="claude-instant-v1-100k", anthropic_api_key="my-api-key")
docs = Docs(llm=model, summary_llm=summary_model)
```


#### Locally Hosted

You can also use any other models (or embeddings) available in [langchain](https://github.com/hwchase17/langchain). Here's an example of using `llama.cpp` to have locally hosted paper-qa:

```py
import paperscraper
from paperqa import Docs
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./ggml-model-q4_0.bin", callbacks=[StreamingStdOutCallbackHandler()]
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
    docs.add(f, citation='File ' + os.path.name(f), docname=os.path.name(f))
answer = docs.query("Where is the search bar in the header defined?")
print(answer)
```

## Version 3 Changes

Version 3 includes many changes to type the code, make it more focused/modular, and enable performance to very large numbers of documents. The major breaking changes are documented below:


### New Features

The following new features are in v3:

1. Memory is now possible in `query` by setting `Docs(memory=True)` - this means follow-up questions will have a record of the previous question and answer.
2. `add_url` and `add_file` are now supported for adding from URLs and file objects
3. Prompts can be customized, and now can be executed pre and post query
4. Consistent use of `dockey` and `docname` for unique and natural language names enable better tracking with external databases
5. Texts and embeddings are no longer required to be part of `Docs` object, so you can use external databases or other strategies to manage them
6. Various simplifications, bug fixes, and performance improvements

### Naming

The following table shows the old names and the new names:

| Old Name | New Name | Explanation |
| :--- | :---: | ---: |
| `key` | `name` | Name is a natural language name for text. |
| `dockey` | `docname` | Docname is a natural language name for a document. |
| `hash` | `dockey` | Dockey is a unique identifier for the document. |


### Breaking Changes


#### Pickled objects

The pickled objects are not compatible with the new version.

#### Agents

The agent functionality has been removed, as it's not a core focus of the library

#### Caching

Caching has been removed because it's not a core focus of the library. See FAQ below for how to use caching.

#### Answers

Answers will not include passages, but instead return dockeys that can be used to retrieve the passages. Tokens/cost will also not be counted since that is built into langchain by default (see below for an example).

#### Search Query

The search query chain has been removed. You can use langchain directly to do this.

## Notebooks

If you want to use this in an jupyter notebook or colab, you need to run the following command:

```python
import nest_asyncio
nest_asyncio.apply()
```

Also - if you know how to make this automated, please let me know!

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
    docs.add(item.pdf, docname=item.key)
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
    docs.add(item.pdf, docname=item.key)
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

## PDF Reading Options

By default [PyPDF](https://pypi.org/project/pypdf/) is used since it's pure python and easy to install. For faster PDF reading, paper-qa will detect and use [PymuPDF (fitz)](https://pymupdf.readthedocs.io/en/latest/):

```sh
pip install pymupdf
```

## Typewriter View

To stream the completions as they occur (giving that ChatGPT typewriter look), you can simply instantiate models with those properties:

```python
from paperqa import Docs
from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

my_llm = ChatOpenAI(callbacks=[StreamingStdOutCallbackHandler()], streaming=True)
docs = Docs(llm=my_llm)
```

## LLM/Embedding Caching

You can using the builtin langchain caching capabilities. Just run this code at the top of yours:

```py
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
```

### Caching Embeddings

In general, embeddings are cached when you pickle a `Docs` regardless of what vector store you use. If you would like to manage caching embeddings via an external database or other strategy,
you can populate a `Docs` object directly via
the `add_texts` object. That can take chunked texts and documents, which are serializable objects, to populate `Docs`.

You also can simply use a separate vector database by setting the `doc_index` and `texts_index` explicitly when building the `Docs` object.

## Customizing Prompts

You can customize any of the prompts, using the `PromptCollection` class. For example, if you want to change the prompt for the question, you can do:

```python
from paperqa import Docs, Answer, PromptCollection
from langchain.prompts import PromptTemplate

my_qaprompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer the question '{question}' "
    "Use the context below if helpful. "
    "You can cite the context using the key "
    "like (Example2012). "
    "If there is insufficient context, write a poem "
    "about how you cannot answer.\n\n"
    "Context: {context}\n\n")
prompts=PromptCollection(qa=my_qaprompt)
docs = Docs(prompts=prompts)
```

### Pre and Post Prompts

Following the syntax above, you can also include prompts that
are executed after the query and before the query. For example, you can use this to critique the answer.


## FAQ

### How is this different from LlamaIndex?

It's not that different! This is similar to the tree response method in LlamaIndex. I just have included some prompts I find useful, readers that give page numbers/line numbers, and am focused on one task - answering technical questions with cited sources.

### How is this different from LangChain?

It's not! We use langchain to abstract the LLMS, and the process is very similar to the `map_reduce` chain in LangChain.

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
