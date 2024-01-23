# PaperQA

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/whitead/paper-qa)
[![tests](https://github.com/whitead/paper-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/whitead/paper-qa)
[![PyPI version](https://badge.fury.io/py/paper-qa.svg)](https://badge.fury.io/py/paper-qa)

## YOU ARE LOOKING AT PRE-RELEASE README

**This is the README for an upcoming v4 release**

You can see the current stable version [here](https://github.com/whitead/paper-qa/tree/84f13ea32c22b85924cd681a4b5f4fbd174afd71)

This is a minimal package for doing question and answering from
PDFs or text files (which can be raw HTML). It strives to give very good answers, with no hallucinations, by grounding responses with in-text citations.

By default, it uses [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) with a simple numpy vector DB to embed and search documents. However, via [langchain](https://github.com/hwchase17/langchain) you can use open-source models or embeddings (see details below).

paper-qa uses the process shown below:

1. embed docs into vectors
2. embed query into vector
3. search for top k passages in docs
4. create summary of each passage relevant to query
5. score and select only relevant summaries
6. put summaries into prompt
7. generate answer with prompt

See our paper for more details:

```bibtex
@article{lala2023paperqa,
  title={PaperQA: Retrieval-Augmented Generative Agent for Scientific Research},
  author={L{\'a}la, Jakub and O'Donoghue, Odhran and Shtedritski, Aleksandar and Cox, Sam and Rodriques, Samuel G and White, Andrew D},
  journal={arXiv preprint arXiv:2312.07559},
  year={2023}
}
```

## Output Example

Question: How can carbon nanotubes be manufactured at a large scale?

Carbon nanotubes can be manufactured at a large scale using the electric-arc technique (Journet6644). This technique involves creating an arc between two electrodes in a reactor under a helium atmosphere and using a mixture of a metallic catalyst and graphite powder in the anode. Yields of 80% of entangled carbon filaments can be achieved, which consist of smaller aligned SWNTs self-organized into bundle-like crystallites (Journet6644). Additionally, carbon nanotubes can be synthesized and self-assembled using various methods such as DNA-mediated self-assembly, nanoparticle-assisted alignment, chemical self-assembly, and electro-addressed functionalization (Tulevski2007). These methods have been used to fabricate large-area nanostructured arrays, high-density integration, and freestanding networks (Tulevski2007). 98% semiconducting CNT network solution can also be used and is separated from metallic nanotubes using a density gradient ultracentrifugation approach (Chen2014). The substrate is incubated in the solution and then rinsed with deionized water and dried with N2 air gun, leaving a uniform carbon network (Chen2014).

### References

Journet6644: Journet, Catherine, et al. "Large-scale production of single-walled carbon nanotubes by the electric-arc technique." nature 388.6644 (1997): 756-758.

Tulevski2007: Tulevski, George S., et al. "Chemically assisted directed assembly of carbon nanotubes for the fabrication of large-scale device arrays." Journal of the American Chemical Society 129.39 (2007): 11964-11968.

Chen2014: Chen, Haitian, et al. "Large-scale complementary macroelectronics using hybrid integration of carbon nanotubes and IGZO thin-film transistors." Nature communications 5.1 (2014): 4097.


## What's New?

Version 4 removed langchain from the package because it no longer supports pickling. This also simplifies the package a bit - especially prompts. Langchain can still be used, but it's not required. You can use any LLMs from langchain, but you will need to use the `LangchainLLMModel` class to wrap the model.

## Install

Install with pip:

```bash
pip install paper-qa
```

You need to have an LLM to use paper-qa. You can use OpenAI, llama.cpp (via Server), or any LLMs from langchain. OpenAI just works, as long as you have set your OpenAI API key (`export OPENAI_API_KEY=sk-...`). See instructions below for other LLMs.

## Usage

To use paper-qa, you need to have a list of paths/files/urls (valid extensions include: .pdf, .txt). You can then use the `Docs` class to add the documents and then query them. `Docs` will try to guess citation formats from the content of the files, but you can also provide them yourself.

```python

from paperqa import Docs

my_docs = ...# get a list of paths

docs = Docs()
for d in my_docs:
    docs.add(d)

answer = docs.query("What manufacturing challenges are unique to bispecific antibodies?")
print(answer.formatted_answer)
```

The answer object has the following attributes: `formatted_answer`, `answer` (answer alone), `question` , and `context` (the summaries of passages found for answer).

### Async

paper-qa is written to be used asynchronously. The synchronous API is just a wrapper around the async. Here are the methods and their async equivalents:

| Sync | Async |
| --- | --- |
| `Docs.add` | `Docs.aadd` |
| `Docs.add_file` | `Docs.aadd_file` |
| `Docs.add_url` | `Docs.add_url` |
| `Docs.get_evidence` | `Docs.aget_evidence` |
| `Docs.query` | `Docs.aquery` |

The synchronous version just call the async version in a loop. Most modern python environments support async natively (including Jupyter notebooks!). So you can do this in a Jupyter Notebook:

```py
from paperqa import Docs

my_docs = ...# get a list of paths

docs = Docs()
for d in my_docs:
    await docs.aadd(d)

answer = await docs.aquery("What manufacturing challenges are unique to bispecific antibodies?")
```

### Adding Documents

`add` will add from paths. You can also use `add_file` (expects a file object) or `add_url` to work with other sources.

### Choosing Model

By default, it uses a hybrid of `gpt-3.5-turbo` and `gpt-4-turbo`. You can adjust this:

```py
docs = Docs(llm='gpt-3.5-turbo')
```

or you can use any other model available in [langchain](https://github.com/hwchase17/langchain):

```py
from paperqa import Docs
from langchain_community.chat_models import ChatAnthropic
docs = Docs(llm="langchain",
            client=ChatAnthropic())
```

Note we split the model into the wrapper and `client`, which is `ChatAnthropic` here. This is because `client` stores the non-pickleable part and langchain LLMs are only sometimes serializable/pickleable. The paper-qa `Docs` must always serializable. Thus, we split the model into two parts.

```py
import pickle
docs = Docs(llm="langchain",
            client=ChatAnthropic())
model_str = pickle.dumps(docs)
docs = pickle.loads(model_str)
# but you have to set the client after loading
docs.set_client(ChatAnthropic())
```

#### Locally Hosted

You can use llama.cpp to be the LLM. Note that you should be using relatively large models, because paper-qa requires following a lot of instructions. You won't get good performance with 7B models.

The easiest way to get set-up is to download a [llama file](https://github.com/Mozilla-Ocho/llamafile) and execute it with `-cb -np 4 -a my-llm-model --embedding` which will enable continuous batching and embeddings.

```py
from paperqa import Docs, LlamaEmbeddingModel
from openai import AsyncOpenAI

# start llamap.cpp client with

local_client = AsyncOpenAI(
    base_url="http://localhost:8080/v1",
    api_key = "sk-no-key-required"
)

docs = Docs(client=local_client,
            embedding=LlamaEmbeddingModel(),
            llm_model=OpenAILLMModel(config=dict(model="my-llm-model", temperature=0.1, frequency_penalty=1.5, max_tokens=512)))
```

### Changing Embedding Model

You can use langchain embedding models, or the [SentenceTransformer](https://www.sbert.net/) models. For example

```py
from paperqa import Docs, SentenceTransformerEmbeddingModel
from openai import AsyncOpenAI

# start llamap.cpp client with

local_client = AsyncOpenAI(
    base_url="http://localhost:8080/v1",
    api_key = "sk-no-key-required"
)

docs = Docs(client=local_client,
            embedding=SentenceTransformerEmbeddingModel(),
            llm_model=OpenAILLMModel(config=dict(model="my-llm-model", temperature=0.1, frequency_penalty=1.5, max_tokens=512)))
```

Just like in the above examples, we have to split the Langchain model into a client and model to keep `Docs` serializable.
```py

from paperqa import Docs, LangchainEmbeddingModel

docs = Docs(embedding_model=LangchainEmbeddingModel(), embedding_client=OpenAIEmbeddings())
```

### Adjusting number of sources

You can adjust the numbers of sources (passages of text) to reduce token usage or add more context. `k` refers to the top k most relevant and diverse (may from different sources) passages. Each passage is sent to the LLM to summarize, or determine if it is irrelevant. After this step, a limit of `max_sources` is applied so that the final answer can fit into the LLM context window. Thus, `k` > `max_sources`  and `max_sources` is the number of sources used in the final answer.

```py
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

### Using External DB/Vector DB and Caching

You may want to cache parsed texts and embeddings in an external database or file. You can then build a Docs object from those directly:

```py

docs = Docs()

for ... in my_docs:
    doc = Doc(docname=...,  citation=..., dockey=..., citation=...)
    texts = [Text(text=..., name=..., doc=doc) for ... in my_texts]
    docs.add_texts(texts, doc)
```

If you want to use an external vector store, you can also do that directly via langchain. For example, to use the [FAISS](https://ai.meta.com/tools/faiss/) vector store from langchain:

```py
from paperqa import LangchainVectorStore, Docs
from langchain_community.vector_store import FAISS
from langchain_openai import OpenAIEmbeddings

my_index = LangchainVectorStore(cls=FAISS, embedding_model=OpenAIEmbeddings())
docs = Docs(texts_index=my_index)

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

## Callbacks Factory

To execute a function on each chunk of LLM completions, you need to provide a function that when called with the name of the step produces a list of functions to execute on each chunk. For example, to get a typewriter view of the completions, you can do:

```python
def make_typewriter(step_name):
    def typewriter(chunk):
        print(chunk, end="")
    return [typewriter] # <- note that this is a list of functions
...
docs.query("What manufacturing challenges are unique to bispecific antibodies?", get_callbacks=make_typewriter)
```

### Caching Embeddings

In general, embeddings are cached when you pickle a `Docs` regardless of what vector store you use. See above for details on more explicit management of them.

## Customizing Prompts

You can customize any of the prompts, using the `PromptCollection` class. For example, if you want to change the prompt for the question, you can do:

```python
from paperqa import Docs, Answer, PromptCollection

my_qaprompt = "Answer the question '{question}' "
    "Use the context below if helpful. "
    "You can cite the context using the key "
    "like (Example2012). "
    "If there is insufficient context, write a poem "
    "about how you cannot answer.\n\n"
    "Context: {context}\n\n"
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

There has been some great work on retrievers in langchain and you could say this is an example of a retreiver.

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

docs.set_client() #defaults to OpenAI
```
