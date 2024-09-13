# PaperQA2

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/whitead/paper-qa)
[![tests](https://github.com/whitead/paper-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/whitead/paper-qa)
[![PyPI version](https://badge.fury.io/py/paper-qa.svg)](https://badge.fury.io/py/paper-qa)

PaperQA2 is a package for doing high-accuracy retrieval augmented generation (RAG) on PDFs or text files,
with a focus on the scientific literature.
See our [recent 2024 paper](https://paper.wikicrow.ai) to see examples of PaperQA2's superhuman performance in scientific tasks like
question answering, summarization, and contradiction detection.

## Quickstart

In this example we take a folder of research paper PDFs, magically get their metadata - including citation counts and a retraction check, then parse and cache PDFs into a full-text search index, and finally answer the user question with an LLM agent.

```bash
pip install paper-qa
cd my_papers
pqa ask 'How can carbon nanotubes be manufactured at a large scale?'
```

### Example Output

Question: Has anyone designed neural networks that compute with proteins or DNA?

> The claim that neural networks have been designed to compute with DNA is supported by multiple sources. The work by Qian, Winfree, and Bruck demonstrates the use of DNA strand displacement cascades to construct neural network components, such as artificial neurons and associative memories, using a DNA-based system (Qian2011Neural pages 1-2, Qian2011Neural pages 15-16, Qian2011Neural pages 54-56). This research includes the implementation of a 3-bit XOR gate and a four-neuron Hopfield associative memory, showcasing the potential of DNA for neural network computation. Additionally, the application of deep learning techniques to genomics, which involves computing with DNA sequences, is well-documented. Studies have applied convolutional neural networks (CNNs) to predict genomic features such as transcription factor binding and DNA accessibility (Eraslan2019Deep pages 4-5, Eraslan2019Deep pages 5-6). These models leverage DNA sequences as input data, effectively using neural networks to compute with DNA. While the provided excerpts do not explicitly mention protein-based neural network computation, they do highlight the use of neural networks in tasks related to protein sequences, such as predicting DNA-protein binding (Zeng2016Convolutional pages 1-2). However, the primary focus remains on DNA-based computation.

## What is PaperQA2

PaperQA2 is engineered to be the best RAG model for working with scientific papers. Here are some features:

- A simple interface to get good answers with grounded responses containing in-text citations.
- State-of-the-art implementation including document metadata-awareness
  in embeddings and LLM-based re-ranking and contextual summarization (RCS).
- Support for agentic RAG, where a language agent can iteratively refine queries and answers.
- Automatic redundant fetching of paper metadata,
  including citation and journal quality data from multiple providers.
- A usable full-text search engine for a local repository of PDF/text files.
- A robust interface for customization, with default support for all [LiteLLM][LiteLLM providers] models.

[LiteLLM providers]: https://docs.litellm.ai/docs/providers
[LiteLLM general docs]: https://docs.litellm.ai/docs/

By default, it uses [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings) and [models](https://platform.openai.com/docs/models) with a Numpy vector DB to embed and search documents. However, you can easily use other closed-source, open-source models or embeddings (see details below).

PaperQA2 depends on some awesome libraries/APIs that make our repo possible. Here are some in a random order:

1. [Semantic Scholar](https://www.semanticscholar.org/)
2. [Crossref](https://www.crossref.org/)
3. [Unpaywall](https://unpaywall.org/)
4. [Pydantic](https://docs.pydantic.dev/latest/)
5. [tantivy](https://github.com/quickwit-oss/tantivy)
6. [LiteLLM][LiteLLM general docs]
7. [pybtex](https://pybtex.org/)
8. [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)

## Installation

For a non-development setup,
install PaperQA2 from [PyPI](https://pypi.org/project/paper-qa/):

```bash
pip install paper-qa
```

For development setup,
please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

PaperQA2 uses an LLM to operate,
so you'll need to either set an appropriate [API key environment variable][LiteLLM providers] (i.e. `export OPENAI_API_KEY=sk-...`)
or set up an open source LLM server (i.e. using [llamafile](https://github.com/Mozilla-Ocho/llamafile).
Any LiteLLM compatible model can be configured to use with PaperQA2.

If you need to index a large set of papers (100+),
you will likely want an API key for both [Crossref](https://www.crossref.org/documentation/metadata-plus/metadata-plus-keys/) and [Semantic Scholar](https://www.semanticscholar.org/product/api#api-key),
which will allow you to avoid hitting public rate limits using these metadata services.
Those can be exported as `CROSSREF_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY` variables.

## PaperQA2 vs PaperQA

We've been working on hard on fundamental upgrades for a while and mostly followed [SemVer](https://semver.org/).
meaning we've incremented the major version number on each breaking change.
This brings us to the current major version number v5.
So why call is the repo now called PaperQA2?
We wanted to remark on the fact though that we've exceeded human performance on [many important metrics](https://paper.wikicrow.ai).
So we arbitrarily call version 5 and onward PaperQA2,
and versions before it as PaperQA1 to denote the significant change in performance.
We recognize that we are challenged at naming and counting at FutureHouse,
so we reserve the right at any time to arbitrarily change the name to PaperCrow.

## What's New in Version 5 (aka PaperQA2)?

Version 5 added:

- A CLI `pqa`
- Agentic workflows invoking tools for
  paper search, gathering evidence, and generating an answer
- Removed much of the statefulness from the `Docs` object
- A migration to LiteLLM for compatibility with many LLM providers
  as well as centralized rate limits and cost tracking
- A bundled set of configurations (read [here](#bundled-settings)))
  containing known-good hyperparameters

Note that `Docs` objects pickled from prior versions of `PaperQA` are incompatible with version 5 and will need to be rebuilt.
Also, our minimum Python version is now Python 3.11.

## Usage

To understand PaperQA2, let's start with the pieces of the underlying algorithm. The default workflow of PaperQA2 is as follows:

| Phase                  | PaperQA2 Actions                                                          |
| ---------------------- | ------------------------------------------------------------------------- |
| **1. Paper Search**    | - Get candidate papers from LLM-generated keyword query                   |
|                        | - Chunk, embed, and add candidate papers to state                         |
| **2. Gather Evidence** | - Embed query into vector                                                 |
|                        | - Rank top _k_ document chunks in current state                           |
|                        | - Create scored summary of each chunk in the context of the current query |
|                        | - Use LLM to re-score and select most relevant summaries                  |
| **3. Generate Answer** | - Put best summaries into prompt with context                             |
|                        | - Generate answer with prompt                                             |

The tools can be invoked in any order by a language agent.
For example, an LLM agent might do a narrow and broad search,
or using different phrasing for the gather evidence step from the generate answer step.

### CLI

The fastest way to test PaperQA2 is via the CLI. First navigate to a directory with some papers and use the `pqa` cli:

```bash
$ pqa ask 'What manufacturing challenges are unique to bispecific antibodies?'
```

You will see PaperQA2 index your local PDF files, gathering the necessary metadata for each of them (using [Crossref](https://www.crossref.org/) and [Semantic Scholar](https://www.semanticscholar.org/)),
search over that index, then break the files into chunked evidence contexts, rank them, and ultimately generate an answer. The next time this directory is queried, your index will already be built (save for any differences detected, like new added papers), so it will skip the indexing and chunking steps.

All prior answers will be indexed and stored, you can view them by querying via the `search` subcommand, or access them yourself in your `PQA_HOME` directory, which defaults to `~/.pqa/`.

```bash
$ pqa search -i 'answers' 'antibodies'
```

PaperQA2 is highly configurable, when running from the command line, `pqa --help` shows all options and short descriptions. For example to run with a higher temperature:

```bash
$ pqa --temperature 0.5 ask 'What manufacturing challenges are unique to bispecific antibodies?'
```

You can view all settings with `pqa view`. Another useful thing is to change to other templated settings - for example `fast` is a setting that answers more quickly and you can see it with `pqa -s fast view`

Maybe you have some new settings you want to save? You can do that with

```bash
pqa -s my_new_settings --temperature 0.5 --llm foo-bar-5 save
```

and then you can use it with

```bash
pqa -s my_new_settings ask 'What manufacturing challenges are unique to bispecific antibodies?'
```

If you run `pqa` with a command which requires a new indexing, say if you change the default chunk_size, a new index will automatically be created for you.

```bash
pqa --parsing.chunk_size 5000 ask 'What manufacturing challenges are unique to bispecific antibodies?'
```

You can also use `pqa` to do full-text search with use of LLMs view the search command. For example, let's save the index from a directory and give it a name:

```bash
pqa -i nanomaterials index
```

Now I can search for papers about thermoelectrics:

```bash
pqa -i nanomaterials search thermoelectrics
```

or I can use the normal ask

```bash
pqa -i nanomaterials ask 'Are there nm scale features in thermoelectric materials?'
```

Both the CLI and module have pre-configured settings based on prior performance and our publications, they can be invoked as follows:

```bash
pqa --settings <setting name> ask 'Are there nm scale features in thermoelectric materials?'
```

#### Bundled Settings

Inside [`paperqa/configs`](paperqa/configs) we bundle known useful settings:

| Setting Name | Description                                                                                                                  |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| high_quality | Highly performant, relatively expensive (due to having `evidence_k` = 15) query using a `ToolSelector` agent.                |
| fast         | Setting to get answers cheaply and quickly.                                                                                  |
| wikicrow     | Setting to emulate the Wikipedia article writing used in our WikiCrow publication.                                           |
| contracrow   | Setting to find contradictions in papers, your query should be a claim that needs to be flagged as a contradiction (or not). |
| debug        | Setting useful solely for debugging, but not in any actual application beyond debugging.                                     |

### Module Usage

PaperQA2's full workflow can be accessed via Python directly:

```python
from paperqa import Settings, ask

answer = ask(
    "What manufacturing challenges are unique to bispecific antibodies?",
    settings=Settings(temperature=0.5),
)
```

The answer object has the following attributes: `formatted_answer`, `answer` (answer alone), `question` , and `context` (the summaries of passages found for answer). `ask` will use the `SearchPapers` tool, which will query a local index of files, you can specify this location via the `Settings` object:

```python
from paperqa import Settings, ask

answer = ask(
    "What manufacturing challenges are unique to bispecific antibodies?",
    settings=Settings(temperature=0.5, paper_directory="my_papers"),
)
```

`ask` is just a convenience wrapper around the real entrypoint, which can be accessed if you'd like to run concurrent asynchronous workloads:

```python
from paperqa import Settings, agent_query, QueryRequest

answer = await agent_query(
    QueryRequest(
        query="What manufacturing challenges are unique to bispecific antibodies?",
        settings=Settings(temperature=0.5, paper_directory="my_papers"),
    )
)
```

The default agent will use an LLM based agent,
but you can also specify a `"fake"` agent to use a hard coded call path of search -> gather evidence -> answer to reduce token usage.

### Adding Documents Manually

If you prefer fine grained control, and you wish to add objects to the docs object yourself (rather than using the search tool), then the previously existing `Docs` object interface can be used:

```python
from paperqa import Docs, Settings

# valid extensions include .pdf, .txt, and .html
doc_paths = ("myfile.pdf", "myotherfile.pdf")

docs = Docs()

for doc in doc_paths:
    docs.add(doc)

settings = Settings()
settings.llm = "claude-3-5-sonnet-20240620"
settings.answer.answer_max_sources = 3

answer = docs.query(
    "What manufacturing challenges are unique to bispecific antibodies?",
    settings=settings,
)

print(answer.formatted_answer)
```

### Async

PaperQA2 is written to be used asynchronously. The synchronous API is just a wrapper around the async. Here are the methods and their async equivalents:

| Sync                | Async                |
| ------------------- | -------------------- |
| `Docs.add`          | `Docs.aadd`          |
| `Docs.add_file`     | `Docs.aadd_file`     |
| `Docs.add_url`      | `Docs.aadd_url`      |
| `Docs.get_evidence` | `Docs.aget_evidence` |
| `Docs.query`        | `Docs.aquery`        |

The synchronous version just calls the async version in a loop. Most modern python environments support async natively (including Jupyter notebooks!). So you can do this in a Jupyter Notebook:

```python
import asyncio
from paperqa import Docs


async def main():
    # valid extensions include .pdf, .txt, and .html
    doc_paths = ("myfile.pdf", "myotherfile.pdf")

    docs = Docs()

    for doc in doc_paths:
        await docs.aadd(doc)

    answer = await docs.aquery(
        "What manufacturing challenges are unique to bispecific antibodies?"
    )

    print(answer.formatted_answer)


asyncio.run(main())
```

### Choosing Model

By default, it uses OpenAI models with `gpt-4o-2024-08-06` for both the re-ranking and summary step, the `summary_llm` setting, and for the answering step, the `llm` setting. You can adjust this easily:

```python
from paperqa import Settings, ask

answer = ask(
    "What manufacturing challenges are unique to bispecific antibodies?",
    settings=Settings(
        llm="gpt-4o-mini", summary_llm="gpt-4o-mini", paper_directory="my_papers"
    ),
)
```

You can use Anthropic or any other model supported by `litellm`:

```python
from paperqa import Settings, ask

answer = ask(
    "What manufacturing challenges are unique to bispecific antibodies?",
    settings=Settings(
        llm="claude-3-5-sonnet-20240620", summary_llm="claude-3-5-sonnet-20240620"
    ),
)
```

#### Locally Hosted

You can use llama.cpp to be the LLM. Note that you should be using relatively large models, because PaperQA2 requires following a lot of instructions. You won't get good performance with 7B models.

The easiest way to get set-up is to download a [llama file](https://github.com/Mozilla-Ocho/llamafile) and execute it with `-cb -np 4 -a my-llm-model --embedding` which will enable continuous batching and embeddings.

```python
from paperqa import Settings, ask

local_llm_config = dict(
    model_list=dict(
        model_name="my_llm_model",
        litellm_params=dict(
            model="my-llm-model",
            api_base="http://localhost:8080/v1",
            api_key="sk-no-key-required",
            temperature=0.1,
            frequency_penalty=1.5,
            max_tokens=512,
        ),
    )
)

answer = ask(
    "What manufacturing challenges are unique to bispecific antibodies?",
    settings=Settings(
        llm="my-llm-model",
        llm_config=local_llm_config,
        summary_llm="my-llm-model",
        summary_llm_config=local_llm_config,
    ),
)
```

### Changing Embedding Model

PaperQA2 defaults to using OpenAI (`text-embedding-3-small`) embeddings, but has flexible options for both vector stores and embedding choices. The simplest way to change an embedding is via the `embedding` argument to the `Settings` object constructor:

```python
from paperqa import Settings, ask

answer = ask(
    "What manufacturing challenges are unique to bispecific antibodies?",
    settings=Settings(embedding="text-embedding-3-large"),
)
```

`embedding` accepts any embedding model name supported by litellm. PaperQA2 also supports an embedding input of `"hybrid-<model_name>"` i.e. `"hybrid-text-embedding-3-small"` to use a hybrid sparse keyword (based on a token modulo embedding) and dense vector embedding, where any litellm model can be used in the dense model name. `"sparse"` can be used to use a sparse keyword embedding only.

Embedding models are used to create PaperQA2's index of the full-text embedding vectors (`texts_index` argument). The embedding model can be specified as a setting when you are adding new papers to the `Docs` object:

```python
from paperqa import Docs, Settings

doc_paths = ("myfile.pdf", "myotherfile.pdf")

docs = Docs()

for doc in doc_paths:
    doc.add(doc_paths, Settings(embedding="text-embedding-large-3"))
```

Note that PaperQA2 uses Numpy as a dense vector store.
Its design of using a keyword search initially reduces the number of chunks needed for each answer to a relatively small number < 1k.
Therefore, `NumpyVectorStore` is a good place to start, it's a simple in-memory store, without an index.
However, if a larger-than-memory vector store is needed, we are currently lacking here.

The hybrid embeddings can be customized:

```python
from paperqa import (
    Docs,
    HybridEmbeddingModel,
    SparseEmbeddingModel,
    LiteLLMEmbeddingModel,
)


doc_paths = ("myfile.pdf", "myotherfile.pdf")

model = HybridEmbeddingModel(
    models=[LiteLLMEmbeddingModel(), SparseEmbeddingModel(ndim=1024)]
)
docs = Docs()
for doc in doc_paths:
    doc.add(doc_paths, embedding_model=model)
```

The sparse embedding (keyword) models default to having 256 dimensions, but this can be specified via the `ndim` argument.

### Adjusting number of sources

You can adjust the numbers of sources (passages of text) to reduce token usage or add more context. `k` refers to the top k most relevant and diverse (may from different sources) passages. Each passage is sent to the LLM to summarize, or determine if it is irrelevant. After this step, a limit of `max_sources` is applied so that the final answer can fit into the LLM context window. Thus, `k` > `max_sources` and `max_sources` is the number of sources used in the final answer.

```python
from paperqa import Settings

settings = Settings()
settings.answer.answer_max_sources = 3
settings.answer.k = 5

docs.query(
    "What manufacturing challenges are unique to bispecific antibodies?",
    settings=settings,
)
```

### Using Code or HTML

You do not need to use papers -- you can use code or raw HTML. Note that this tool is focused on answering questions, so it won't do well at writing code. One note is that the tool cannot infer citations from code, so you will need to provide them yourself.

```python
import glob
import os
from paperqa import Docs

source_files = glob.glob("**/*.js")

docs = Docs()
for f in source_files:
    # this assumes the file names are unique in code
    docs.add(f, citation="File " + os.path.name(f), docname=os.path.name(f))
answer = docs.query("Where is the search bar in the header defined?")
print(answer)
```

### Using External DB/Vector DB and Caching

You may want to cache parsed texts and embeddings in an external database or file. You can then build a Docs object from those directly:

```python
from paperqa import Docs, Doc, Text

docs = Docs()

for ... in my_docs:
    doc = Doc(docname=..., citation=..., dockey=..., citation=...)
    texts = [Text(text=..., name=..., doc=doc) for ... in my_texts]
    docs.add_texts(texts, doc)
```

## Where do I get papers?

Well that's a really good question! It's probably best to just download PDFs of papers you think will help answer your question and start from there.

### Zotero

_It's been a while since we've tested this - so let us know if it runs into issues!_

If you use [Zotero](https://www.zotero.org/) to organize your personal bibliography,
you can use the `paperqa.contrib.ZoteroDB` to query papers from your library,
which relies on [pyzotero](https://github.com/urschrei/pyzotero).

Install `pyzotero` via the `zotero` extra for this feature:

```bash
pip install paperqa[zotero]
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

### Paper Scraper

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
answer = docs.query(
    "What manufacturing challenges are unique to bispecific antibodies?"
)
print(answer)
```

## Callbacks

To execute a function on each chunk of LLM completions, you need to provide a function that can be executed on each chunk. For example, to get a typewriter view of the completions, you can do:

```python
def typewriter(chunk: str) -> None:
    print(chunk, end="")


docs = Docs()

# add some docs...

docs.query(
    "What manufacturing challenges are unique to bispecific antibodies?",
    callbacks=[typewriter],
)
```

### Caching Embeddings

In general, embeddings are cached when you pickle a `Docs` regardless of what vector store you use. So as long as you save your underlying `Docs` object, you should be able to avoid re-embedding your documents.

## Customizing Prompts

You can customize any of the prompts using settings.

```python
from paperqa import Docs, Settings

my_qa_prompt = (
    "Answer the question '{question}' "
    "Use the context below if helpful. "
    "You can cite the context using the key "
    "like (Example2012). "
    "If there is insufficient context, write a poem "
    "about how you cannot answer.\n\n"
    "Context: {context}\n\n"
)

docs = Docs()
settings = Settings()
settings.prompts.qa = my_qa_prompt
docs.query(
    "Are covid-19 vaccines effective?",
    settings=settings,
)
```

### Pre and Post Prompts

Following the syntax above, you can also include prompts that
are executed after the query and before the query. For example, you can use this to critique the answer.

## FAQ

### How come I get different results than your papers?

Internally at FutureHouse, we have a slightly different set of tools. We're trying to get some of them, like citation traversal, into this repo. However, we have APIs and licenses to access research papers that we cannot share openly. Similarly, in our research papers' results we do not start with the known relevant PDFs. Our agent has to identify them using keyword search over all papers, rather than just a subset. We're gradually aligning these two versions of PaperQA, but until there is an open-source way to freely access papers (even just open source papers) you will need to provide PDFs yourself.

### How is this different from LlamaIndex?

It's not that different! This is similar to the tree response method in LlamaIndex. We also support agentic workflows and local indexes for easier operations with the scientific literature. Another big difference is our strong focus on scientific papers and their underlying metadata.

### How is this different from LangChain?

There has been some great work on retrievers in LangChain, and you could say this is an example of a retriever with an LLM-based re-ranking and contextual summary. Another big difference is our strong focus on scientific papers and their underlying metadata.

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

## Citation

Please read and cite the following papers if you use this software:

```bibtex
@article{skarlinski2024language,
  title={Language agents achieve superhuman synthesis of scientific knowledge},
  author={
    Michael D. Skarlinski and
    Sam Cox and
    Jon M. Laurent and
    James D. Braza and
    Michaela Hinks and
    Michael J. Hammerling and
    Manvitha Ponnapati and
    Samuel G. Rodriques and
    Andrew D. White},
  year={2024},
  journal={preprint},
  url={https://paper.wikicrow.ai}
}


@article{lala2023paperqa,
  title={PaperQA: Retrieval-Augmented Generative Agent for Scientific Research},
  author={L{\'a}la, Jakub and O'Donoghue, Odhran and Shtedritski, Aleksandar and Cox, Sam and Rodriques, Samuel G and White, Andrew D},
  journal={arXiv preprint arXiv:2312.07559},
  year={2023}
}
```
