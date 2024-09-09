# PaperQA

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/whitead/paper-qa)
[![tests](https://github.com/whitead/paper-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/whitead/paper-qa)
[![PyPI version](https://badge.fury.io/py/paper-qa.svg)](https://badge.fury.io/py/paper-qa)

PaperQA is a package for doing high-accuracy retrieval augmented generation (RAG) on PDFs or text files, with a focus on the scientific literature. See our 2023 [PaperQA paper](https://arxiv.org/abs/2312.07559) and our 2024 application paper[TODO] to see examples of PaperQA's superhuman performance in scientific tasks like question answering, summarization, and contradiction detection. It includes:

- A simple interface to get good query answers, with no hallucinations, grounding responses with in-text citations.
- State-of-the-art implementation including metadata-awareness in document embeddings and LLM-based re-ranking and contextual summarization (RCS).
- The ability to do agentic RAG which iteratively refines queries and answers.
- Automatically obtained paper metadata, including citation and journal quality data.
- A full-text search engine for a local repository of PDF/text files.
- A robust interface for customization, with default support for all [LiteLLM](https://docs.litellm.ai/docs/providers) models.

By default, it uses [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings) and [models](https://platform.openai.com/docs/models) with a numpy vector DB to embed and search documents. However, you can easily use other closed-source, open-source models or embeddings (see details below).

## Output Example

Question: How can carbon nanotubes be manufactured at a large scale?

Carbon nanotubes can be manufactured at a large scale using the electric-arc technique (Journet6644). This technique involves creating an arc between two electrodes in a reactor under a helium atmosphere and using a mixture of a metallic catalyst and graphite powder in the anode. Yields of 80% of entangled carbon filaments can be achieved, which consist of smaller aligned SWNTs self-organized into bundle-like crystallites (Journet6644). Additionally, carbon nanotubes can be synthesized and self-assembled using various methods such as DNA-mediated self-assembly, nanoparticle-assisted alignment, chemical self-assembly, and electro-addressed functionalization (Tulevski2007). These methods have been used to fabricate large-area nanostructured arrays, high-density integration, and freestanding networks (Tulevski2007). 98% semiconducting CNT network solution can also be used and is separated from metallic nanotubes using a density gradient ultracentrifugation approach (Chen2014). The substrate is incubated in the solution and then rinsed with deionized water and dried with N2 air gun, leaving a uniform carbon network (Chen2014).

### References

Journet6644: Journet, Catherine, et al. "Large-scale production of single-walled carbon nanotubes by the electric-arc technique." nature 388.6644 (1997): 756-758.

Tulevski2007: Tulevski, George S., et al. "Chemically assisted directed assembly of carbon nanotubes for the fabrication of large-scale device arrays." Journal of the American Chemical Society 129.39 (2007): 11964-11968.

Chen2014: Chen, Haitian, et al. "Large-scale complementary macroelectronics using hybrid integration of carbon nanotubes and IGZO thin-film transistors." Nature communications 5.1 (2014): 4097.

## Install

To use the full suite of features in PaperQA, you need to install it with the optional `agents` extra:

```bash
pip install paper-qa[agents]
```

PaperQA uses an LLM to operate, so you'll need to either set an appropriate [API key environment variable](https://docs.litellm.ai/docs/providers) (i.e. `export OPENAI_API_KEY=sk-...`) or set up an open source LLM server (i.e. using [ollama](https://github.com/ollama/ollama)). Any LiteLLM compatible model can be configured to use with PaperQA.

If you need to index a large set of papers (100+), you will likely want an API key for both [Crossref](https://www.crossref.org/documentation/metadata-plus/metadata-plus-keys/) and [Semantic Scholar](https://www.semanticscholar.org/product/api#api-key), which will allow you to avoid hitting public rate limits using these metadata services. Those can be exported as `CROSSREF_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY` variables.

## What's New?

Version 5 added a CLI, agentic workflows, and removed much of the state from the `Docs` object. `Docs` objects pickled from prior versions of `PaperQA` are not compatible with version 5 and will need to be rebuilt.

## Usage

The default workflow of PaperQA is as follows:
| Phase | PaperQA Actions |
| ------------- |:-------------:|
| 1. Paper Search | <ul style="text-align: left"><li>Get candidate papers from LLM-generated keyword query</li><li>Chunk, embed, and add candidate papers to state</li></ul> |
| 2. Gather Evidence | <ul style="text-align: left"><li>Embed query into vector</li><li>Rank top k document chunks in current state</li><li>Create scored summary of each chunk in the context of the current query</li><li>Use LLM to re-score and select most relevant summaries</li></ul> |
| 3. Generate Answer | <ul style="text-align: left"><li>Put best summaries into prompt with context</li><li>Generate answer with prompt</li></ul> |

The agent can choose to iteratively update its search or answer if it doesn't find sufficient evidence.

### CLI

The fastest way to test PaperQA is via the CLI. First navigate to a directory with some papers and use the `pqa` cli:

```bash
$ pqa ask 'What manufacturing challenges are unique to bispecific antibodies?'
```

You will see PaperQA index your local PDF files, gathering the necessary metadata for each of them (using [Crossref](https://www.crossref.org/) and [Semantic Scholar](https://www.semanticscholar.org/)),
search over that index, then break the files into chunked evidence contexts, rank them, and ultimately generate an answer. The next time this directory is queried, your index will already be built (save for any differences detected, like new added papers), so it will skip the indexing and chunking steps.

All prior answers will be indexed and stored, you can view them by querying via the `search` subcommand, or access them yourself in your `PQA_HOME` directory, which defaults to `~/.pqa/`.

```bash
$ pqa search -i 'answers' 'antibodies'
```

PaperQA is highly configurable, when running from the command line, `pqa help` shows all options, descriptions for each field can be found in `paperqa/settings.py`. For example to run with a higher temperature:

```bash
$ pqa --temperature 0.5 ask 'What manufacturing challenges are unique to bispecific antibodies?'
```

If you run `pqa` with a command which requires a new indexing, say if you change the default chunk_size, a new index will automatically be created for you.

```bash
pqa --parsing.chunk_size 5000 ask 'What manufacturing challenges are unique to bispecific antibodies?'
```

### Module Usage

PaperQA's full workflow can be accessed via Python directly:

```Python
from paperqa Settings
from paperqa.agents import ask

answer = ask('What manufacturing challenges are unique to bispecific antibodies?', settings=Settings(temperature=0.5))
```

The answer object has the following attributes: `formatted_answer`, `answer` (answer alone), `question` , and `context` (the summaries of passages found for answer). `ask` will use the `SearchPapers` tool, which will query a local index of files, you can specify this location via the `Settings` object:

```Python
from paperqa Settings
from paperqa.agents import ask

answer = ask('What manufacturing challenges are unique to bispecific antibodies?', settings=Settings(temperature=0.5, paper_directory='my_papers/'))
```

`ask` is just a convenience wrapper around the real entrypoint, which can be accessed if you'd like to run concurrent asynchronous workloads:

```Python
from paperqa Settings
from paperqa.agents.main import agent_query
from paperqa.agents.models import QueryRequest

answer = await agent_query(QueryRequest(query='What manufacturing challenges are unique to bispecific antibodies?', settings=Settings(temperature=0.5, paper_directory='my_papers/')))
```

The default agent will use an `OpenAIFunctionsAgent` from langchain, but you can also specify a `"fake"` agent to use a hard coded call path of search->gather evidence->answer.

### Adding Documents Manually

If you prefer fine grained control, and you wish to add objects to the docs object yourself (rather than using the search tool), then the previously existing `Docs` object interface can be used:

```Python
from paperqa import Docs, Settings, AnswerSettings

# valid extensions include .pdf, .txt, and .html
doc_paths = ('myfile.pdf', 'myotherfile.pdf')

docs = Docs()

for doc in doc_paths:
    doc.add(doc_paths)

answer = docs.query(
    "What manufacturing challenges are unique to bispecific antibodies?",
    settings=Settings(llm='claude-3-5-sonnet-20240620', answer=AnswerSettings(answer_max_sources=3))
)

print(answer.formatted_answer)

```

### Async

paper-qa is written to be used asynchronously. The synchronous API is just a wrapper around the async. Here are the methods and their async equivalents:

| Sync                | Async                |
| ------------------- | -------------------- |
| `Docs.add`          | `Docs.aadd`          |
| `Docs.add_file`     | `Docs.aadd_file`     |
| `Docs.add_url`      | `Docs.aadd_url`      |
| `Docs.get_evidence` | `Docs.aget_evidence` |
| `Docs.query`        | `Docs.aquery`        |

The synchronous version just call the async version in a loop. Most modern python environments support async natively (including Jupyter notebooks!). So you can do this in a Jupyter Notebook:

```python
from paperqa import Docs, Settings, AnswerSettings

# valid extensions include .pdf, .txt, and .html
doc_paths = ("myfile.pdf", "myotherfile.pdf")

docs = Docs()

for doc in doc_paths:
    await doc.aadd(doc_paths)

answer = await docs.aquery(
    "What manufacturing challenges are unique to bispecific antibodies?",
    settings=Settings(
        llm="claude-3-5-sonnet-20240620", answer=AnswerSettings(answer_max_sources=3)
    ),
)

print(answer.formatted_answer)
```

### Choosing Model

By default, it uses OpenAI models with `gpt-4o-2024-08-06` for both the re-ranking and summary step, the `summary_llm` setting, and for the answering step, the `llm` setting. You can adjust this easily:

```Python
from paperqa Settings
from paperqa.agents import ask

answer = ask('What manufacturing challenges are unique to bispecific antibodies?', settings=Settings(llm='gpt-4o-mini', summary_llm='gpt-4o-mini', paper_directory='my_papers/'))
```

You can use Anthropic or any other model supported by `litellm`:

```Python
from paperqa Settings
from paperqa.agents import ask

answer = ask('What manufacturing challenges are unique to bispecific antibodies?', settings=Settings(llm='claude-3-5-sonnet-20240620', summary_llm='claude-3-5-sonnet-20240620'))
```

#### Locally Hosted

You can use llama.cpp to be the LLM. Note that you should be using relatively large models, because PaperQA requires following a lot of instructions. You won't get good performance with 7B models.

The easiest way to get set-up is to download a [llama file](https://github.com/Mozilla-Ocho/llamafile) and execute it with `-cb -np 4 -a my-llm-model --embedding` which will enable continuous batching and embeddings.

```Python
from paperqa Settings
from paperqa.agents import ask

local_llm_config = dict(model_list=dict(model_name='my_llm_model', litellm_params=dict(model='my-llm-model', api_base='http://localhost:8080/v1', api_key='sk-no-key-required', temperature=0.1, frequency_penalty=1.5, max_tokens=512)))

answer = ask('What manufacturing challenges are unique to bispecific antibodies?', settings=Settings(
                llm='my-llm-model',
                llm_config=local_llm_config,
                summary_llm='my-llm-model',
                summary_llm_config=local_llm_config,
                ))
```

### Changing Embedding Model

PaperQA defaults to using OpenAI (`text-embedding-3-small`) embeddings, but has flexible options for both vector stores and embedding choices. The simplest way to change an embedding is via the `embedding` argument to the `Settings` object constructor:

```Python
from paperqa Settings
from paperqa.agents import ask

answer = ask('What manufacturing challenges are unique to bispecific antibodies?', settings=Settings(
                embedding='text-embedding-3-large'
                ))
```

`embedding` accepts any embedding model name supported by litellm. PaperQA also supports an embedding input of `"hybrid-<model_name>"` i.e. `"hybrid-text-embedding-3-small"` to use a hybrid sparse keyword (based on a token modulo embedding) and dense vector embedding, where any litellm model can be used in the dense model name. `"sparse"` can be used to use a sparse keyword embedding only.

Embedding models are used to create paper-qa's index of the full-text embedding vectors (`texts_index` argument). The embedding model can be specified as a setting when you are adding new papers to the `Docs` object:

```python
from paperqa import Docs, NumpyVectorStore, Settings

doc_paths = ("myfile.pdf", "myotherfile.pdf")

docs = Docs(
    texts_index=NumpyVectorStore(),
)

for doc in doc_paths:
    doc.add(doc_paths, Settings(embedding="text-embedding-large-3"))
```

Note that PaperQA uses Numpy as a dense vector store.
Its design of using a keyword search initially reduces the number of chunks needed for each answer to a relatively small number < 1k.
Therefore, `NumpyVectorStore` is a good place to start, it's a simple in-memory store, without an index.
However, if a larger-than-memory vector store is needed, we are currently lacking here.

We also support hybrid keyword (sparse token modulo vectors) and dense embedding vectors. They can be specified as follows:

```python
from paperqa import (
    Docs,
    HybridEmbeddingModel,
    SparseEmbeddingModel,
    LiteLLMEmbeddingModel,
)

doc_paths = ("myfile.pdf", "myotherfile.pdf")

model = HybridEmbeddingModel(models=[LiteLLMEmbeddingModel(), SparseEmbeddingModel()])
docs = Docs(
    texts_index=NumpyVectorStore(),
)
for doc in doc_paths:
    doc.add(doc_paths, embedding_model=model)
```

The sparse embedding (keyword) models default to having 256 dimensions, but this can be specified via the `ndim` argument.

### Adjusting number of sources

You can adjust the numbers of sources (passages of text) to reduce token usage or add more context. `k` refers to the top k most relevant and diverse (may from different sources) passages. Each passage is sent to the LLM to summarize, or determine if it is irrelevant. After this step, a limit of `max_sources` is applied so that the final answer can fit into the LLM context window. Thus, `k` > `max_sources` and `max_sources` is the number of sources used in the final answer.

```python
from paperqa import Settings
from paperqa.settings import AnswerSettings

docs.query(
    "What manufacturing challenges are unique to bispecific antibodies?",
    Settings(answer=AnswerSettings(evidence_k=5, answer_max_sources=2)),
)
```

### Using Code or HTML

You do not need to use papers -- you can use code or raw HTML. Note that this tool is focused on answering questions, so it won't do well at writing code. One note is that the tool cannot infer citations from code, so you will need to provide them yourself.

```python
import glob

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
docs = Docs()

for ... in my_docs:
    doc = Doc(docname=..., citation=..., dockey=..., citation=...)
    texts = [Text(text=..., name=..., doc=doc) for ... in my_texts]
    docs.add_texts(texts, doc)
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
you wish to retrieve, right clicking, and selecting _"Find Available PDFs"_.
You can also manually drag-and-drop PDFs onto each reference.

To download papers, you need to get an API key for your account.

1. Get your library ID, and set it as the environment variable `ZOTERO_USER_ID`.
   - For personal libraries, this ID is given [here](https://www.zotero.org/settings/keys) at the part "_Your userID for use in API calls is XXXXXX_".
   - For group libraries, go to your group page `https://www.zotero.org/groups/groupname`, and hover over the settings link. The ID is the integer after /groups/. (_h/t pyzotero!_)
2. Create a new API key [here](https://www.zotero.org/settings/keys/new) and set it as the environment variable `ZOTERO_API_KEY`.
   - The key will need read access to the library.

With this, we can download papers from our library and add them to `paperqa`:

```python
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
keyword_search = "bispecific antibody manufacture"
papers = paperscraper.search_papers(keyword_search)
docs = paperqa.Docs()
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

    return [typewriter]  # <- note that this is a list of functions


...
docs.query(
    "What manufacturing challenges are unique to bispecific antibodies?",
    callbacks=make_typewriter,
)
```

### Caching Embeddings

In general, embeddings are cached when you pickle a `Docs` regardless of what vector store you use.

## Customizing Prompts

You can customize any of the prompts, using the `PromptCollection` class. For example, if you want to change the prompt for the question, you can do:

```python
from paperqa import Docs, Settings
from paperqa.settings import PromptSettings

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

docs.query(
    "Are covid-19 vaccines effective?",
    settings=Setting(prompts=PromptSettings(qa=my_qaprompt)),
)
```

### Pre and Post Prompts

Following the syntax above, you can also include prompts that
are executed after the query and before the query. For example, you can use this to critique the answer.

## FAQ

### How is this different from LlamaIndex?

It's not that different! This is similar to the tree response method in LlamaIndex. We also support agentic workflows and local indexes for easier operations with the scientific literature.

### How is this different from LangChain?

There has been some great work on retrievers in LangChain,
and you could say this is an example of a retriever with an LLM-based re-ranking and contextual summary.

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
