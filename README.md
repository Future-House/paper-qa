# Paper QA

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/whitead/paper-qa)
[![tests](https://github.com/whitead/paper-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/whitead/paper-qa)
[![PyPI version](https://badge.fury.io/py/paper-qa.svg)](https://badge.fury.io/py/paper-qa)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

This is a minimal package for doing question and answering from
PDFs or text files (which can be raw HTML). It strives to give very good answers, with no hallucinations, by grounding responses with in-text citations. It uses [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) with a vector DB called [FAISS](https://github.com/facebookresearch/faiss) to embed and search documents. [langchain](https://github.com/hwchase17/langchain) helps
generate answers.

It uses this process

```text
embed docs into vectors -> embed query into vector -> search for top k passages in docs

create summary of each passage relevant to query -> put summaries into prompt -> generate answer
```

## Hugging Face Demo

[Hugging Face Demo](https://huggingface.co/spaces/whitead/paper-qa)


![image](https://user-images.githubusercontent.com/908389/218957863-4aa2fa2c-14cf-4b0d-82fd-bf837f5f550b.png)

## Example

Question: How can carbon nanotubes be manufactured at a large scale?

Carbon nanotubes can be manufactured at a large scale using the electric-arc technique (Journet6644). This technique involves creating an arc between two electrodes in a reactor under a helium atmosphere and using a mixture of a metallic catalyst and graphite powder in the anode. Yields of 80% of entangled carbon filaments can be achieved, which consist of smaller aligned SWNTs self-organized into bundle-like crystallites (Journet6644). Additionally, carbon nanotubes can be synthesized and self-assembled using various methods such as DNA-mediated self-assembly, nanoparticle-assisted alignment, chemical self-assembly, and electro-addressed functionalization (Tulevski2007). These methods have been used to fabricate large-area nanostructured arrays, high-density integration, and freestanding networks (Tulevski2007). 98% semiconducting CNT network solution can also be used and is separated from metallic nanotubes using a density gradient ultracentrifugation approach (Chen2014). The substrate is incubated in the solution and then rinsed with deionized water and dried with N2 air gun, leaving a uniform carbon network (Chen2014).

### References

Journet6644: Journet, Catherine, et al. "Large-scale production of single-walled carbon nanotubes by the electric-arc technique." nature 388.6644 (1997): 756-758.

Tulevski2007: Tulevski, George S., et al. "Chemically assisted directed assembly of carbon nanotubes for the fabrication of large-scale device arrays." Journal of the American Chemical Society 129.39 (2007): 11964-11968.

Chen2014: Chen, Haitian, et al. "Large-scale complementary macroelectronics using hybrid integration of carbon nanotubes and IGZO thin-film transistors." Nature communications 5.1 (2014): 4097.

## Install

Install with pip:

```bash
pip install paper-qa
```

## Usage

Make sure you have set your OPENAI_API_KEY environment variable to your [openai api key](https://platform.openai.com/account/api-keys)

To use paper-qa, you need to have a list of paths (valid extensions include: .pdf, .txt) and a list of citations (strings) that correspond to the paths. You can then use the `Docs` class to add the documents and then query them.

*This uses a lot of tokens!! About 5-10k tokens per answer + embedding cost (negligible unless many documents used). That is up to $0.02 per answer with current GPT-3 pricing. Use wisely.*

```python

from paperqa import Docs

# get a list of paths

docs = Docs()
for d in my_docs:
    docs.add(d)

# takes ~ 1 min and costs $0.10-$0.20 to execute this line
answer = docs.query("What manufacturing challenges are unique to bispecific antibodies?")
print(answer.formatted_answer)
```

The answer object has the following attributes: `formatted_answer`, `answer` (answer alone), `question`, `context` (the summaries of passages found for answer), `references` (the docs from which the passages came), and `passages` which contain the raw text of the passages as a dictionary.

## Adjusting number of sources

You can adjust the numbers of sources (passages of text) to reduce token usage or add more context. `k` refers to the top k most relevant and diverse (may from different sources) passages. Each passage is sent to the LLM to summarize, or determine if it is irrelevant. After this step, a limit of `max_sources` is applied so that the final answer can fit into the LLM context window. Thus, `k` > `max_sources`  and `max_sources` is the number of sources used in the final answer.

```python
docs.query("What manufacturing challenges are unique to bispecific antibodies?", k = 5, max_sources = 2)
```

## Where do I get papers?

Well that's a really good question! It's probably best to just download PDFs of papers you think will help answer your question and start from there.

If you want to do it automatically, I've found an unrelated project called [paper-scraper](https://github.com/blackadad/paper-scraper) that looks
like it might help. But beware, this project looks like it uses some scraping tools that may violate publisher's rights or be in a gray area of legality.

```py
keyword_search = 'bispecific antibody manufacture'
papers = paperscraper.search_papers(keyword_search)
docs = paperqa.Docs()
for path,data in papers.items():
    try:
        docs.add(path, data['citation'], data['key'])
    except ValueError as e:
        # sometimes this happens if PDFs aren't downloaded or readable
        print('Could not read', path, e)
# takes ~ 1 min and costs $0.50 to execute this line
answer = docs.query("What manufacturing challenges are unique to bispecific antibodies?")
print(answer.formatted_answer)
```

## FAQ

### How is this different from gpt-index?

It's not that different! This is similar to the tree response method in GPT-index. I just have included some prompts I find useful, readers that give page numbers/line numbers, and am focused on one tasks - answering technical questions with cited sources.

### How is this different from LangChain?

It's not! We use langchain to abstract the LLMS, and the process is very similar to the `map_reduce` chain in LangChain.

### Caching

This code will cache responses from LLMS by default in `$HOME/.paperqa/llm_cache.db`. Delete this file to clear the cache.

### Can I use different LLMs?

Yes, you can use any LLMs from [langchain](langchain.readthedocs.io/) by passing the `llm` argument to the `Docs` class. You can use different LLMs for summarization and for question answering too.

### Where do the documents come from?

You can provide your own. I use some of my own code to pull papers from Google Scholar. This code is not included because it may enable people to violate Google's terms of service and publisher's terms of service.

### Can I save or load?

The `Docs` class can be pickled and unpickled. This is useful if you want to save the embeddings of the documents and then load them later. The database is stored in `$HOME/.paperqa/{name}` where `name` is `default`, or you can pass a `name` when you instantiate the `paperqa` doc object.

```python
import pickle

with open("my_docs.pkl", "wb") as f:
    pickle.dump(docs, f)

with open("my_docs.pkl", "rb") as f:
    docs = pickle.load(f)
```
