# Paper QA

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/whitead/paper-qa)
[![tests](https://github.com/whitead/paper-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/whitead/paper-qa)
[![PyPI version](https://badge.fury.io/py/paper-qa.svg)](https://badge.fury.io/py/paper-qa)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

This is a simple and incomplete package for doing question and answering from
PDFs or text files (open an issue for more formats). It uses [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) with a vector DB called [FAISS](https://github.com/facebookresearch/faiss) to embed and search documents. [langchain](https://github.com/hwchase17/langchain) helps
generate answers.

It uses this process

```
embed docs into vectors -> embed query into vector -> search for top k passages in docs

create summary of each passage relevant to query -> put summaries into prompt -> generate answer
```

## What's New (v0.0.5)

- Replaced gpt-index since we were doing some custom metadata
- Now have page numbers directly in references
- You can now load very large PDFs
- Focusing now only on txt and PDFs to get better reading capabilities

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

*This uses a lot of tokens!! About 10-30k tokens per answer + embedding cost (negligible unless many documents used). That is up to $0.50 per answer with current GPT-3 pricing. Use wisely.*

```python

from paperqa import Docs

# get a list of paths, citations

docs = Docs()
for d, c in zip(my_docs, my_citations):
    docs.add(d, c)

# takes ~ 1 min and costs $0.50 to execute this line
answer = docs.query("What manufacturing challenges are unique to bispecific antibodies?")
print(answer.formatted_answer)
```

The answer object has the following attributes: `formatted_answer`, `answer` (answer alone), `question`, `context` (the summaries of passages found for answer), `references` (the docs from which the passages came).

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

## Adjusting number of sources

You can adjust the numbers of sources/passages to reduce token usage or add more context. `k` controls number of passages to search in each source and `max_sources` controls the number of sources included in the context.

```python
docs.query("What manufacturing challenges are unique to bispecific antibodies?", k = 1, max_sources = 3)
```

## FAQ

### How is this different from gpt-index?

gpt-index does generate answers, but in a somewhat opinionated way. It doesn't have a great way to track where text comes from and it's not easy to force it to pull from multiple documents. I don't know which way is better, but for writing scholarly text I found it to work better to pull from multiple relevant documents and then generate an answer. I would like to PR to do this to gpt-index but it looks pretty involved right now.

### Where do the documents come from?

I use some of my own code to pull papers from Google Scholar. This code is not included because it may enable people to violate Google's terms of service and publisher's terms of service.

### Can I save or load?

The `Docs` class can be pickled and unpickled. This is useful if you want to save the embeddings of the documents and then load them later.

```python
import pickle

with open("my_docs.pkl", "wb") as f:
    pickle.dump(docs, f)

with open("my_docs.pkl", "rb") as f:
    docs = pickle.load(f)
```
