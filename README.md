# Paper QA


[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/whitead/paper-qa)
[![tests](https://github.com/whitead/paper-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/whitead/paper-qa)
[![PyPI version](https://badge.fury.io/py/paper-qa.svg)](https://badge.fury.io/py/paper-qa)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

This is a simple and incomplete package for doing question and answering from
documents. It uses [gpt-index](https://github.com/jerryjliu/gpt_index) to
embed and search documents and [langchain](https://github.com/hwchase17/langchain) to
generate answers.

It uses this process

```
embed docs into vectors -> embed query into vector -> search for top k passages in docs

create summary of each passage relevant to query -> put summaries into prompt -> generate answer
```

## Install

Install from github with pip:

```bash
pip install git+https://github.com/whitead/paper-qa.git
```

## Usage

Make sure you have set your OPENAI_API_KEY environment variable to your [openai api key](https://beta.openai.com/docs/developer-quickstart/your-api-keys)

To use the package, you need to have a list of paths (valid extensions include: .pdf, .txt, .jpg, .pptx, .docx, .csv, .epub, .md, .mp4, .mp3) and a list of citations (strings) that correspond to the paths. You can then use the `Docs` class to add the documents and then query them.

```python

from paperqa import Docs

# get a list of paths, citations

docs = Docs()
for d, c in zip(my_docs, my_citations):
    docs.add(d, c)

# takes ~ 1 min
answer = docs.query("What manufacturing challenges are unique to bispecific antibodies?")
print(answer.formatted_answer)
```

The answer object has the following attributes: `formatted_answer`, `answer` (answer alone), `questions`, `context` (the summaries of passages found for answer), `refernces` (the docs from which the passages came).

### How is this different from gpt-index?

gpt-index does generate answers, but in a somewhat opinionated way. It doesn't have a great way to track where text comes from and it's not easy to force it to pull from multiple documents. I don't know which way is better, but for writing scholarly text I found it to work better to pull from multiple relevant documents and then generate an answer. I would like to PR to do this to gpt-index but it looks pretty involved right now.

### Where do the documents come from?

I use some of my own code to pull papers from Google Scholar. This code is not included because it may enable people to violate Google's terms of service and publisher's terms of service.

### Saving/loading

The `Docs` class can be pickled and unpickled. This is useful if you want to save the embeddings of the documents and then load them later.

```python
import pickle

with open("my_docs.pkl", "wb") as f:
    pickle.dump(docs, f)

with open("my_docs.pkl", "rb") as f:
    docs = pickle.load(f)
```
