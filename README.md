# Paper QA

This is a simple and incomplete package for doing question and answering from
documents. It uses [gpt-index](https://github.com/jerryjliu/gpt_index) to
embed and search documents and [langchain](https://github.com/hwchase17/langchain) to
generate answers.

It uses this process

```
embed docs into vectors -> embed query into vector -> search for top k passages in docs

create summary of each doc relevant to query -> put summary into prompt -> generate answer
```

## Install

Install from github with pip:

```bash
pip install git+https://github.com/whitead/paper-qa.git
```

## Usage

Make sure you have set your OPENAI_API_KEY environment variable to your [openai api key](https://beta.openai.com/docs/developer-quickstart/your-api-keys)

```python

from paperqa import Docs

docs = Docs()
for d in my_docs:
    docs.add(d, citation)

# takes ~ 1 min
answer = docs.query("What manufacturing challenges are unique to bispecific antibodies?")
print(answer.formatted_answer)
```

The answer object has the following attributes: `formatted_answer`, `answer` (answer alone), `questions`, `context` (the summaries of passages found for answer), `refernces` (the docs from which the passages came).

## How is this different from gpt-index?

gpt-index does generate answers, but in a somewhat opinionated way. It doesn't have a great way to track where text comes from and it's not easy to force it to pull from multiple documents. I don't know which way is better, but for writing scholarly text I found it to work better to pull from multiple relevant documents and then generate an answer.

## Where do the documents come from?

I use some of my own code to pull papers from Google Scholar. This code is not included because it may enable people to violate Google's terms of service and publisher's terms of service.
