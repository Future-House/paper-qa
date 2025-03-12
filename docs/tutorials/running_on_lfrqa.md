---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

# Measuring PaperQA2 with LFRQA
> This tutorial is available as a Jupyter notebook [here](https://github.com/Future-House/paper-qa/blob/main/docs/tutorials/running_on_lfrqa.md)


## Overview

The **LFRQA dataset** was introduced in the paper [_RAG-QA Arena: Evaluating Domain Robustness for Long-Form Retrieval-Augmented Question Answering_](https://arxiv.org/pdf/2407.13998). It features **1,404 science questions** (along with other categories) that have been human-annotated with answers. This tutorial walks through the process of setting up the dataset for use and benchmarking.

## Download the Annotations

First, we need to obtain the annotated dataset from the official repository:


```python
# Create a new directory for the dataset
!mkdir -p data/rag-qa-benchmarking

# Get the annotated questions
!curl https://raw.githubusercontent.com/awslabs/rag-qa-arena/refs/heads/main/data/\
annotations_science_with_citation.jsonl \
-o data/rag-qa-benchmarking/annotations_science_with_citation.jsonl
```

<!-- #region -->


## Download the Robust-QA Documents

LFRQA is built upon **Robust-QA**, so we must download the relevant documents:

<!-- #endregion -->

```python
# Download the Lotte dataset, which includes the required documents
!curl https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz --output lotte.tar.gz

# Extract the dataset
!tar -xvzf lotte.tar.gz

# Move the science test collection to our dataset folder
!cp lotte/science/test/collection.tsv ./data/rag-qa-benchmarking/science_test_collection.tsv

# Clean up unnecessary files
!rm lotte.tar.gz
!rm -rf lotte
```

For more details, refer to the original paper: [_RAG-QA Arena: Evaluating Domain Robustness for Long-Form Retrieval-Augmented Question Answering_](https://arxiv.org/pdf/2407.13998).

<!-- #region -->


## Load the Data

We now load the documents into a pandas dataframe:

<!-- #endregion -->

```python
import os

import pandas as pd

# Load questions and answers dataset
rag_qa_benchmarking_dir = os.path.join("data", "rag-qa-benchmarking")

# Load documents dataset
lfrqa_docs_df = pd.read_csv(
    os.path.join(rag_qa_benchmarking_dir, "science_test_collection.tsv"),
    sep="\t",
    names=["doc_id", "doc_text"],
)
```

## Select the Documents to Use
RobustQA consists on 1.7M documents. Hence, it takes around 3 hours to build the whole index.

To run a test, we can use 1% of the dataset. This will be accomplished by selecting the first 1% available documents and the questions referent to these documents.

```python
proportion_to_use = 1 / 100
amount_of_docs_to_use = int(len(lfrqa_docs_df) * proportion_to_use)
print(f"Using {amount_of_docs_to_use} out of {len(lfrqa_docs_df)} documents")
```

## Prepare the Document Files
We now create the document directory and store each document as a separate text file, so that paperqa can build the index.

```python
partial_docs = lfrqa_docs_df.head(amount_of_docs_to_use)
lfrqa_directory = os.path.join(rag_qa_benchmarking_dir, "lfrqa")
os.makedirs(
    os.path.join(lfrqa_directory, "science_docs_for_paperqa", "files"), exist_ok=True
)

for i, row in partial_docs.iterrows():
    doc_id = row["doc_id"]
    doc_text = row["doc_text"]

    with open(
        os.path.join(
            lfrqa_directory, "science_docs_for_paperqa", "files", f"{doc_id}.txt"
        ),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(doc_text)

    if i % int(len(partial_docs) * 0.05) == 0:
        progress = (i + 1) / len(partial_docs)
        print(f"Progress: {progress:.2%}")
```

## Create the Manifest File
The **manifest file** keeps track of document metadata for the dataset. We need to fill some fields so that paperqa doesn’t try to get metadata using llm calls. This will make the indexing process faster.

```python
manifest = partial_docs.copy()
manifest["file_location"] = manifest["doc_id"].apply(lambda x: f"files/{x}.txt")
manifest["doi"] = ""
manifest["title"] = manifest["doc_id"]
manifest["key"] = manifest["doc_id"]
manifest["docname"] = manifest["doc_id"]
manifest["citation"] = "_"
manifest = manifest.drop(columns=["doc_id", "doc_text"])
manifest.to_csv(
    os.path.join(lfrqa_directory, "science_docs_for_paperqa", "manifest.csv"),
    index=False,
)
```

## Filter and Save Questions
Finally, we load the questions and filter them to ensure we only include questions that reference the selected documents:

```python
questions_df = pd.read_json(
    os.path.join(rag_qa_benchmarking_dir, "annotations_science_with_citation.jsonl"),
    lines=True,
)
partial_questions = questions_df[
    questions_df.gold_doc_ids.apply(
        lambda ids: all(_id < amount_of_docs_to_use for _id in ids)
    )
]
partial_questions.to_csv(
    os.path.join(lfrqa_directory, "questions.csv"),
    index=False,
)

print("Using", len(partial_questions), "questions")
```

## Install paperqa
From now on, we will be using the paperqa library, so we need to install it:

```python
!pip install paper-qa
```

## Index the Documents

Now we will build an index for the LFRQA documents. The index is a **Tantivy index**, which is a fast, full-text search engine library written in Rust. Tantivy is designed to handle large datasets efficiently, making it ideal for searching through a vast collection of papers or documents.

Feel free to adjust the concurrency settings as you like. Because we defined a manifest, we don’t need any API keys for building this index because we don't discern any citation metadata, but you do need LLM API keys to answer questions.

Remember that this process is quick for small portions of the dataset, but can take around 3 hours for the whole dataset.

```python
import nest_asyncio

nest_asyncio.apply()
```

We add the line above to handle async code within a notebook.

However, to improve compatibility and speed up the indexing process, we strongly recommend running the following code in a separate `.py` file

```python
import os

from paperqa import Settings
from paperqa.agents import build_index
from paperqa.settings import AgentSettings, IndexSettings, ParsingSettings

settings = Settings(
    agent=AgentSettings(
        index=IndexSettings(
            name="lfrqa_science_index",
            paper_directory=os.path.join(
                "data", "rag-qa-benchmarking", "lfrqa", "science_docs_for_paperqa"
            ),
            index_directory=os.path.join(
                "data", "rag-qa-benchmarking", "lfrqa", "science_docs_for_paperqa_index"
            ),
            manifest_file="manifest.csv",
            concurrency=10_000,
            batch_size=10_000,
        )
    ),
    parsing=ParsingSettings(
        use_doc_details=False,
        defer_embedding=True,
    ),
)

build_index(settings=settings)
```

After this runs, you will have an index ready to use!


## Benchmark!
After you have built the index, you are ready to run the benchmark. We advice running this in a separate `.py` file.

To run this, you will need to have the [`ldp`](https://github.com/Future-House/ldp) and [`fhaviary[lfrqa]`](https://github.com/Future-House/aviary/blob/main/packages/lfrqa/README.md#installation) packages installed.


```python
!pip install ldp "fhaviary[lfrqa]"
```

```python
import asyncio
import json
import logging
import os

import pandas as pd
from aviary.envs.lfrqa import LFRQAQuestion, LFRQATaskDataset
from ldp.agent import SimpleAgent
from ldp.alg.runners import Evaluator, EvaluatorConfig

from paperqa import Settings
from paperqa.settings import AgentSettings, IndexSettings

logging.basicConfig(level=logging.ERROR)

log_results_dir = os.path.join("data", "rag-qa-benchmarking", "results")
os.makedirs(log_results_dir, exist_ok=True)


async def log_evaluation_to_json(
    lfrqa_question_evaluation: dict,
) -> None:  # noqa: RUF029
    json_path = os.path.join(
        log_results_dir, f"{lfrqa_question_evaluation['qid']}.json"
    )
    with open(json_path, "w") as f:  # noqa: ASYNC230
        json.dump(lfrqa_question_evaluation, f, indent=2)


async def evaluate() -> None:
    settings = Settings(
        agent=AgentSettings(
            index=IndexSettings(
                name="lfrqa_science_index",
                paper_directory=os.path.join(
                    "data", "rag-qa-benchmarking", "lfrqa", "science_docs_for_paperqa"
                ),
                index_directory=os.path.join(
                    "data",
                    "rag-qa-benchmarking",
                    "lfrqa",
                    "science_docs_for_paperqa_index",
                ),
            )
        )
    )

    data: list[LFRQAQuestion] = [
        LFRQAQuestion(**row)
        for row in pd.read_csv(
            os.path.join("data", "rag-qa-benchmarking", "lfrqa", "questions.csv")
        )[["qid", "question", "answer", "gold_doc_ids"]].to_dict(orient="records")
    ]

    dataset = LFRQATaskDataset(
        data=data,
        settings=settings,
        evaluation_callback=log_evaluation_to_json,
    )

    evaluator = Evaluator(
        config=EvaluatorConfig(batch_size=3),
        agent=SimpleAgent(),
        dataset=dataset,
    )
    await evaluator.evaluate()


if __name__ == "__main__":
    asyncio.run(evaluate())
```


After running this, you can find the results in the `data/rag-qa-benchmarking/results` folder. Here is an example of how to read them:

```python
import glob

json_files = glob.glob(os.path.join(rag_qa_benchmarking_dir, "results", "*.json"))

data = []
for file in json_files:
    with open(file) as f:
        json_data = json.load(f)
        json_data["qid"] = file.split("/")[-1].replace(".json", "")
        data.append(json_data)

results_df = pd.DataFrame(data).set_index("qid")
results_df["winner"].value_counts(normalize=True)
```
