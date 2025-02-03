# Preparing the LFRQA Dataset

## Overview

The **LFRQA dataset** was introduced in the paper [*LFRQA: Large-Scale Few-Shot Retrieval Question Answering*](https://arxiv.org/pdf/2407.13998). It features **1,404 science questions** (along with other categories) that have been human-annotated with answers. This tutorial walks through the process of setting up the dataset for use.

## Step 1: Download the Annotations

First, we need to obtain the annotated dataset from the official repository:

```bash
# Clone the RAG-QA Arena repository
!git clone <https://github.com/awslabs/rag-qa-arena>

# Create a new directory for the dataset
!mkdir rag-qa-benchmarking

# Move the science annotations file to our working directory
!mv rag-qa-arena/data/annotations_science_with_citation.jsonl ./rag-qa-benchmarking/

# Clean up the repository folder
!rm -rf rag-qa-arena
```

## Step 2: Download the Robust-QA Documents

LFRQA is built upon **Robust-QA**, so we must download the relevant documents:

```bash
# Download the Lotte dataset, which includes the required documents
!curl <https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz> --output lotte.tar.gz

# Extract the dataset
!tar -xvzf lotte.tar.gz

# Move the science test collection to our dataset folder
!cp lotte/science/test/collection.tsv ./rag-qa-benchmarking/science_test_collection.tsv

# Clean up unnecessary files
!rm lotte.tar.gz
!rm -rf lotte
```

For more details, refer to the original paper: [*LFRQA: Large-Scale Few-Shot Retrieval Question Answering*](https://arxiv.org/pdf/2407.13998).

## Step 3: Load the Data

We now load both the questions and documents into a usable format:

```python
import os
import pandas as pd

# Load questions and answers dataset
questions = pd.read_json("rag-qa-benchmarking/annotations_science_with_citation.jsonl", lines=True)

# Load documents dataset
docs = pd.read_csv(
    "rag-qa-benchmarking/science_test_collection.tsv",
    sep="\\t",
    header=None,
)
docs.columns = ["doc_id", "doc_text"]

```

## Step 4: Select the Documents to Use

If needed, we can limit the number of documents used. RobustQA consists on 1.7M documents, so the index will take a long time to build. 

If you want to run a quick test, you can use a small proportion.

Setting the proportion will take only that fraction of the documents, and the questions that can be answered only on that fraction of the dataset.

```python
proportion_to_use = 1 / 100
amount_of_docs_to_use = int(len(docs) * proportion_to_use)
partial_docs = docs.head(amount_of_docs_to_use)
print(f"Using {amount_of_docs_to_use} out of {len(docs)} documents")

```

## Step 5: Prepare the Document Files

We now create the document directory and store each document as a separate text file. This is because of how paperqa builds the index.

If you’re using the whole dataset, this may take a while.

```python
papers_directory = "rag-qa-benchmarking/lfrqa"
os.makedirs(f"{papers_directory}/science_docs_for_paperqa/files", exist_ok=True)

for i, row in partial_docs.iterrows():
    doc_id = row["doc_id"]
    doc_text = row["doc_text"]

    with open(f"{papers_directory}/science_docs_for_paperqa/files/{doc_id}.txt", "w", encoding="utf-8") as f:
        f.write(doc_text)

    if i % int(len(partial_docs) * 0.05) == 0:
        progress = (i + 1) / len(partial_docs)
        print(f"Progress: {progress:.2%}")
```

## Step 6: Create the Manifest File

The **manifest file** keeps track of document metadata for the dataset. It is also necesary so that paperqa doesn’t try to get metadata using llm calls.

```python
manifest = partial_docs.copy()
manifest['file_location'] = manifest['doc_id'].apply(lambda x: f"files/{x}.txt")
manifest['doi'] = ""
manifest['title'] = manifest['doc_id']
manifest['key'] = manifest['doc_id']
manifest['docname'] = manifest['doc_id']
manifest['citation'] = "_"
manifest.drop(columns=['doc_id', 'doc_text'], inplace=True)
manifest.to_csv(f"{papers_directory}/science_docs_for_paperqa/manifest.csv", index=False)

```

## Step 7: Filter and Save Questions

Finally, we filter the question set to ensure we only include questions that reference the selected documents:

```python
partial_questions = questions[
    questions.gold_doc_ids.apply(lambda ids: all(id < amount_of_docs_to_use for id in ids))
]
partial_questions.to_csv(f"{papers_directory}/questions.csv", index=False)

```

## Step 8: Index the documents

Copy the following to a file and run it. Feel free to adjust the concurrency as you like. 

PaperQA builds the index every time a question is asked and new files are present. So running this will build the index for you.

You don’t need any api keys for building the index, but you do need them to answer questions.

This process is quick for small portions of the whole document’s dataset, but can take ~3hs for the whole of it.

If you want to use the index we built with the whole dataset, you can get it here. Make sure you put it in the correct folder and change the index name to `"lfrqa_science_index_complete"`

```python
from paperqa import Settings, ask

settings = Settings()
settings.agent.index.name = "lfrqa_science_index"
settings.agent.index.paper_directory = (
    "rag-qa-benchmarking/lfrqa/science_docs_for_paperqa"
)
settings.agent.index.index_directory = (
    "rag-qa-benchmarking/lfrqa/science_docs_for_paperqa_index"
)

settings.agent.index.manifest_file = "manifest.csv"

settings.parsing.use_doc_details = False
settings.parsing.defer_embedding = True
settings.agent.index.concurrency = 30000

answer_response = (
    ask(
        "$5^n+n$ is never prime?",
        settings=settings,
    ),
)

print("_" * 100)

```

After this runs, you will get an answer!

## Step 9:  Benchmark!

After you have built the index, you are ready to run the benchmark.

Copy the following into a file `gradable.py`  and run it. You can also use the parameter num_questions in `LFRQATaskDataset` so you can make quick tests. 

```python
import asyncio
from ldp.agent import SimpleAgent
from ldp.alg.callbacks import MeanMetricsCallback
from ldp.alg.runners import Evaluator, EvaluatorConfig

from paperqa import Settings
from paperqa.agents.task import LFRQATaskDataset

async def evaluate() -> None:
    settings = Settings()
    settings.agent.index.name = "lfrqa_science_index"
    settings.agent.index.paper_directory = (
        "rag-qa-benchmarking/lfrqa/science_docs_for_paperqa"
    )
    settings.agent.index.index_directory = (
        "rag-qa-benchmarking/lfrqa/science_docs_for_paperqa_index"
    )
    settings.agent.index.manifest_file = "manifest.csv"

    settings.parsing.use_doc_details = False

    dataset = LFRQATaskDataset(
        data_path="rag-qa-benchmarking/lfrqa/questions.csv",
        settings=settings,
    )
    metrics_callback = MeanMetricsCallback(eval_dataset=dataset)

    evaluator = Evaluator(
        config=EvaluatorConfig(batch_size=3),
        agent=SimpleAgent(),
        dataset=dataset,
        callbacks=[metrics_callback],
    )
    await evaluator.evaluate()

    print(metrics_callback.eval_means)

if __name__ == "__main__":
    asyncio.run(evaluate())

```

## Conclusion

This tutorial has walked through the complete process of preparing and using the LFRQA dataset for benchmarking RAG systems. By following these steps, you'll have a fully functional setup for testing question-answering capabilities against a diverse set of scientific questions. The framework allows for flexible testing with different dataset sizes, making it suitable for both quick prototyping and comprehensive evaluations.