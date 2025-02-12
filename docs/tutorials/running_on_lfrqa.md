# Preparing the LFRQA Dataset

## Overview

The **LFRQA dataset** was introduced in the paper [_LFRQA: Large-Scale Few-Shot Retrieval Question Answering_](https://arxiv.org/pdf/2407.13998). It features **1,404 science questions** (along with other categories) that have been human-annotated with answers. This tutorial walks through the process of setting up the dataset for use.

## Download the Annotations

First, we need to obtain the annotated dataset from the official repository:

```bash
# Clone the RAG-QA Arena repository
git clone https://github.com/awslabs/rag-qa-arena

# Create a new directory for the dataset
mkdir data
mkdir data/rag-qa-benchmarking

# Move the science annotations file to our working directory
mv rag-qa-arena/data/annotations_science_with_citation.jsonl ./data/rag-qa-benchmarking/

# Clean up the repository folder
rm -rf data/rag-qa-arena
```

## Download the Robust-QA Documents

LFRQA is built upon **Robust-QA**, so we must download the relevant documents:

```bash
# Download the Lotte dataset, which includes the required documents
curl https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz --output data/lotte.tar.gz

# Extract the dataset
tar -xvzf data/lotte.tar.gz

# Move the science test collection to our dataset folder
cp data/lotte/science/test/collection.tsv ./data/rag-qa-benchmarking/science_test_collection.tsv

# Clean up unnecessary files
rm data/lotte.tar.gz
rm -rf data/lotte
```

For more details, refer to the original paper: [_LFRQA: Large-Scale Few-Shot Retrieval Question Answering_](https://arxiv.org/pdf/2407.13998).

## Load the Data

We now load both the questions and documents into a usable format:

```python
import os
import pandas as pd

# Load questions and answers dataset
questions = pd.read_json(
    os.path.join(
        "data", "rag-qa-benchmarking", "annotations_science_with_citation.jsonl"
    ),
    lines=True,
)

# Load documents dataset
docs = pd.read_csv(
    os.path.join("data", "rag-qa-benchmarking", "science_test_collection.tsv"),
    sep="\t",
    names=["doc_id", "doc_text"],
)
```

## Select the Documents to Use

If needed, we can limit the number of documents used. RobustQA consists on 1.7M documents, so the index will take a long time to build.

If you want to run a quick test, you can use a small proportion.

Setting the proportion will take only that fraction of the documents, and the questions that can be answered only on that fraction of the dataset.

```python
proportion_to_use = 1 / 100
amount_of_docs_to_use = int(len(docs) * proportion_to_use)
print(f"Using {amount_of_docs_to_use} out of {len(docs)} documents")
```

## Prepare the Document Files

We now create the document directory and store each document as a separate text file. This is because of how paperqa builds the index.

If you’re using the whole dataset, this may take a while.

```python
partial_docs = docs.head(amount_of_docs_to_use)
lfrqa_directory = os.path.join("data", "rag-qa-benchmarking", "lfrqa")
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

The **manifest file** keeps track of document metadata for the dataset. It is also necessary so that paperqa doesn’t try to get metadata using llm calls.

```python
manifest = partial_docs.copy()
manifest["file_location"] = manifest["doc_id"].apply(lambda x: f"files/{x}.txt")
manifest["doi"] = ""
manifest["title"] = manifest["doc_id"]
manifest["key"] = manifest["doc_id"]
manifest["docname"] = manifest["doc_id"]
manifest["citation"] = ""
manifest.drop(columns=["doc_id", "doc_text"], inplace=True)
manifest.to_csv(
    os.path.join(lfrqa_directory, "science_docs_for_paperqa", "manifest.csv"),
    index=False,
)
```

## Filter and Save Questions

Finally, we filter the question set to ensure we only include questions that reference the selected documents:

```python
partial_questions = questions[
    questions.gold_doc_ids.apply(
        lambda ids: all(id < amount_of_docs_to_use for id in ids)
    )
]
partial_questions.to_csv(
    os.path.join(lfrqa_directory, "questions.csv"),
    index=False,
)
```

## Install paperqa

```bash
pip install paper-qa
```

## Index the documents

Copy the following to a file and run it. Feel free to adjust the concurrency as you like.

PaperQA builds the index every time a question is asked and new files are present. So running this will build the index for you.

You don’t need any api keys for building the index, but you do need them to answer questions.

This process is quick for small portions of the whole document’s dataset, but can take ~3hs for the whole of it.

If you want to use the index we built with the whole dataset, you can get it here. Make sure that you have all the docs in the `data/rag-qa-benchmarking/lfrqa/science_docs_for_paperqa/files` folder so the indexing process doesn't delete the missing ones.

```python
from paperqa import Settings, ask
from paperqa.settings import AgentSettings, IndexSettings, ParsingSettings

settings = Settings(
    agent=AgentSettings(
        index=IndexSettings(
            name="lfrqa_science_index",
            paper_directory="data/rag-qa-benchmarking/lfrqa/science_docs_for_paperqa",
            index_directory="data/rag-qa-benchmarking/lfrqa/science_docs_for_paperqa_index",
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
answer_response = (
    ask(
        "$5^n+n$ is never prime?",
        settings=settings,
    ),
)
```

After this runs, you will get an answer!

## Benchmark!

After you have built the index, you are ready to run the benchmark.

Copy the following into a file `gradable.py` and run it. You can also use the parameter num_questions in `LFRQATaskDataset` so you can make quick tests.

To run this, you will need to have the [`ldp`](https://github.com/Future-House/ldp) package installed.

```python
import os
import json
import asyncio
import pandas as pd
from ldp.agent import SimpleAgent
from ldp.alg.runners import Evaluator, EvaluatorConfig
from paperqa import Settings
from paperqa.settings import AgentSettings, IndexSettings
from paperqa.agents.task import LFRQATaskDataset, LFRQAQuestion


async def log_evaluation_to_json(lfrqa_question_evaluation: dict) -> None:
    results_dir = os.path.join("data", "rag-qa-benchmarking", "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f"{lfrqa_question_evaluation['qid']}.json")
    with open(json_path, "w") as f:
        json.dump(lfrqa_question_evaluation, f, indent=2)


async def evaluate() -> None:
    settings = Settings(
        agent=AgentSettings(
            index=IndexSettings(
                name="lfrqa_science_index_1",
                paper_directory="data/rag-qa-benchmarking/lfrqa/science_docs_for_paperqa",
                index_directory="data/rag-qa-benchmarking/lfrqa/science_docs_for_paperqa_index",
            )
        )
    )

    data: list[LFRQAQuestion] = [
        LFRQAQuestion(**row)
        for row in pd.read_csv("data/rag-qa-benchmarking/lfrqa/questions.csv")[
            ["qid", "question", "answer", "gold_doc_ids"]
        ].to_dict(orient="records")
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
import json

json_files = glob.glob(os.path.join("data", "rag-qa-benchmarking", "results", "*.json"))

data = []
for file in json_files:
    with open(file) as f:
        json_data = json.load(f)
        json_data["qid"] = file.split("/")[-1].replace(".json", "")
        data.append(json_data)

df = pd.DataFrame(data).set_index("qid")
df["winner"].value_counts(normalize=True)
```
