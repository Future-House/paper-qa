# Preparing the LFRQA Dataset

## Overview

The **LFRQA dataset** was introduced in the paper [*LFRQA: Large-Scale Few-Shot Retrieval Question Answering*](https://arxiv.org/pdf/2407.13998). It features **1,404 science questions** (along with other categories) that have been human-annotated with answers. This tutorial walks through the process of setting up the dataset for use.

## Step 1: Download the Annotations

First, we need to obtain the annotated dataset from the official repository:

```bash
# Clone the RAG-QA Arena repository
!git clone https://github.com/awslabs/rag-qa-arena

# Create a new directory for the dataset
!mkdir rag-qa-benchmarking

# Move the annotations file to our working directory
!mv rag-qa-arena/data/annotations_science_with_citation.jsonl ./rag-qa-benchmarking/

# Clean up the repository folder
!rm -rf rag-qa-arena
```

## Step 2: Download the Robust-QA Documents

LFRQA is built upon **Robust-QA**, so we must download the relevant documents:

```bash
# Download the Lotte dataset, which includes the required documents
!curl https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz --output lotte.tar.gz

# Extract the dataset
!tar -xvzf lotte.tar.gz

# Move the science test collection to our dataset folder
!cp lotte/science/test/collection.tsv ./rag-qa-benchmarking/science_test_collection.tsv

# Clean up unnecessary files
!rm lotte.tar.gz
!rm -rf lotte
```

## Step 3: Load the Data

We now load both the questions and documents into a usable format:

```python
import os
import pandas as pd

# Load questions dataset
questions = pd.read_json("rag-qa-benchmarking/annotations_science_with_citation.jsonl", lines=True)

# Load documents dataset
docs = pd.read_csv(
    "rag-qa-benchmarking/science_test_collection.tsv",
    sep="\t",
    header=None,
)
docs.columns = ["doc_id", "doc_text"]
```

## Step 4: Select the Documents to Use

If needed, we can limit the number of documents used:

```python
percentage_to_use = 100  # Adjust this to use a subset of documents
proportion_to_use = percentage_to_use / 100
papers_directory = 'rag-qa-benchmarking/lfrqa'
amount_of_docs_to_use = int(len(docs) * proportion_to_use)
partial_docs = docs.head(amount_of_docs_to_use)
print(f"Using {amount_of_docs_to_use} out of {len(docs)} documents")
```

## Step 5: Prepare the Document Files

We now create the document directory and store each document as a separate text file:

```python
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

The **manifest file** keeps track of document metadata for the dataset:

```python
manifest = partial_docs.copy()
manifest['file_location'] = manifest['doc_id'].apply(lambda x: f"files/{x}.txt")
manifest['doi'] = ""
manifest['overwrite_fields_from_metadata'] = False
manifest['title'] = manifest['doc_id'].apply(lambda x: x)
manifest['key'] = manifest['title']
manifest['docname'] = manifest['title']
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

## Conclusion

You now have the **LFRQA dataset** prepared for use! The dataset includes properly formatted documents and a manifest file that makes it easier to work with **PaperQA** or other retrieval-based QA frameworks.

For more details, refer to the original paper: [*LFRQA: Large-Scale Few-Shot Retrieval Question Answering*](https://arxiv.org/pdf/2407.13998).

