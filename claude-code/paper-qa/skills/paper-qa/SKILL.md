---
name: paper-qa
description: Use when an agent needs to answer questions from scientific papers with citations using PaperQA, build or query a local paper collection, choose settings, add documents from files/URLs/Zotero/OpenReview, or produce evidence-grounded answers while preserving citation provenance.
---

# PaperQA Skill

## Maintenance Note

`docs/skills/paper-qa/SKILL.md` is the source of truth for this skill. `claude-code/paper-qa/skills/paper-qa/SKILL.md` vendors the same content for Claude Code plugin packaging and should be kept in sync when the source skill changes.

## Overview

Use PaperQA for evidence-grounded question answering over scientific documents. This skill tells an agent how to prepare sources, choose settings, run searches, and report answers with citations.

## When to Use

Use this skill when a user wants to:

- answer a research question from papers or PDFs
- search, index, or query a local paper collection
- add papers from files, folders, URLs, Zotero, or OpenReview
- tune PaperQA settings for models, embeddings, retrieval, or answer behavior
- produce answers that keep citation provenance visible

## Decision Order

1. Prefer user-provided PDFs, local folders, Zotero/OpenReview collections, or known URLs before broad web search.
2. Build or reuse a PaperQA collection before answering.
3. Use PaperQA settings rather than ad hoc prompts for LLM, embedding, retrieval, and answer behavior.
4. Preserve citation metadata and quote only short excerpts.
5. Report missing evidence instead of inventing citations.

## Workflow

1. Clarify the research question and corpus boundary.
2. Gather papers or documents from local PDFs, folders, URLs, Zotero, or OpenReview.
3. Create or load a PaperQA document collection.
4. Ask the question with settings suited to the user's environment.
5. Return the answer, citations, source list, and evidence limitations.
6. Save reusable commands or settings when the user will repeat the workflow.

## Command Patterns

Prefer the current PaperQA CLI and API examples in this repository. Typical agent tasks:

- inspect `README.md` for installation, CLI, and API examples
- inspect `docs/tutorials/settings_tutorial.md` for settings guidance
- inspect `docs/tutorials/where_do_I_get_papers.md` for source acquisition
- run a small smoke query before large indexing jobs

Example CLI flow:

```bash
pip install "paper-qa>=5"
mkdir -p papers
curl -L -o papers/PaperQA2.pdf https://arxiv.org/pdf/2409.13740
cd papers
pqa ask "What is PaperQA2?"
```

For provider-specific models, configure the relevant environment variables and PaperQA settings described in the README instead of hardcoding credentials into commands or prompts.

## Output Expectations

Return:

- a concise answer
- cited evidence from the corpus
- a source list
- uncertainty or missing evidence
- exact commands or settings used when helpful for reproducibility

## Guardrails

- Do not fabricate citations.
- Do not silently mix unrelated corpora.
- Do not upload private documents to remote services unless the user has approved the provider and settings.
- Respect publisher and license constraints for full text.
- If a query has no support in the corpus, say so and suggest corpus expansion.
