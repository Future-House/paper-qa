# PaperQA Claude Code Plugin

This plugin packages a PaperQA skill for Claude Code. It helps Claude Code decide when to use PaperQA, how to prepare a paper corpus, which settings docs to inspect, and how to preserve citation provenance in answers.

## Install

From a clone of this repository:

```bash
claude plugin install ./claude-code/paper-qa
```

Then start a new Claude Code session and ask for PaperQA-backed help, for example:

```text
Use PaperQA to answer this question from the PDFs in ./papers: what evidence supports retrieval-augmented generation for biomedical QA?
```

## Contents

- `.claude-plugin/plugin.json`: Claude Code plugin metadata.
- `skills/paper-qa/SKILL.md`: auto-activating PaperQA research workflow guidance.

The plugin expects PaperQA to be installed in the user's environment, for example with `pip install "paper-qa>=5"`.
