---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: test
    language: python
    name: python3
---

# Setup

> This tutorial is available as a Jupyter notebook [here](https://github.com/Future-House/paper-qa/blob/main/docs/tutorials/settings_tutorial.ipynb).

This tutorial aims to show how to use the `Settings` class to configure `PaperQA`.
Firstly, we will be using `OpenAI` and `Anthropic` models, so we need to set the `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variables.
We will use both models to make it clear when `paperqa` agent is using either one or the other.
We use `python-dotenv` to load the environment variables from a `.env` file.
Hence, our first step is to create a `.env` file and install the required packages.

```python
# fmt: off
# Create .env file with OpenAI API and Anthropic API keys
# Replace <your-openai-api-key> and <your-anthropic-api-key> with your actual API keys
!echo "OPENAI_API_KEY=<your-openai-api-key>" > .env # fmt: skip
!echo "ANTHROPIC_API_KEY=<your-anthropic-api-key>" >> .env # fmt: skip

!uv pip install -q nest-asyncio python-dotenv aiohttp fhlmi "paper-qa[local]"
# fmt: on
```

```python
import os

import aiohttp
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv(".env")
```

```python
print("You have set the following environment variables:")
print(
    f"OPENAI_API_KEY:    {'is set' if os.environ['OPENAI_API_KEY'] else 'is not set'}"
)
print(
    f"ANTHROPIC_API_KEY: {'is set' if os.environ['ANTHROPIC_API_KEY'] else 'is not set'}"
)
```

We will use the `lmi` package to get the model names and the `.papers` directory to save documents we will use.

```python
from lmi import CommonLLMNames

llm_openai = CommonLLMNames.OPENAI_TEST.value
llm_anthropic = CommonLLMNames.ANTHROPIC_TEST.value

# Create the `papers` directory if it doesn't exist
os.makedirs("papers", exist_ok=True)

# Download the paper from arXiv and save it to the `papers` directory
url = "https://arxiv.org/pdf/2407.01603"
async with aiohttp.ClientSession() as session, session.get(url, timeout=60) as response:
    content = await response.read()
    with open("papers/2407.01603.pdf", "wb") as f:
        f.write(content)
```

The `Settings` class is used to configure the PaperQA settings.
Official documentation can be found [here](https://github.com/Future-House/paper-qa?tab=readme-ov-file#settings-cheatsheet) and the open source code can be found [here](https://github.com/Future-House/paper-qa/blob/main/paperqa/settings.py).

Here is a basic example of how to use the `Settings` class. We will be unnecessarily verbose for the sake of clarity. Please notice that most of the settings are optional and the defaults are good for most cases. Refer to the [descriptions of each setting](https://github.com/Future-House/paper-qa/blob/main/paperqa/settings.py) for more information.

Within this `Settings` object, I'd like to discuss specifically how the llms are configured and how `paperqa` looks for papers.

A common source of confusion is that multiple `llms` are used in paperqa. We have `llm`, `summary_llm`, `agent_llm`, and `embedding`. Hence, if `llm` is set to an `Anthropic` model, `summary_llm` and `agent_llm` will still require a `OPENAI_API_KEY`, since `OpenAI` models are the default.

Among the objects that use `llms` in `paperqa`, we have `llm`, `summary_llm`, `agent_llm`, and `embedding`:

- `llm`: Main LLM used by the agent to reason about the question, extract metadata from documents, etc.
- `summary_llm`: LLM used to summarize the papers.
- `agent_llm`: LLM used to answer questions and select tools.
- `embedding`: Embedding model used to embed the papers.

Let's see some examples around this concept. First, we define the settings with `llm` set to an `OpenAI` model. Please notice this is not an complete list of settings. But take your time to read through this `Settings` class and all customization that can be done.

```python
import pathlib

from paperqa.prompts import (
    CONTEXT_INNER_PROMPT,
    CONTEXT_OUTER_PROMPT,
    citation_prompt,
    default_system_prompt,
    env_reset_prompt,
    env_system_prompt,
    qa_prompt,
    select_paper_prompt,
    structured_citation_prompt,
    summary_json_prompt,
    summary_json_system_prompt,
    summary_prompt,
)
from paperqa.settings import (
    AgentSettings,
    AnswerSettings,
    IndexSettings,
    ParsingSettings,
    PromptSettings,
    Settings,
)

settings = Settings(
    llm=llm_openai,
    llm_config={
        "model_list": [
            {
                "model_name": llm_openai,
                "litellm_params": {
                    "model": llm_openai,
                    "temperature": 0.1,
                    "max_tokens": 4096,
                },
            }
        ],
        "rate_limit": {
            llm_openai: "30000 per 1 minute",
        },
    },
    summary_llm=llm_openai,
    summary_llm_config={
        "rate_limit": {
            llm_openai: "30000 per 1 minute",
        },
    },
    embedding="text-embedding-3-small",
    embedding_config={},
    temperature=0.1,
    batch_size=1,
    verbosity=1,
    manifest_file=None,
    paper_directory=pathlib.Path.cwd().joinpath("papers"),
    index_directory=pathlib.Path.cwd().joinpath("papers/index"),
    answer=AnswerSettings(
        evidence_k=10,
        evidence_detailed_citations=True,
        evidence_retrieval=True,
        evidence_summary_length="about 100 words",
        evidence_skip_summary=False,
        answer_max_sources=5,
        max_answer_attempts=None,
        answer_length="about 200 words, but can be longer",
        max_concurrent_requests=10,
    ),
    parsing=ParsingSettings(
        chunk_size=5000,
        overlap=250,
        citation_prompt=citation_prompt,
        structured_citation_prompt=structured_citation_prompt,
    ),
    prompts=PromptSettings(
        summary=summary_prompt,
        qa=qa_prompt,
        select=select_paper_prompt,
        pre=None,
        post=None,
        system=default_system_prompt,
        use_json=True,
        summary_json=summary_json_prompt,
        summary_json_system=summary_json_system_prompt,
        context_outer=CONTEXT_OUTER_PROMPT,
        context_inner=CONTEXT_INNER_PROMPT,
    ),
    agent=AgentSettings(
        agent_llm=llm_openai,
        agent_llm_config={
            "model_list": [
                {
                    "model_name": llm_openai,
                    "litellm_params": {
                        "model": llm_openai,
                    },
                }
            ],
            "rate_limit": {
                llm_openai: "30000 per 1 minute",
            },
        },
        agent_prompt=env_reset_prompt,
        agent_system_prompt=env_system_prompt,
        search_count=8,
        index=IndexSettings(
            paper_directory=pathlib.Path.cwd().joinpath("papers"),
            index_directory=pathlib.Path.cwd().joinpath("papers/index"),
        ),
    ),
)
```

As it is evident, `Paperqa` is absolutely customizable. And here we reinterate that despite this possible fine customization, the defaults are good for most cases. Although, the user is welcome to explore the settings and customize the `paperqa` to their needs.

We also set settings.verbosity to 1, which will print the agent configuration. Feel free to set it to 0 to silence the logging after your first run.

```python
from paperqa import ask

response = ask(
    "What are the most relevant language models used for chemistry?", settings=settings
)
```

Which probably worked fine. Let's now try to remove `OPENAI_API_KEY` and run again the same question with the same settings.

```python
os.environ["OPENAI_API_KEY"] = ""
print("You have set the following environment variables:")
print(
    f"OPENAI_API_KEY:    {'is set' if os.environ['OPENAI_API_KEY'] else 'is not set'}"
)
print(
    f"ANTHROPIC_API_KEY: {'is set' if os.environ['ANTHROPIC_API_KEY'] else 'is not set'}"
)
```

```python
response = ask(
    "What are the most relevant language models used for chemistry?", settings=settings
)
```

It would obviously fail. We don't have a valid `OPENAI_API_KEY`, so the agent will not be able to use `OpenAI` models. Let's change it to an `Anthropic` model and see if it works.

```python
settings.llm = llm_anthropic
settings.llm_config = {
    "model_list": [
        {
            "model_name": llm_anthropic,
            "litellm_params": {
                "model": llm_anthropic,
                "temperature": 0.1,
                "max_tokens": 512,
            },
        }
    ],
    "rate_limit": {
        llm_anthropic: "30000 per 1 minute",
    },
}
settings.summary_llm = llm_anthropic
settings.summary_llm_config = {
    "rate_limit": {
        llm_anthropic: "30000 per 1 minute",
    },
}
settings.agent = AgentSettings(
    agent_llm=llm_anthropic,
    agent_llm_config={
        "rate_limit": {
            llm_anthropic: "30000 per 1 minute",
        },
    },
    index=IndexSettings(
        paper_directory=pathlib.Path.cwd().joinpath("papers"),
        index_directory=pathlib.Path.cwd().joinpath("papers/index"),
    ),
)
settings.embedding = "st-multi-qa-MiniLM-L6-cos-v1"
response = ask(
    "What are the most relevant language models used for chemistry?", settings=settings
)
```

<!-- #region -->
Now the agent is able to use `Anthropic` models only and although we don't have a valid `OPENAI_API_KEY`, the question is answered because the agent will not use `OpenAI` models. See that we also changed the `embedding` because it was using `text-embedding-3-small` by default, which is a `OpenAI` model. `Paperqa` implements a few embedding models. Please refer to the [documentation](https://github.com/Future-House/paper-qa?tab=readme-ov-file#embedding-model) for more information.

Notice that we redefined `settings.agent.paper_directory` and `settings.agent.index` settings. `Paperqa` actually uses the setting from `settings.agent`. However, for convenience, we implemented an alias in `settings.paper_directory` and `settings.index_directory`.

In addition, notice that this is a very verbose example for the sake of clarity. We could have just set only the llms names and used default settings for the rest:

```python
llm_anthropic_config = {
    "model_list": [{
            "model_name": llm_anthropic,
    }]
}

settings.llm = llm_anthropic
settings.llm_config = llm_anthropic_config
settings.summary_llm = llm_anthropic
settings.summary_llm_config = llm_anthropic_config
settings.agent = AgentSettings(
    agent_llm=llm_anthropic,
    agent_llm_config=llm_anthropic_config,
    index=IndexSettings(
        paper_directory=pathlib.Path.cwd().joinpath("papers"),
        index_directory=pathlib.Path.cwd().joinpath("papers/index"),
    ),
)
settings.embedding = "st-multi-qa-MiniLM-L6-cos-v1"
```

<!-- #endregion -->

# The output

`Paperqa` returns a `PQASession` object, which contains not only the answer but also all the information gatheres to answer the questions. We recommend printing the `PQASession` object (`print(response.session)`) to understand the information it contains. Let's check the `PQASession` object:

```python
print(response.session)
```

```python
print("Let's examine the PQASession object returned by paperqa:\n")

print(f"Status: {response.status.value}")

print("1. Question asked:")
print(f"{response.session.question}\n")

print("2. Answer provided:")
print(f"{response.session.answer}\n")
```

In addition to the answer, the `PQASession` object contains all the references and contexts used to generate the answer.

Because `paperqa` splits the documents into chunks, each chunk is a valid reference. You can see that it also references the page where the context was found.

```python
print("3. References cited:")
print(f"{response.session.references}\n")
```

Lastly, `PQASession.session.contexts` contains the contexts used to generate the answer. Each context has a score, which is the similarity between the question and the context.
`Paperqa` uses this score to choose what contexts is more relevant to answer the question.

```python
print("4. Contexts used to generate the answer:")
print(
    "These are the relevant text passages that were retrieved and used to formulate the answer:"
)
for i, ctx in enumerate(response.session.contexts, 1):
    print(f"\nContext {i}:")
    print(f"Source: {ctx.text.name}")
    print(f"Content: {ctx.context}")
    print(f"Score: {ctx.score}")
```
