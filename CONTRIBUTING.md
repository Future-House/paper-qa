# Contributing to PaperQA

Thank you for your interest in contributing to PaperQA!
Here are some guidelines to help you get started.

## Setting up the development environment

We use [`uv`](https://github.com/astral-sh/uv) for our local development.

1. Install `uv` by following the instructions on the [uv website](https://astral.sh/uv/).
2. Run the following command to install all dependencies and set up the development environment:

   ```bash
   uv sync
   ```

## Installing the package for development

If you prefer to use `pip` for installing the package in development mode, you can do so by running:

```bash
pip install -e ".[dev]"
```

Where the `dev` extra includes development dependencies such as `pytest`.

## Running tests and other tooling

Use the following commands:

- Run the same credential-free tests as CI:

  ```bash
  LITELLM_LOCAL_MODEL_COST_MAP=true PQA_OFFLINE_TESTS=true \
    pytest -m 'offline and not live_api'
  ```

  This mode uses local fixtures and VCR cassettes, supplies placeholder API keys,
  and disables cassette recording. Tests marked `live_api` need provider credentials
  and are excluded from CI.

- Run all tests, including live API tests (requires provider keys):

  ```bash
  pytest
  # or for multiprocessing based parallelism
  pytest -n auto
  ```

- Run `pre-commit` for formatting and type checking

  ```bash
  pre-commit run --all-files
  ```

- Run `ty` or `pylint` directly:

  ```bash
  ty check
  # or
  pylint paperqa
  ```

See our GitHub Actions [`tests.yml`](.github/workflows/tests.yml) for further reference.

## Using `pytest-recording` and VCR cassettes

We use the [`pytest-recording`](https://github.com/kiwicom/pytest-recording) plugin
to create VCR cassettes to cache HTTP requests,
making our unit tests more deterministic.

To record a new VCR cassette:

```bash
uv run pytest --record-mode=once tests/desired_test_module.py
```

And the new cassette(s) should appear in [`tests/cassettes`](tests/cassettes).
Use `--record-mode=all` to refresh an existing cassette after model changes.

Our configuration for `pytest-recording` can be found in [`tests/conftest.py`](tests/conftest.py).
This includes header removals (e.g. OpenAI `authorization` key)
from responses to ensure sensitive information is excluded from the cassettes.

Keep cassettes small where possible. Media requests and tokenizer data can make
some recorded files larger than 1 MB.

Happy coding!
