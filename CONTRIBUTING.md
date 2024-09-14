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
pip install -e .
```

## Running tests and other tooling

Use the following commands:

- Run tests (requires an OpenAI key in your environment)

  ```bash
  pytest
  # or for multiprocessing based parallelism
  pytest -n auto
  ```

- Run `pre-commit` for formatting and type checking

  ```bash
  pre-commit run --all-files
  ```

- Run `mypy`, `refurb`, or `pylint` directly:

  ```bash
  mypy paperqa
  # or
  refurb paperqa
  # or
  pylint paperqa
  ```

See our GitHub Actions [`tests.yml`](.github/workflows/tests.yml) for further reference.

## Additional resources

For more information on contributing, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file in the repository.

Happy coding!
