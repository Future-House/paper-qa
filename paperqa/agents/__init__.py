from __future__ import annotations

import ast
import logging
import operator
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from typing_extensions import Annotated

from .. import __version__
from ..utils import get_loop, pqa_directory

try:
    import anyio
    import typer
    from rich.console import Console
    from rich.logging import RichHandler

    from .main import agent_query, search
    from .models import AnswerResponse, MismatchedModelsError, QueryRequest
    from .search import SearchIndex, get_directory_index

except ImportError as e:
    raise ImportError(
        '"agents" module is not installed please install it using "pip install paper-qa[agents]"'
    ) from e

logger = logging.getLogger(__name__)

app = typer.Typer()


def configure_agent_logging(
    verbosity: int = 0, default_level: int = logging.INFO
) -> None:
    """Default to INFO level, but suppress loquacious loggers."""
    verbosity_map = {
        0: {
            "paperqa.agents.helpers": logging.WARNING,
            "paperqa.agents.main": logging.WARNING,
            "anthropic": logging.WARNING,
            "openai": logging.WARNING,
            "httpx": logging.WARNING,
            "paperqa.agents.models": logging.WARNING,
        }
    }

    verbosity_map[1] = verbosity_map[0] | {
        "paperqa.agents.main": logging.INFO,
        "paperqa.models": logging.INFO,
    }

    verbosity_map[2] = verbosity_map[1] | {
        "paperqa.agents.helpers": logging.DEBUG,
        "paperqa.agents.main": logging.DEBUG,
        "paperqa.agents.main.agent_callers": logging.DEBUG,
        "paperqa.models": logging.DEBUG,
        "paperqa.agents.search": logging.DEBUG,
    }

    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        show_level=False,
        console=Console(force_terminal=True),
    )

    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    def is_paperqa_related(logger_name: str) -> bool:
        return logger_name.startswith("paperqa") or logger_name in {
            "anthropic",
            "openai",
            "httpx",
        }

    for logger_name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and is_paperqa_related(logger_name):
            logger.setLevel(
                verbosity_map.get(min(verbosity, 2), {}).get(logger_name, default_level)
            )
            if not any(isinstance(h, RichHandler) for h in logger.handlers):
                logger.addHandler(rich_handler)


def get_file_timestamps(path: os.PathLike | str) -> dict[str, str]:
    # Get the stats for the file/directory
    stats = os.stat(path)

    # Get created time (ctime)
    created_time = datetime.fromtimestamp(stats.st_ctime)

    # Get modified time (mtime)
    modified_time = datetime.fromtimestamp(stats.st_mtime)

    return {
        "created_at": created_time.strftime("%Y-%m-%d %H:%M:%S"),
        "modified_at": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def parse_dot_to_dict(str_w_dots: str, value: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key in str_w_dots.split(".")[::-1]:
        if not parsed:
            try:
                eval_value = ast.literal_eval(value)
                if isinstance(eval_value, (set, list)):
                    parsed[key] = eval_value
                else:
                    parsed[key] = value
            except (ValueError, SyntaxError):
                parsed[key] = value
        else:
            parsed = {key: parsed}
    return parsed


def pop_nested_dict_recursive(d: dict[str, Any], path: str) -> tuple[Any, bool]:
    """
    Pop a value from a nested dictionary (in-place) using a period-separated path.

    Recursively remove empty dictionaries after popping.
    """
    keys = path.split(".")

    if len(keys) == 1:
        if keys[0] not in d:
            raise KeyError(f"Key not found: {keys[0]}")
        value = d.pop(keys[0])
        return value, len(d) == 0

    if keys[0] not in d or not isinstance(d[keys[0]], dict):
        raise KeyError(f"Invalid path: {path}")

    value, should_remove = pop_nested_dict_recursive(d[keys[0]], ".".join(keys[1:]))

    if should_remove:
        d.pop(keys[0])

    return value, len(d) == 0


def get_settings(
    settings_path: str | os.PathLike | None = None,
) -> dict[str, Any]:

    if settings_path is None:
        settings_path = pqa_directory("settings") / "settings.yaml"

    if os.path.exists(settings_path):
        with open(settings_path) as f:
            return yaml.safe_load(f)

    return {}


def merge_dicts(dict_a: dict, dict_b: dict) -> dict:
    """
    Merge two dictionaries where if dict_a has a key with a subdictionary.

    dict_b only overwrites the keys in dict_a's subdictionary if they are
    also specified in dict_b, but otherwise keeps all the subkeys.
    """
    result = dict_a.copy()  # Start with a shallow copy of dict_a

    for key, value in dict_b.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # If both dict_a and dict_b have a dict for this key, recurse
            result[key] = merge_dicts(result[key], value)
        else:
            # Otherwise, just update the value
            result[key] = value

    return result


def get_merged_settings(
    settings: dict[str, Any], settings_path: Path | None = None
) -> dict[str, Any]:
    """Merges a new settings with the current settings saved to file."""
    current_settings = get_settings(settings_path)

    # deal with the nested key case
    return merge_dicts(current_settings, settings)


@app.command("set")
def set_setting(
    variable: Annotated[
        str,
        typer.Argument(
            help=(
                "PaperQA variable to set, see agents.models.QueryRequest object for all settings, "
                "nested options can be set using periods, ex. agent_tools.paper_directory"
            )
        ),
    ],
    value: Annotated[
        str,
        typer.Argument(
            help=(
                "Value to set to the variable, will be cast to the correct type automatically."
            )
        ),
    ],
) -> None:
    """Set a persistent PaperQA setting."""
    configure_agent_logging(verbosity=0)

    settings_path = pqa_directory("settings") / "settings.yaml"

    current_settings = get_merged_settings(
        parse_dot_to_dict(variable, value), settings_path=settings_path
    )

    try:
        QueryRequest(**current_settings)
    except MismatchedModelsError:
        pass
    except ValueError as e:
        raise ValueError(
            f"{variable} (with value {value}) is not a valid setting."
        ) from e

    logger.info(f"{variable} set to {str(value)[:100]}!")

    with open(settings_path, "w") as f:
        yaml.dump(current_settings, f)


@app.command()
def show(
    variable: Annotated[
        str,
        typer.Argument(
            help=(
                "PaperQA variable to show, see agents.models.QueryRequest object for all settings, "
                "nested options can be set using periods, ex. agent_tools.paper_directory. "
                "Can show all indexes with `indexes` input, answers with `answers` input, "
                "and `all` for all settings."
            )
        ),
    ],
    limit: Annotated[
        int, typer.Option(help="limit results, only used for 'answers'.")
    ] = 5,
) -> Any:
    """Show a persistent PaperQA setting, special inputs include `indexes`, `answers` and `all`."""
    configure_agent_logging(verbosity=0)

    # handle special case when user wants to see indexes
    if variable == "indexes":
        for index in os.listdir(pqa_directory("indexes")):
            index_times = get_file_timestamps(pqa_directory("indexes") / index)
            logger.info(f"{index}, {index_times}")
        return os.listdir(pqa_directory("indexes"))

    if variable == "answers":
        all_answers = []
        answer_file_location = pqa_directory("indexes") / "answers" / "docs"
        if os.path.exists(answer_file_location):
            for answer_file in os.listdir(answer_file_location):
                all_answers.append(
                    get_file_timestamps(os.path.join(answer_file_location, answer_file))
                )
                with open(os.path.join(answer_file_location, answer_file)) as f:
                    answer = yaml.safe_load(f)
                all_answers[-1].update({"answer": answer})
            all_answers = sorted(
                all_answers, key=operator.itemgetter("modified_at"), reverse=True
            )[:limit]
            for answer in all_answers:
                logger.info(
                    f"Q: {answer['answer']['answer']['question']}\n---\nA: {answer['answer']['answer']['answer']}\n\n\n"
                )
        return all_answers

    current_settings = get_settings(pqa_directory("settings") / "settings.yaml")

    if variable == "all":
        logger.info(current_settings)
        return current_settings

    try:
        value, _ = pop_nested_dict_recursive(current_settings, variable)
    except KeyError:
        logger.info(f"{variable} is not set.")
        return None
    else:
        logger.info(f"{variable}: {value}")
        return value


@app.command()
def clear(
    variable: Annotated[
        str,
        typer.Argument(
            help=(
                "PaperQA variable to clear, see agents.models.QueryRequest object for all settings, "
                "nested options can be set using periods, ex. agent_tools.paper_directory. "
                "Index names can also be used if the --index flag is set."
            )
        ),
    ],
    index: Annotated[
        bool,
        typer.Option(
            "--index",
            is_flag=True,
            help="index flag to indicate that this index name should be cleared.",
        ),
    ] = False,
) -> None:
    """Clear a persistent PaperQA setting, include the --index flag to remove an index."""
    configure_agent_logging(verbosity=0)

    settings_path = pqa_directory("settings") / "settings.yaml"

    current_settings = get_settings(settings_path)

    if not index:
        _ = pop_nested_dict_recursive(current_settings, variable)
        with open(settings_path, "w") as f:
            yaml.dump(current_settings, f)
        logger.info(f"{variable} cleared!")

    elif variable in os.listdir(pqa_directory("indexes")):
        shutil.rmtree(pqa_directory("indexes") / variable)
        logger.info(f"Index {variable} cleared!")
    else:
        logger.info(f"Index {variable} not found!")


@app.command()
def ask(
    query: Annotated[str, typer.Argument(help=("Question or task ask of PaperQA"))],
    agent_type: Annotated[
        str,
        typer.Option(
            help=(
                "Type of agent to use, for now either "
                "`OpenAIFunctionsAgent` or `fake`. `fake` uses "
                "a hard coded tool path (search->gather evidence->answer)."
            )
        ),
    ] = "fake",
    verbosity: Annotated[
        int, typer.Option(help=("Level of verbosity from 0->2 (inclusive)"))
    ] = 0,
    directory: Annotated[
        Path | None,
        typer.Option(help=("Directory of papers or documents to run PaperQA over.")),
    ] = None,
    index_directory: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Index directory to store paper index and answers. Default will be `~/.pqa`"
            )
        ),
    ] = None,
    manifest_file: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Optional manifest file (CSV) location to map relative a "
                "`file_location` column to `doi` or `title` columns. "
                "If not used, then the file will be read by an LLM "
                "which attempts to extract the title, authors and DOI."
            )
        ),
    ] = None,
) -> AnswerResponse:
    """Query PaperQA via an agent."""
    configure_agent_logging(verbosity=verbosity)

    loop = get_loop()

    # override settings file if requested directly
    to_merge = {}

    if directory is not None:
        to_merge = {"agent_tools": {"paper_directory": directory}}

    if index_directory is not None:
        if "agent_tools" not in to_merge:
            to_merge = {"agent_tools": {"index_directory": index_directory}}
        else:
            to_merge["agent_tools"].update({"index_directory": index_directory})

    if manifest_file is not None:
        if "agent_tools" not in to_merge:
            to_merge = {"agent_tools": {"manifest_file": manifest_file}}
        else:
            to_merge["agent_tools"].update({"manifest_file": manifest_file})

    request = QueryRequest(
        query=query,
        **get_merged_settings(
            to_merge,
            settings_path=pqa_directory("settings") / "settings.yaml",
        ),
    )

    return loop.run_until_complete(
        agent_query(
            request,
            docs=None,
            verbosity=verbosity,
            agent_type=agent_type,
            index_directory=request.agent_tools.index_directory,
        )
    )


@app.command("search")
def search_query(
    query: Annotated[str, typer.Argument(help=("Query for keyword search"))],
    index_name: Annotated[
        str,
        typer.Argument(
            help=(
                "Name of the index to search, or use `answers`"
                " to search all indexed answers"
            )
        ),
    ] = "answers",
    index_directory: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Index directory to store paper index and answers. Default will be `~/.pqa`"
            )
        ),
    ] = None,
) -> list[tuple[AnswerResponse, str] | tuple[Any, str]]:
    """Search using a pre-built PaperQA index."""
    configure_agent_logging(verbosity=0)

    loop = get_loop()
    return loop.run_until_complete(
        search(
            query,
            index_name=index_name,
            index_directory=index_directory or pqa_directory("indexes"),
        )
    )


@app.command("index")
def build_index(
    directory: Annotated[
        Path | None,
        typer.Argument(help=("Directory of papers or documents to run PaperQA over.")),
    ] = None,
    index_directory: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Index directory to store paper index and answers. Default will be `~/.pqa`"
            )
        ),
    ] = None,
    manifest_file: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Optional manifest file (CSV) location to map relative a "
                "`file_location` column to `doi` or `title` columns. "
                "If not used, then the file will be read by an LLM "
                "which attempts to extract the title, authors and DOI."
            )
        ),
    ] = None,
    verbosity: Annotated[
        int, typer.Option(help=("Level of verbosity from 0->2 (inclusive)"))
    ] = 0,
) -> SearchIndex:
    """Build a PaperQA search index, this will also happen automatically upon using `ask`."""
    configure_agent_logging(verbosity=verbosity)

    to_merge = {}

    if directory is not None:
        to_merge = {"agent_tools": {"paper_directory": directory}}

    if index_directory is not None:
        if "agent_tools" not in to_merge:
            to_merge = {"agent_tools": {"index_directory": index_directory}}
        else:
            to_merge["agent_tools"].update({"index_directory": index_directory})

    if manifest_file is not None:
        if "agent_tools" not in to_merge:
            to_merge = {"agent_tools": {"manifest_file": manifest_file}}
        else:
            to_merge["agent_tools"].update({"manifest_file": manifest_file})

    configure_agent_logging(verbosity)

    request_settings = QueryRequest(
        query="",
        **get_merged_settings(
            to_merge,
            settings_path=pqa_directory("settings") / "settings.yaml",
        ),
    )

    loop = get_loop()

    return loop.run_until_complete(
        get_directory_index(
            directory=anyio.Path(request_settings.agent_tools.paper_directory),
            index_directory=request_settings.agent_tools.index_directory,
            index_name=request_settings.get_index_name(
                request_settings.agent_tools.paper_directory,
                request_settings.embedding,
                request_settings.parsing_configuration,
            ),
            manifest_file=(
                anyio.Path(request_settings.agent_tools.manifest_file)
                if request_settings.agent_tools.manifest_file
                else None
            ),
            embedding=request_settings.embedding,
            chunk_chars=request_settings.parsing_configuration.chunksize,
            overlap=request_settings.parsing_configuration.overlap,
        )
    )


@app.command()
def version():
    configure_agent_logging(verbosity=0)
    logger.info(f"PaperQA version: {__version__}")


if __name__ == "__main__":
    app()
