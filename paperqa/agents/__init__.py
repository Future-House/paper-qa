from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_settings import CliSettingsSource

from .. import __version__
from ..config import Settings
from ..utils import get_loop

try:
    from rich.console import Console
    from rich.logging import RichHandler

    from .main import agent_query, search
    from .models import AnswerResponse, QueryRequest
    from .search import SearchIndex, get_directory_index

except ImportError as e:
    raise ImportError(
        '"agents" module is not installed please install it using "pip install paper-qa[agents]"'
    ) from e

logger = logging.getLogger(__name__)


def configure_cli_logging(verbosity: int = 0) -> None:
    """Suppress loquacious loggers according to verbosity level."""
    verbosity_map = {
        0: {
            "paperqa.agents": logging.INFO,
            "paperqa.agents.helpers": logging.WARNING,
            "paperqa.agents.main": logging.WARNING,
            "paperqa.agents.main.agent_callers": logging.INFO,
            "anthropic": logging.WARNING,
            "openai": logging.WARNING,
            "httpx": logging.WARNING,
            "paperqa.agents.models": logging.WARNING,
            "paperqa.agents.search": logging.INFO,
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

    for logger_name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and (
            log_level := verbosity_map.get(min(verbosity, 2), {}).get(logger_name)
        ):
            logger.setLevel(log_level)
            if not any(isinstance(h, RichHandler) for h in logger.handlers):
                logger.addHandler(rich_handler)

    if verbosity > 0:
        print(f"PaperQA version: {__version__}")


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


def ask(query: str, settings: Settings) -> AnswerResponse:
    """Query PaperQA via an agent."""
    configure_cli_logging(verbosity=settings.verbosity)

    loop = get_loop()

    request = QueryRequest(
        query=query,
        settings=settings,
    )

    return loop.run_until_complete(
        agent_query(
            request,
            docs=None,
            verbosity=settings.verbosity,
            agent_type=settings.agent.agent_type,
        )
    )


def search_query(
    query: str,
    index_name: str,
    settings: Settings,
) -> list[tuple[AnswerResponse, str] | tuple[Any, str]]:
    """Search using a pre-built PaperQA index."""
    configure_cli_logging(verbosity=0)
    loop = get_loop()
    return loop.run_until_complete(
        search(
            query,
            index_name=index_name,
            index_directory=settings.index_directory,
        )
    )


def build_index(
    settings: Settings,
) -> SearchIndex:
    """Build a PaperQA search index, this will also happen automatically upon using `ask`."""
    configure_cli_logging(verbosity=settings.verbosity)
    loop = get_loop()

    return loop.run_until_complete(get_directory_index(settings=settings))


def main():
    parser = argparse.ArgumentParser(description="PaperQA CLI")

    parser.add_argument(
        "--version", "-v", action="version", version=f"PaperQA version: {__version__}"
    )

    parser.add_argument(
        "--settings",
        "-s",
        type=str,
        help="Name or path of settings file",
        default="default",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Set command
    set_parser = subparsers.add_parser("set", help="Set a persistent PaperQA setting")
    set_parser.add_argument("variable", help="PaperQA variable to set")
    set_parser.add_argument("value", help="Value to set to the variable")

    # Show command
    view_parser = subparsers.add_parser("view", help="View the chosen settings")
    view_parser.add_argument("query", help="Question or task to ask of PaperQA")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Query PaperQA via an agent")
    ask_parser.add_argument("query", help="Question or task to ask of PaperQA")
    ask_parser.add_argument("--agent-type", default="fake", help="Type of agent to use")
    ask_parser.add_argument(
        "--verbosity", type=int, default=0, help="Level of verbosity (0-2)"
    )
    ask_parser.add_argument(
        "--directory", type=Path, help="Directory of papers or documents"
    )
    ask_parser.add_argument(
        "--index-directory",
        type=Path,
        help="Index directory to store paper index and answers",
    )
    ask_parser.add_argument(
        "--manifest-file", type=Path, help="Optional manifest file (CSV) location"
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search using a pre-built PaperQA index"
    )
    search_parser.add_argument("query", help="Query for keyword search")
    search_parser.add_argument("index_name", help="Name of the index to search")

    # Index command
    index_parser = subparsers.add_parser("index", help="Build a PaperQA search index")

    cli_settings = CliSettingsSource(Settings, root_parser=parser)

    print("ABOUT TO PARSE")
    args = parser.parse_args()

    settings = Settings.from_name(args.settings, cli_source=cli_settings)

    print(args.command)

    match args.command:
        case "ask":
            ask(args.query, settings)
        case "view":
            logger.info("Viewing settings")
            logger.info(settings)
        case "search":
            search_query(args.query, args.index_name, settings)
        case "index":
            build_index(args.verbosity)
        case _:
            configure_cli_logging(verbosity=1)
            commands = ", ".join({"view", "ask", "search", "index"})
            brief_help = f"\nRun with commands: {{{commands}}}\n\n"
            brief_help += "For more information, run with --help"
            print(brief_help)


if __name__ == "__main__":
    main()
