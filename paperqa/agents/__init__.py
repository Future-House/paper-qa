from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
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

    module_logger = logging.getLogger("paperqa")

    if not any(isinstance(h, RichHandler) for h in module_logger.handlers):
        module_logger.addHandler(rich_handler)

    for logger_name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and (
            log_level := verbosity_map.get(min(verbosity, 2), {}).get(logger_name)
        ):
            logger.setLevel(log_level)

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
        "--settings",
        "-s",
        default="default",
        help="Named settings to use. Will search in local, pqa directory, and package last",
    )

    subparsers = parser.add_subparsers(
        title="commands", dest="command", description="Available commands"
    )

    # Show command
    subparsers.add_parser("view", help="View the chosen settings")

    # Create CliSettingsSource instance
    cli_settings = CliSettingsSource(Settings, root_parser=parser)

    # Now use argparse to parse the remaining arguments
    args, remaining_args = parser.parse_known_args()
    # Parse arguments using CliSettingsSource
    settings = Settings.from_name(
        args.settings, cli_source=cli_settings(args=remaining_args)
    )

    match args.command:
        case "ask":
            ask(args.query, settings)
        case "view":
            configure_cli_logging(settings.verbosity)
            logger.info(f"Viewing: {args.settings}")
            logger.info(settings.model_dump_json(indent=2))
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
