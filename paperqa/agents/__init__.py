from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from pydantic_settings import CliSettingsSource
from rich.console import Console
from rich.logging import RichHandler

from paperqa.settings import Settings
from paperqa.utils import get_loop, pqa_directory, setup_default_logs
from paperqa.version import __version__

from .main import agent_query, index_search
from .models import AnswerResponse, QueryRequest
from .search import SearchIndex, get_directory_index

logger = logging.getLogger(__name__)


def configure_cli_logging(verbosity: int = 0) -> None:
    """Suppress loquacious loggers according to verbosity level."""
    setup_default_logs()

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
            "litellm": logging.WARNING,
            "LiteLLM Router": logging.WARNING,
            "LiteLLM Proxy": logging.WARNING,
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
        "litellm": logging.INFO,
        "LiteLLM Router": logging.INFO,
        "LiteLLM Proxy": logging.INFO,
    }

    verbosity_map[3] = verbosity_map[2] | {
        "litellm": logging.DEBUG,  # <-- every single LLM call
    }

    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        show_level=False,
        console=Console(force_terminal=True),
    )

    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    module_logger = logging.getLogger(__name__.split(".", maxsplit=1)[0])

    if not any(isinstance(h, RichHandler) for h in module_logger.handlers):
        module_logger.addHandler(rich_handler)

    for logger_name, logger_ in logging.Logger.manager.loggerDict.items():
        if isinstance(logger_, logging.Logger) and (
            log_level := verbosity_map.get(min(verbosity, 2), {}).get(logger_name)
        ):
            logger_.setLevel(log_level)

    if verbosity > 0:
        print(f"PaperQA version: {__version__}")


def ask(query: str, settings: Settings) -> AnswerResponse:
    """Query PaperQA via an agent."""
    configure_cli_logging(verbosity=settings.verbosity)
    return get_loop().run_until_complete(
        agent_query(
            QueryRequest(query=query, settings=settings),
            agent_type=settings.agent.agent_type,
        )
    )


def search_query(
    query: str,
    index_name: str,
    settings: Settings,
) -> list[tuple[AnswerResponse, str] | tuple[Any, str]]:
    """Search using a pre-built PaperQA index."""
    configure_cli_logging(verbosity=settings.verbosity)
    if index_name == "default":
        index_name = settings.get_index_name()
    return get_loop().run_until_complete(
        index_search(
            query,
            index_name=index_name,
            index_directory=settings.index_directory,
        )
    )


def build_index(
    index_name: str,
    directory: str | os.PathLike,
    settings: Settings,
) -> SearchIndex:
    """Build a PaperQA search index, this will also happen automatically upon using `ask`."""
    if index_name == "default":
        index_name = settings.get_index_name()
    configure_cli_logging(verbosity=settings.verbosity)
    settings.paper_directory = directory
    return get_loop().run_until_complete(
        get_directory_index(index_name=index_name, settings=settings)
    )


def save_settings(
    settings: Settings,
    settings_path: str | os.PathLike,
) -> None:
    """Save the settings to a file."""
    configure_cli_logging(verbosity=settings.verbosity)
    # check if this could be interpreted at an absolute path
    if os.path.isabs(settings_path):
        full_settings_path = os.path.expanduser(settings_path)
    else:
        full_settings_path = os.path.join(pqa_directory("settings"), settings_path)
        if not full_settings_path.endswith(".json"):
            full_settings_path += ".json"

    is_overwrite = os.path.exists(full_settings_path)

    with open(full_settings_path, "w") as f:
        f.write(settings.model_dump_json(indent=2))

    if is_overwrite:
        logger.info(f"Settings overwritten to: {full_settings_path}")
    else:
        logger.info(f"Settings saved to: {full_settings_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperQA CLI")

    parser.add_argument(
        "--settings",
        "-s",
        default="high_quality",
        help=(
            "Named settings to use. Will search in local, pqa directory, and package"
            " last"
        ),
    )

    parser.add_argument(
        "--index", "-i", default="default", help="Index name to search or create"
    )

    subparsers = parser.add_subparsers(
        title="commands", dest="command", description="Available commands"
    )

    subparsers.add_parser("view", help="View the chosen settings")
    save_parser = subparsers.add_parser("save", help="View the chosen settings")
    save_parser.add_argument(
        "location", help="Location for new settings (name or an absolute path)"
    )

    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question of current index (based on settings)"
    )
    ask_parser.add_argument("query", help="Question to ask")

    search_parser = subparsers.add_parser(
        "search",
        help=(
            "Search the index specified by --index."
            " Pass `--index answers` to search previous answers."
        ),
    )
    search_parser.add_argument("query", help="Keyword search")

    build_parser = subparsers.add_parser(
        "index", help="Build a search index from given directory"
    )
    build_parser.add_argument("directory", help="Directory to build index from")

    # Create CliSettingsSource instance
    cli_settings = CliSettingsSource[argparse.ArgumentParser](
        Settings, root_parser=parser
    )

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
        case "save":
            save_settings(settings, args.location)
        case "search":
            search_query(args.query, args.index, settings)
        case "index":
            build_index(args.index, args.directory, settings)
        case _:
            commands = ", ".join({"view", "ask", "search", "index"})
            brief_help = f"\nRun with commands: {{{commands}}}\n\n"
            brief_help += "For more information, run with --help"
            print(brief_help)


if __name__ == "__main__":
    main()
