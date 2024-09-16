from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

from paperqa.clients.crossref import CROSSREF_HEADER_KEY
from paperqa.clients.semantic_scholar import SEMANTIC_SCHOLAR_HEADER_KEY
from paperqa.settings import Settings
from paperqa.types import Answer
from paperqa.utils import setup_default_logs

PAPER_DIRECTORY = Path(__file__).parent


@pytest.fixture(autouse=True, scope="session")
def _load_env() -> None:
    load_dotenv()


@pytest.fixture(autouse=True)
def _setup_default_logs() -> None:
    setup_default_logs()


@pytest.fixture(scope="session", name="vcr_config")
def fixture_vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [CROSSREF_HEADER_KEY, SEMANTIC_SCHOLAR_HEADER_KEY],
        "match_on": ["method", "host", "path", "query"],
        "allow_playback_repeats": True,
        "cassette_library_dir": "tests/cassettes",
    }


@pytest.fixture
def tmp_path_cleanup(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path
    # Cleanup after the test
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def agent_home_dir(
    tmp_path_cleanup: str | os.PathLike,
) -> Generator[str | os.PathLike, None, None]:
    """Set up a unique temporary folder for the agent module."""
    with patch.dict("os.environ", {"PQA_HOME": str(tmp_path_cleanup)}):
        yield tmp_path_cleanup


@pytest.fixture
def agent_index_dir(agent_home_dir: Path) -> Path:
    return agent_home_dir / ".pqa" / "indexes"


@pytest.fixture(scope="session", name="stub_data_dir")
def fixture_stub_data_dir() -> Path:
    return Path(__file__).parent / "stub_data"


@pytest.fixture
def agent_test_settings(agent_index_dir: Path, stub_data_dir: Path) -> Settings:
    settings = Settings(
        paper_directory=stub_data_dir,
        index_directory=agent_index_dir,
        embedding="sparse",
    )
    settings.agent.search_count = 2
    settings.answer.answer_max_sources = 2
    settings.answer.evidence_k = 10
    return settings


@pytest.fixture
def agent_stub_answer() -> Answer:
    return Answer(question="What is is a self-explanatory model?")


@pytest.fixture
def stub_data_dir_w_near_dupes(stub_data_dir: Path, tmp_path: Path) -> Iterator[Path]:

    # add some near duplicate files then removes them after testing
    for filename in ("bates.txt", "obama.txt"):
        if not (tmp_path / f"{filename}_modified.txt").exists():
            with (stub_data_dir / filename).open() as f:
                content = f.read()
            with (tmp_path / f"{Path(filename).stem}_modified.txt").open("w") as f:
                f.write(content)
                f.write("## MODIFIED FOR HASH")

    yield tmp_path

    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def reset_log_levels(caplog):
    logging.getLogger().setLevel(logging.DEBUG)

    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True

    caplog.set_level(logging.DEBUG)

    yield

    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
