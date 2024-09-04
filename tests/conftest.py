from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Generator, Iterator
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

from paperqa.clients.crossref import CROSSREF_HEADER_KEY
from paperqa.clients.semantic_scholar import SEMANTIC_SCHOLAR_HEADER_KEY
from paperqa.types import Answer


@pytest.fixture(autouse=True, scope="session")
def _load_env():
    load_dotenv()


@pytest.fixture(scope="session")
def vcr_config():
    return {
        "filter_headers": [CROSSREF_HEADER_KEY, SEMANTIC_SCHOLAR_HEADER_KEY],
        "record_mode": "none",
        "match_on": ["method", "host", "path", "query"],
        "allow_playback_repeats": True,
        "cassette_library_dir": "tests/cassettes",
    }


@pytest.fixture
def tmp_path_cleanup(
    tmp_path: Path,
) -> Generator[Path, None, None]:
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


@pytest.fixture
def agent_stub_answer() -> Answer:
    return Answer(question="What is is a self-explanatory model?")


@pytest.fixture
def stub_data_dir() -> Path:
    return Path(__file__).parent / "stub_data"


@pytest.fixture
def stub_data_dir_w_near_dupes(stub_data_dir: Path, tmp_path: Path) -> Iterator[Path]:

    # add some near duplicate files then removes them after testing
    for filename in ("example.txt", "example2.txt"):
        if not (tmp_path / f"{filename}_modified.txt").exists():
            with open(stub_data_dir / filename) as f:
                content = f.read()
            with open(tmp_path / f"{Path(filename).stem}_modified.txt", "w") as f:
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
