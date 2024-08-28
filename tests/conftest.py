from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Generator
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


@pytest.fixture(name="agent_test_kit")
def fixture_stub_answer() -> Answer:
    return Answer(question="What is is a self-explanatory model?")


@pytest.fixture(name="stub_paper_path", scope="session")
def fixture_stub_paper_path() -> Path:
    # Corresponds with https://www.semanticscholar.org/paper/A-Perspective-on-Explanations-of-Molecular-Models-Wellawatte-Gandhi/1db1bde653658ec9b30858ae14650b8f9c9d438b
    return Path(__file__).parent / "paper.pdf"
