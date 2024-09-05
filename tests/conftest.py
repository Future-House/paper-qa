from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

from paperqa.clients.crossref import CROSSREF_HEADER_KEY
from paperqa.clients.semantic_scholar import SEMANTIC_SCHOLAR_HEADER_KEY
from paperqa.config import Settings
from paperqa.types import Answer

PAPER_DIRECTORY = Path(__file__).parent


@pytest.fixture(autouse=True, scope="session")
def _load_env():
    load_dotenv()


@pytest.fixture(scope="session")
def vcr_config():
    return {
        "filter_headers": [CROSSREF_HEADER_KEY, SEMANTIC_SCHOLAR_HEADER_KEY],
    }


@pytest.fixture
def bates_fixture():
    url = "https://en.wikipedia.org/wiki/Frederick_Bates_(politician)"
    with urllib.request.urlopen(url) as response:
        html = response.read()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(html)
        f.flush()
    return f.name


@pytest.fixture
def flag_day_fixture():
    url = "https://en.wikipedia.org/wiki/National_Flag_of_Canada_Day"
    with urllib.request.urlopen(url) as response:
        html = response.read()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(html)
        f.flush()
    return f.name


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
    bates_fixture: os.PathLike,
    flag_day_fixture: os.PathLike,
) -> Generator[str | os.PathLike, None, None]:
    """Set up a unique temporary folder for the agent module."""
    # download necessary files if not present
    # tests are written to assume files are present in tests
    tests_dir = Path(__file__).parent
    with open(tests_dir / "bates.html", "w") as f, open(bates_fixture) as bates_f:
        f.write(bates_f.read())
    with (
        open(tests_dir / "flag_day.html", "w") as f,
        open(flag_day_fixture) as flag_day_f,
    ):
        f.write(flag_day_f.read())
    with patch.dict("os.environ", {"PQA_HOME": str(tmp_path_cleanup)}):
        yield tmp_path_cleanup


@pytest.fixture
def agent_index_dir(agent_home_dir: Path, bates_fixture, flag_day_fixture) -> Path:
    return agent_home_dir / ".pqa" / "indexes"


@pytest.fixture
def agent_test_settings(agent_index_dir: Path) -> Settings:
    settings = Settings()
    settings.agent.paper_directory = PAPER_DIRECTORY
    settings.agent.index_directory = agent_index_dir
    settings.agent.search_count = 2
    settings.embedding = "sparse"
    settings.answer.answer_max_sources = 2
    settings.answer.evidence_k = 10
    return settings


@pytest.fixture(name="agent_test_kit")
def fixture_stub_answer() -> Answer:
    return Answer(question="What is is a self-explanatory model?")


@pytest.fixture(name="stub_paper_path", scope="session")
def fixture_stub_paper_path() -> Path:
    # Corresponds with https://www.semanticscholar.org/paper/A-Perspective-on-Explanations-of-Molecular-Models-Wellawatte-Gandhi/1db1bde653658ec9b30858ae14650b8f9c9d438b
    return Path(__file__).parent / "paper.pdf"
