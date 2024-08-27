from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Generator

import pytest
from dotenv import load_dotenv

import paperqa
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
def tmp_path_cleanup(tmp_path: Path) -> Generator[Path, None, None]:
    yield tmp_path
    # Cleanup after the test
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def agent_module_dir(tmp_path_cleanup: Path, monkeypatch) -> Path:
    """Set up a unique temporary folder for the agent module."""
    monkeypatch.setenv("PQA_HOME", str(tmp_path_cleanup))
    importlib.reload(paperqa.agents)
    return tmp_path_cleanup


@pytest.fixture
def agent_index_dir(agent_module_dir: Path) -> Path:
    return agent_module_dir / ".pqa" / "indexes"


@pytest.fixture(name="agent_test_kit")
def fixture_agent_test_kit() -> Answer:
    return Answer(question="What is is a self-explanatory model?")


@pytest.fixture(name="stub_paper_path_with_details", scope="session")
def fixture_stub_paper_path_with_details() -> tuple[Path, dict]:
    # Corresponds with https://www.semanticscholar.org/paper/A-Perspective-on-Explanations-of-Molecular-Models-Wellawatte-Gandhi/1db1bde653658ec9b30858ae14650b8f9c9d438b
    return (
        Path(__file__).parent / "paper.pdf",
        {
            "paper_id": "042b619959c990c7",
            "url": "https://pubs.acs.org/doi/10.1021/acs.jctc.2c01235",
            "title": "A Perspective on Explanations of Molecular Prediction Models",
            "year": 2023,
            "key": "Wellawatte2023A",
            "doi": "10.1021/acs.jctc.2c01235",
        },
    )
