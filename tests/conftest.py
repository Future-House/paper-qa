import pytest
from dotenv import load_dotenv

from paperqa.clients.crossref import CROSSREF_HEADER_KEY
from paperqa.clients.semantic_scholar import SEMANTIC_SCHOLAR_HEADER_KEY


@pytest.fixture(autouse=True, scope="session")
def _load_env():
    load_dotenv()


@pytest.fixture(scope="session")
def vcr_config():
    return {
        "filter_headers": [CROSSREF_HEADER_KEY, SEMANTIC_SCHOLAR_HEADER_KEY],
    }
