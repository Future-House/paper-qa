import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True, scope="session")
def _load_env():
    load_dotenv()


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["Crossref-Plus-API-Token", "x-api-key"],
    }
