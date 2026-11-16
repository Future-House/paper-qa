from pathlib import Path
from unittest.mock import patch

import pytest

from paperqa.agents.search import get_directory_index
from paperqa.docs import Docs
from paperqa.settings import Settings

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
@pytest.mark.parametrize("use_absolute_paper_directory", [False, True])
async def test_directory_index_preserves_unchanged_files(
    tmp_path: Path, use_absolute_paper_directory: bool
) -> None:
    paper_dir = tmp_path / "papers"
    paper_dir.mkdir()
    paper_path = paper_dir / "paper.txt"
    paper_path.write_text(
        "This local document describes how a search index preserves unchanged files."
        " Reusing parsed text avoids unnecessary work during repeated queries.\n"
    )
    (paper_dir / "manifest.csv").write_text(
        "file_location,title,year\npaper.txt,Offline cache example,2025\n"
    )
    settings = Settings(
        parsing={"use_doc_details": False, "defer_embedding": True},
        agent={
            "index": {
                "name": "unchanged-files",
                "paper_directory": paper_dir,
                "index_directory": tmp_path / "indexes",
                "manifest_file": "manifest.csv",
                "use_absolute_paper_directory": use_absolute_paper_directory,
            }
        },
    )
    first_index = await get_directory_index(settings)
    expected_location = str(paper_path) if use_absolute_paper_directory else "paper.txt"
    first_files = dict(await first_index.index_files)
    assert set(first_files) == {expected_location}
    first_document = await first_index.get_saved_object(expected_location)
    assert isinstance(first_document, Docs)

    with patch.object(Docs, "aadd", autospec=True, side_effect=Docs.aadd) as mock_aadd:
        second_index = await get_directory_index(settings)

    mock_aadd.assert_not_awaited()
    assert await second_index.index_files == first_files
    second_document = await second_index.get_saved_object(expected_location)
    assert isinstance(second_document, Docs)
    assert second_document.model_dump() == first_document.model_dump()
