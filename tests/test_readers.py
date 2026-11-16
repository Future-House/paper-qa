from pathlib import Path

import pytest

from paperqa.readers import parse_text
from paperqa.utils import ImpossibleParsingError

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("line_count", [1, 5, 6])
def test_parse_text_page_size_limit_accepts_short_lines(
    tmp_path: Path, line_count: int
) -> None:
    path = tmp_path / "document.md"
    path.write_text("text\n" * line_count)

    parsed = parse_text(path, split_lines=True, page_size_limit=5)

    assert parsed.content == ["text\n"] * line_count
    assert parsed.metadata.total_parsed_text_length == 5 * line_count


@pytest.mark.parametrize("line_count", [1, 5, 6])
def test_parse_text_page_size_limit_rejects_long_line(
    tmp_path: Path, line_count: int
) -> None:
    path = tmp_path / "document.md"
    path.write_text("text\n" * (line_count - 1) + "longer\n")

    with pytest.raises(ImpossibleParsingError, match="7 chars long"):
        parse_text(path, split_lines=True, page_size_limit=5)


@pytest.mark.parametrize("page_size_limit", [None, 0])
def test_parse_text_page_size_limit_can_be_disabled(
    tmp_path: Path, page_size_limit: int | None
) -> None:
    path = tmp_path / "document.md"
    path.write_text("longer\n")

    parsed = parse_text(path, split_lines=True, page_size_limit=page_size_limit)

    assert parsed.content == ["longer\n"]
