"""
Tests for borderless (three-line / booktabs-style) table detection.

Covers:
- Unit tests for each pure helper function (no real PDF needed).
- Integration tests using synthetic PDFs created with PyMuPDF itself.

Synthetic PDF layout (A4, 595 x 842 pt)
----------------------------------------
y=200  ───────────────────────  top border   (toprule)
       Strain   Titer (g/L)   Yield (%)     header row
y=215  ───────────────────────  separator    (midrule)
       E. coli       12.3        45.2        data row 1
       S. cerevi.     8.7        38.6        data row 2
       B. subtil.    15.1        51.0        data row 3
y=270  ───────────────────────  bottom border (bottomrule)
"""
from pathlib import Path

import pymupdf
import pytest

from paperqa_pymupdf.borderless_tables import (
    TABLE_CAPTION_RE,
    _assign_col,
    _build_text_lines,
    _caption_based_regions,
    _cluster_rules,
    _find_col_ranges,
    _get_wide_h_rules,
    _to_markdown,
    _words_to_grid,
    detect_borderless_tables,
    merge_multiline_cells,
)

# Minimal pixmap attribute set for testing (avoids importing reader internals).
_PIXMAP_ATTRS: frozenset[str] = frozenset({"width", "height", "n", "stride"})

# ── shared constants ───────────────────────────────────────────────────────────

_COL_POSITIONS = [85.0, 235.0, 385.0]
_HEADERS = ["Strain", "Titer (g/L)", "Yield (%)"]
_DATA = [
    ("E. coli", "12.3", "45.2"),
    ("S. cerevisiae", "8.7", "38.6"),
    ("B. subtilis", "15.1", "51.0"),
]
_RULE_YS = [200.0, 215.0, 270.0]


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def three_line_table_pdf(tmp_path: Path) -> Path:
    """A4 PDF with one borderless three-line table and no other graphics."""
    pdf = pymupdf.open()
    page = pdf.new_page(width=595, height=842)
    margin = 72.0
    page_w = 595.0

    for y in _RULE_YS:
        page.draw_line(
            pymupdf.Point(margin, y),
            pymupdf.Point(page_w - margin, y),
            color=(0, 0, 0),
            width=0.5,
        )

    for label, x in zip(_HEADERS, _COL_POSITIONS):
        page.insert_text(pymupdf.Point(x, 209.0), label, fontsize=10)

    for row_idx, row_data in enumerate(_DATA):
        y = 228.0 + row_idx * 14.0
        for x, cell in zip(_COL_POSITIONS, row_data):
            page.insert_text(pymupdf.Point(x, y), cell, fontsize=10)

    path = tmp_path / "three_line_table.pdf"
    pdf.save(str(path))
    return path


@pytest.fixture
def two_rule_table_pdf(tmp_path: Path) -> Path:
    """PDF where only the top and bottom rules are present (no midrule)."""
    pdf = pymupdf.open()
    page = pdf.new_page(width=595, height=842)
    margin = 72.0
    page_w = 595.0
    for y in (_RULE_YS[0], _RULE_YS[2]):
        page.draw_line(
            pymupdf.Point(margin, y),
            pymupdf.Point(page_w - margin, y),
            color=(0, 0, 0),
            width=0.5,
        )
    for label, x in zip(_HEADERS, _COL_POSITIONS):
        page.insert_text(pymupdf.Point(x, 209.0), label, fontsize=10)
    for row_idx, row_data in enumerate(_DATA):
        y = 228.0 + row_idx * 14.0
        for x, cell in zip(_COL_POSITIONS, row_data):
            page.insert_text(pymupdf.Point(x, y), cell, fontsize=10)
    path = tmp_path / "two_rule_table.pdf"
    pdf.save(str(path))
    return path


@pytest.fixture
def text_only_pdf(tmp_path: Path) -> Path:
    """PDF with no drawings — no tables should be detected."""
    pdf = pymupdf.open()
    page = pdf.new_page(width=595, height=842)
    page.insert_text(pymupdf.Point(72, 100), "Just some body text.", fontsize=12)
    path = tmp_path / "text_only.pdf"
    pdf.save(str(path))
    return path


@pytest.fixture
def caption_table_pdf(tmp_path: Path) -> Path:
    """PDF with a 'Table 1.' caption above a three-line table.

    Layout::

        y=150  "Table 1. Fermentation performance of engineered strains."
        y=200  ──────────────────────  toprule
               Strain   Titer   Yield  header
        y=215  ──────────────────────  midrule
               E. coli  12.3    45.2   data rows ...
        y=270  ──────────────────────  bottomrule
    """
    pdf = pymupdf.open()
    page = pdf.new_page(width=595, height=842)
    margin = 72.0
    page_w = 595.0

    page.insert_text(
        pymupdf.Point(margin, 150.0),
        "Table 1. Fermentation performance of engineered strains.",
        fontsize=10,
    )

    for y in _RULE_YS:
        page.draw_line(
            pymupdf.Point(margin, y),
            pymupdf.Point(page_w - margin, y),
            color=(0, 0, 0),
            width=0.5,
        )
    for label, x in zip(_HEADERS, _COL_POSITIONS):
        page.insert_text(pymupdf.Point(x, 209.0), label, fontsize=10)
    for row_idx, row_data in enumerate(_DATA):
        y = 228.0 + row_idx * 14.0
        for x, cell in zip(_COL_POSITIONS, row_data):
            page.insert_text(pymupdf.Point(x, y), cell, fontsize=10)

    path = tmp_path / "caption_table.pdf"
    pdf.save(str(path))
    return path


# ── unit tests: TABLE_CAPTION_RE ───────────────────────────────────────────────


class TestTableCaptionRE:

    def test_matches_english_table_n(self) -> None:
        assert TABLE_CAPTION_RE.search("Table 1. Summary of results")
        assert TABLE_CAPTION_RE.search("Table 2: Comparison")
        assert TABLE_CAPTION_RE.search("table 3 properties")

    def test_matches_tab_abbreviation(self) -> None:
        assert TABLE_CAPTION_RE.search("Tab. 4 experimental conditions")

    def test_matches_supplementary(self) -> None:
        assert TABLE_CAPTION_RE.search("Supplementary Table S1 raw data")

    def test_matches_chinese_caption(self) -> None:
        assert TABLE_CAPTION_RE.search("表1 发酵实验结果")
        assert TABLE_CAPTION_RE.search("附表2 补充数据")

    def test_does_not_match_plain_prose(self) -> None:
        assert not TABLE_CAPTION_RE.search("This table shows the results")
        assert not TABLE_CAPTION_RE.search("As shown in the following table")


# ── unit tests: _find_col_ranges ───────────────────────────────────────────────


class TestFindColRanges:

    def test_three_well_separated_columns(self) -> None:
        words = [
            (10.0, 0.0, 60.0, 10.0, "A", 0, 0, 0),
            (120.0, 0.0, 180.0, 10.0, "B", 0, 0, 1),
            (260.0, 0.0, 310.0, 10.0, "C", 0, 0, 2),
        ]
        cols = _find_col_ranges(words, min_gap_pt=20.0)
        assert len(cols) == 3
        assert cols[0][0] < cols[1][0] < cols[2][0]

    def test_adjacent_words_merge_into_one_column(self) -> None:
        words = [
            (10.0, 0.0, 60.0, 10.0, "hello", 0, 0, 0),
            (65.0, 0.0, 110.0, 10.0, "world", 0, 0, 1),
        ]
        cols = _find_col_ranges(words, min_gap_pt=20.0)
        assert len(cols) == 1

    def test_empty_word_list_returns_empty(self) -> None:
        assert _find_col_ranges([]) == []

    def test_gap_exactly_at_threshold_creates_new_column(self) -> None:
        words = [
            (0.0, 0.0, 50.0, 10.0, "A", 0, 0, 0),
            (70.0, 0.0, 100.0, 10.0, "B", 0, 0, 1),
        ]
        cols = _find_col_ranges(words, min_gap_pt=20.0)
        assert len(cols) == 2

    def test_ranges_are_non_overlapping_and_sorted(self) -> None:
        words = [
            (5.0, 0.0, 40.0, 10.0, "w1", 0, 0, 0),
            (100.0, 0.0, 150.0, 10.0, "w2", 0, 0, 1),
            (200.0, 0.0, 250.0, 10.0, "w3", 0, 0, 2),
        ]
        cols = _find_col_ranges(words, min_gap_pt=10.0)
        assert all(cols[i][1] < cols[i + 1][0] for i in range(len(cols) - 1))


# ── unit tests: _assign_col ────────────────────────────────────────────────────


class TestAssignCol:

    COL_RANGES = [(10.0, 60.0), (120.0, 180.0), (250.0, 310.0)]

    def test_word_centre_in_first_column(self) -> None:
        word = (15.0, 0.0, 55.0, 10.0, "x", 0, 0, 0)
        assert _assign_col(word, self.COL_RANGES) == 0

    def test_word_centre_in_middle_column(self) -> None:
        word = (130.0, 0.0, 170.0, 10.0, "x", 0, 0, 0)
        assert _assign_col(word, self.COL_RANGES) == 1

    def test_word_centre_in_last_column(self) -> None:
        word = (260.0, 0.0, 300.0, 10.0, "x", 0, 0, 0)
        assert _assign_col(word, self.COL_RANGES) == 2

    def test_word_outside_all_columns_snaps_to_nearest(self) -> None:
        word = (370.0, 0.0, 410.0, 10.0, "x", 0, 0, 0)
        assert _assign_col(word, self.COL_RANGES) == 2


# ── unit tests: _words_to_grid ─────────────────────────────────────────────────


class TestWordsToGrid:

    COL_RANGES = [(10.0, 60.0), (120.0, 180.0), (250.0, 310.0)]

    def test_single_row(self) -> None:
        words = [
            (15.0, 0.0, 55.0, 10.0, "A", 0, 0, 0),
            (130.0, 0.0, 175.0, 10.0, "B", 0, 0, 1),
            (260.0, 0.0, 300.0, 10.0, "C", 0, 0, 2),
        ]
        grid = _words_to_grid(words, self.COL_RANGES, row_gap_pt=6.0)
        assert grid == [["A", "B", "C"]]

    def test_two_rows_separated_by_gap(self) -> None:
        words = [
            (15.0, 0.0, 55.0, 10.0, "A", 0, 0, 0),
            (130.0, 0.0, 175.0, 10.0, "B", 0, 0, 1),
            (260.0, 0.0, 300.0, 10.0, "C", 0, 0, 2),
            (15.0, 20.0, 55.0, 30.0, "D", 0, 1, 0),
            (130.0, 20.0, 175.0, 30.0, "E", 0, 1, 1),
            (260.0, 20.0, 300.0, 30.0, "F", 0, 1, 2),
        ]
        grid = _words_to_grid(words, self.COL_RANGES, row_gap_pt=6.0)
        assert len(grid) == 2
        assert grid[0] == ["A", "B", "C"]
        assert grid[1] == ["D", "E", "F"]

    def test_two_words_close_in_y_same_row(self) -> None:
        words = [
            (15.0, 0.0, 55.0, 10.0, "A", 0, 0, 0),
            (130.0, 3.0, 175.0, 13.0, "B", 0, 0, 1),
        ]
        grid = _words_to_grid(words, self.COL_RANGES, row_gap_pt=6.0)
        assert len(grid) == 1

    def test_empty_input_returns_empty(self) -> None:
        assert _words_to_grid([], self.COL_RANGES) == []


# ── unit tests: _cluster_rules ─────────────────────────────────────────────────


class TestClusterRules:

    def test_three_close_rules_form_one_cluster(self) -> None:
        rules = [(100.0, 50.0, 500.0), (115.0, 50.0, 500.0), (250.0, 50.0, 500.0)]
        clusters = _cluster_rules(rules)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_rules_far_apart_form_separate_clusters(self) -> None:
        rules = [
            (100.0, 50.0, 500.0),
            (210.0, 50.0, 500.0),
            (700.0, 50.0, 500.0),
            (800.0, 50.0, 500.0),
        ]
        clusters = _cluster_rules(rules)
        assert len(clusters) == 2
        assert len(clusters[0]) == 2
        assert len(clusters[1]) == 2

    def test_single_rule_filtered_out(self) -> None:
        assert _cluster_rules([(100.0, 50.0, 500.0)]) == []

    def test_empty_input_returns_empty(self) -> None:
        assert _cluster_rules([]) == []


# ── unit tests: _to_markdown ───────────────────────────────────────────────────


class TestToMarkdown:

    def test_basic_two_column_table(self) -> None:
        md = _to_markdown(["Name", "Score"], [["Alice", "95"], ["Bob", "87"]])
        assert "| Name | Score |" in md
        assert "| --- | --- |" in md
        assert "| Alice | 95 |" in md
        assert "| Bob | 87 |" in md

    def test_pipe_in_cell_is_escaped(self) -> None:
        md = _to_markdown(["Col"], [["a|b"]])
        assert r"a\|b" in md

    def test_header_and_data_row_order(self) -> None:
        md = _to_markdown(["H"], [["row1"], ["row2"]])
        lines = md.splitlines()
        assert lines[0].startswith("| H ")
        assert lines[1].startswith("| ---")
        assert "row1" in lines[2]
        assert "row2" in lines[3]


# ── unit tests: merge_multiline_cells ─────────────────────────────────────────


class TestMergeMultilineCells:

    def test_simple_continuation_merged(self) -> None:
        rows = [
            ["E. coli", "12.3", "45.2"],
            ["", "K1 strain", ""],
        ]
        merged = merge_multiline_cells(rows)
        assert len(merged) == 1
        assert merged[0][0] == "E. coli"
        assert "K1 strain" in merged[0][1]

    def test_independent_rows_not_merged(self) -> None:
        rows = [
            ["E. coli", "12.3", "45.2"],
            ["S. cerevi.", "8.7", "38.6"],
        ]
        merged = merge_multiline_cells(rows)
        assert len(merged) == 2

    def test_empty_input_returns_empty(self) -> None:
        assert merge_multiline_cells([]) == []

    def test_single_row_unchanged(self) -> None:
        rows = [["A", "B", "C"]]
        assert merge_multiline_cells(rows) == [["A", "B", "C"]]


# ── integration tests: _get_wide_h_rules ──────────────────────────────────────


class TestGetWideHRules:

    def test_detects_three_rules_in_three_line_pdf(
        self, three_line_table_pdf: Path
    ) -> None:
        with pymupdf.open(str(three_line_table_pdf)) as doc:
            page = doc[0]
            rules = _get_wide_h_rules(page, float(page.rect.width))

        assert len(rules) == 3, f"Expected 3 rules, got {len(rules)}: {rules}"
        y_coords = [r[0] for r in rules]
        for detected, expected in zip(sorted(y_coords), sorted(_RULE_YS)):
            assert abs(detected - expected) < 5.0

    def test_no_rules_on_text_only_page(self, text_only_pdf: Path) -> None:
        with pymupdf.open(str(text_only_pdf)) as doc:
            page = doc[0]
            rules = _get_wide_h_rules(page, float(page.rect.width))
        assert rules == []

    def test_detects_two_rules_in_two_rule_pdf(
        self, two_rule_table_pdf: Path
    ) -> None:
        with pymupdf.open(str(two_rule_table_pdf)) as doc:
            page = doc[0]
            rules = _get_wide_h_rules(page, float(page.rect.width))
        assert len(rules) == 2


# ── integration tests: caption-based detection ────────────────────────────────


class TestCaptionBasedRegions:

    def test_caption_anchors_table_region(self, caption_table_pdf: Path) -> None:
        """Caption text triggers a candidate bbox that covers the table below it."""
        with pymupdf.open(str(caption_table_pdf)) as doc:
            page = doc[0]
            page_w = float(page.rect.width)
            page_h = float(page.rect.height)
            words = page.get_text("words") or []
            lines = _build_text_lines(words)
            rules = _get_wide_h_rules(page, page_w)

        candidates = _caption_based_regions(lines, rules, page_w, page_h)
        assert candidates, "Caption-based detection should produce at least one candidate"
        table_y_min, table_y_max = _RULE_YS[0], _RULE_YS[-1]
        found = any(
            bbox[1] <= table_y_min + 30 and bbox[3] >= table_y_max - 10
            for bbox in candidates
        )
        assert found, (
            f"No candidate covers y=[{table_y_min}, {table_y_max}]; got: {candidates}"
        )

    def test_no_caption_no_candidate(self, three_line_table_pdf: Path) -> None:
        """When there is no caption text, caption strategy returns no candidates."""
        with pymupdf.open(str(three_line_table_pdf)) as doc:
            page = doc[0]
            page_w = float(page.rect.width)
            page_h = float(page.rect.height)
            words = page.get_text("words") or []
            lines = _build_text_lines(words)
            rules = _get_wide_h_rules(page, page_w)

        candidates = _caption_based_regions(lines, rules, page_w, page_h)
        assert candidates == [], (
            "three_line_table_pdf has no caption text, so caption strategy must return []"
        )


# ── integration tests: detect_borderless_tables ────────────────────────────────


class TestDetectBorderlessTables:

    def test_full_three_rule_table(self, three_line_table_pdf: Path) -> None:
        with pymupdf.open(str(three_line_table_pdf)) as doc:
            page = doc[0]
            results = detect_borderless_tables(
                page,
                page_num=0,
                page_width=float(page.rect.width),
                dpi=None,
                pymupdf_pixmap_attrs=_PIXMAP_ATTRS,
            )

        assert len(results) == 1, "Expected exactly one borderless table"
        (table,) = results
        assert table.info["type"] == "table"
        assert table.info.get("detection_method") == "borderless"
        assert table.info["page_num"] == 1
        assert isinstance(table.data, bytes) and table.data

        assert table.text, "Expected non-empty markdown text"
        assert "Strain" in table.text
        assert any(cell in table.text for cell in ("12.3", "8.7", "15.1"))
        assert "|" in table.text

    def test_two_rule_table_also_detected(self, two_rule_table_pdf: Path) -> None:
        with pymupdf.open(str(two_rule_table_pdf)) as doc:
            page = doc[0]
            results = detect_borderless_tables(
                page,
                page_num=0,
                page_width=float(page.rect.width),
                dpi=None,
                pymupdf_pixmap_attrs=_PIXMAP_ATTRS,
            )
        assert len(results) == 1, "Should detect a table even with only 2 horizontal rules"

    def test_no_results_on_text_only_page(self, text_only_pdf: Path) -> None:
        with pymupdf.open(str(text_only_pdf)) as doc:
            page = doc[0]
            results = detect_borderless_tables(
                page,
                page_num=0,
                page_width=float(page.rect.width),
                dpi=None,
                pymupdf_pixmap_attrs=_PIXMAP_ATTRS,
            )
        assert results == []

    def test_already_detected_bbox_suppresses_duplicate(
        self, three_line_table_pdf: Path
    ) -> None:
        with pymupdf.open(str(three_line_table_pdf)) as doc:
            page = doc[0]
            page_w = float(page.rect.width)
            first = detect_borderless_tables(
                page, page_num=0, page_width=page_w, dpi=None,
                pymupdf_pixmap_attrs=_PIXMAP_ATTRS,
            )
            assert first, "Precondition: table must be detected in the first pass"
            already = [tuple(first[0].info["bbox"])]  # type: ignore[misc]
            second = detect_borderless_tables(
                page, page_num=0, page_width=page_w, dpi=None,
                pymupdf_pixmap_attrs=_PIXMAP_ATTRS,
                already_detected_bboxes=already,
            )
        assert second == [], "Duplicate region must be suppressed"

    def test_media_info_is_json_serializable(self, three_line_table_pdf: Path) -> None:
        import json as _json

        with pymupdf.open(str(three_line_table_pdf)) as doc:
            page = doc[0]
            results = detect_borderless_tables(
                page, page_num=0, page_width=float(page.rect.width),
                dpi=None, pymupdf_pixmap_attrs=_PIXMAP_ATTRS,
            )
        assert results
        _json.dumps(results[0].info)  # must not raise

    def test_caption_pdf_detects_table(self, caption_table_pdf: Path) -> None:
        """Caption + rules PDF must yield one table via the caption strategy."""
        with pymupdf.open(str(caption_table_pdf)) as doc:
            page = doc[0]
            results = detect_borderless_tables(
                page,
                page_num=0,
                page_width=float(page.rect.width),
                dpi=None,
                pymupdf_pixmap_attrs=_PIXMAP_ATTRS,
            )
        assert len(results) >= 1
        assert any(m.info.get("detection_method") == "borderless" for m in results)
        assert any("Strain" in m.text for m in results)


# ── end-to-end test: parse_pdf_to_pages integration ───────────────────────────


def test_parse_pdf_to_pages_detects_borderless_table(
    three_line_table_pdf: Path,
) -> None:
    """The full parse_pdf_to_pages pipeline must emit the borderless table."""
    from paperqa_pymupdf import parse_pdf_to_pages

    result = parse_pdf_to_pages(str(three_line_table_pdf), parse_media=True)
    assert "1" in result.content
    assert isinstance(result.content["1"], tuple)
    _, media = result.content["1"]

    tables = [m for m in media if m.info.get("type") == "table"]
    assert len(tables) == 1, f"Expected 1 table in the parsed output, got {len(tables)}"
    (table,) = tables
    assert table.info.get("detection_method") == "borderless"
    assert "Strain" in table.text
    assert table.data


def test_parse_pdf_to_pages_no_tables_on_text_only_page(
    text_only_pdf: Path,
) -> None:
    from paperqa_pymupdf import parse_pdf_to_pages

    result = parse_pdf_to_pages(str(text_only_pdf), parse_media=True)
    if "1" in result.content and isinstance(result.content["1"], tuple):
        _, media = result.content["1"]
        tables = [m for m in media if m.info.get("type") == "table"]
        assert tables == [], "No tables should be detected on a text-only page"
