"""
Detection and text extraction of borderless academic tables.

"Borderless" (aka *three-line* or *booktabs*-style) tables are common in
LaTeX-compiled academic papers.  They use only three full-width horizontal
rules (\\toprule, \\midrule, \\bottomrule) with no vertical separators between
cells.  PyMuPDF's default ``find_tables()`` misses them because it requires
cell-border intersections to locate column boundaries.

This module supplements ``find_tables()`` with **four complementary detection
strategies**, then scores each candidate by multi-evidence confidence:

1. **Caption anchoring** (primary) — searches for "Table N" / "表N" text and
   extends a bbox downward to encompass the table body.
2. **Wide horizontal-rule clustering** (secondary) — groups drawing-layer rules
   into candidate table bands (the classic booktabs approach).
3. **Text-alignment runs** (fallback) — detects tables by repeated word
   x-position bins across consecutive text lines, even when no rules are drawn.
4. **Rotated-text detection** (special case) — finds 90°-rotated tables using
   PyMuPDF's character direction vectors from ``get_text("rawdict")``.

Ported from PaperSort's ``TableRegionDetector`` + ``extract_threeline`` with
all pdfplumber-specific APIs replaced by PyMuPDF equivalents.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Any

from paperqa.types import ParsedMedia
from paperqa.utils import clean_invalid_unicode

if TYPE_CHECKING:
    import pymupdf

logger = logging.getLogger(__name__)

# ── tunable constants ──────────────────────────────────────────────────────────
# A horizontal rule must span at least this fraction of the page width.
_MIN_LINE_WIDTH_RATIO: float = 0.20
# A path bounding rect taller than this (points) is not a ruled line.
_MAX_RULE_HEIGHT_PT: float = 3.0
# Maximum vertical gap (points) between two rules to be in the same table.
_LINE_CLUSTER_GAP_PT: float = 200.0
# A cluster needs at least this many rules to qualify as a table.
_MIN_LINES_PER_CLUSTER: int = 2
# Minimum top-to-bottom span (points) for a table to be kept.
_MIN_TABLE_HEIGHT_PT: float = 20.0
# Minimum whitespace gap (points) between adjacent column x-spans.
_MIN_COL_GAP_PT: float = 8.0
# Reject tables with fewer than this many columns.
_MIN_COLS: int = 2
# Reject tables with fewer than this many data rows (rows below the header).
_MIN_DATA_ROWS: int = 1
# Expand the clip rect by this many points when extracting words/pixels.
_CLIP_MARGIN_PT: float = 3.0
# Words whose y-tops differ by less than this are on the same row.
_ROW_GAP_PT: float = 6.0
# Words on the same group-row when grouping for numeric-error detection.
_GROUP_ROW_GAP_PT: float = 4.0
# Regions with IoU above this threshold are considered duplicates.
_OVERLAP_IOU_THRESHOLD: float = 0.30
# Minimum evidence score to keep a scored candidate.
_REGION_MIN_KEEP_SCORE: float = 0.25
# IoU above which two candidate bboxes are merged into one.
_REGION_MERGE_IOU: float = 0.80
# Max y-difference (points) to snap words onto the same text line.
_REGION_LINE_SNAP: float = 3.0
# Maximum points below a caption to search for associated table content.
_CAPTION_SCAN_DEPTH: float = 620.0
# Minimum consecutive table-like lines to emit an alignment-based candidate.
_ALIGNMENT_MIN_RUNS: int = 3
# Minimum x-alignment bins for a line to be considered table-like.
_ALIGNMENT_MIN_BINS: int = 2
# Minimum rotated characters to attempt rotated table detection.
_ROTATED_MIN_CHARS: int = 20
# x-distance tolerance for grouping rotated characters into columns.
_ROTATED_CLUSTER_TOL: float = 6.0
# Maximum x-gap (points) between adjacent column groups in rotated tables.
_ROTATED_MAX_GAP: float = 90.0
# Keywords that suggest a rotated table's header column (biomedical + general).
_ROTATED_HEADER_KEYWORDS: frozenset[str] = frozenset({
    "host", "characteristic", "medium", "titer", "reference",
    "strain", "enzyme", "substrate", "product", "yield",
    "condition", "temperature", "concentration",
    "method", "model", "dataset", "accuracy", "precision", "recall",
    "sample", "value", "result", "parameter", "species",
})
# Minimum character gap (points) to mark a cell boundary in rotated text extraction.
_ROTATED_EXTRACT_GAP_PT: float = 10.0
# Tolerance for clustering row-boundary y-coords across data groups in rotated tables.
_ROTATED_EXTRACT_CLUSTER_TOL: float = 8.0
# Intra-group character gap (points) that marks a new cell in the header column.
_ROTATED_HEADER_GAP_PT: float = 15.0

# ── regex patterns ─────────────────────────────────────────────────────────────
TABLE_CAPTION_RE = re.compile(
    r"^(?:\d+\s+)?(?:table|tab\.)\s*s?\d+(?=\b|[A-Z])"
    r"|^(?:\d+\s+)?supplementary\s*table\s*s?\d+(?=\b|[A-Z])"
    r"|^(?:\d+\s+)?table\s*\d+[.:]"
    r"|^(?:\d+\s+)?表\s*[S号]?\s*\d+\b"
    r"|^(?:\d+\s+)?附表\s*[S号]?\s*\d+\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"[-+]?\d+(?:\.\d+)?%?|[<>]=?\s*\d+|p\s*[<=>]\s*0?\.\d+",
    re.IGNORECASE,
)
_NUMBER_TOKEN_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)%?$")
_ROW_LABEL_RE = re.compile(r"^[A-Za-z]{0,4}\d+[A-Za-z]?$")
_PLUS_MINUS_TEXT: frozenset[str] = frozenset({"±", "+/-", "卤"})
_BODY_WORDS: frozenset[str] = frozenset({
    "abstract", "introduction", "methods", "results",
    "discussion", "conclusion", "references",
})
_PROSE_WORDS: frozenset[str] = frozenset({
    "the", "of", "and", "was", "were", "that", "this", "from", "with", "for",
    "thereby", "respectively", "synthesis", "pathway", "broth", "strain",
    "metabolic", "genes", "yield", "introduced", "strengthened",
})

# Type alias: (x0, y0, x1, y1) in page coordinates (origin = top-left corner).
_BBox = tuple[float, float, float, float]
# Wide horizontal rule as (y_mid, x0, x1), y increases downward.
_Rule = tuple[float, float, float]


# ── internal data types ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _TextLine:
    """A single line of text built from PyMuPDF word tuples."""
    text: str
    bbox: _BBox
    words: list[tuple]  # raw PyMuPDF word tuples (x0,y0,x1,y1,text,…)


# ── geometry helpers ───────────────────────────────────────────────────────────

def _iou(a: _BBox, b: _BBox) -> float:
    """Intersection-over-union for two axis-aligned bboxes (x0, y0, x1, y1)."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _overlap_ratio(inner: _BBox, outer: _BBox) -> float:
    """Fraction of *inner*'s area that is covered by *outer*."""
    ix0 = max(inner[0], outer[0])
    iy0 = max(inner[1], outer[1])
    ix1 = min(inner[2], outer[2])
    iy1 = min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_inner = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area_inner if area_inner > 0 else 0.0


def _overlaps_any(
    bbox: _BBox,
    others: list[_BBox],
    threshold: float = _OVERLAP_IOU_THRESHOLD,
) -> bool:
    """Return True if *bbox* substantially overlaps any region in *others*."""
    return any(_iou(bbox, other) >= threshold for other in others)


def _clip_to_page(
    bbox: _BBox,
    page_width: float,
    page_height: float,
) -> _BBox | None:
    """Clip bbox to page bounds; return None if the result is empty."""
    x0 = max(0.0, bbox[0])
    y0 = max(0.0, bbox[1])
    x1 = min(page_width, bbox[2])
    y1 = min(page_height, bbox[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _serialize_info_value(v: Any) -> Any:
    """Convert PyMuPDF geometry types to JSON-serializable Python scalars."""
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    # IRect, Rect, Point — convert via iteration if possible
    try:
        return list(v)
    except TypeError:
        return str(v)


# ── text line building ─────────────────────────────────────────────────────────

def _build_text_lines(
    words: list[tuple],
    line_snap: float = _REGION_LINE_SNAP,
) -> list[_TextLine]:
    """Group PyMuPDF word tuples ``(x0,y0,x1,y1,text,…)`` into text lines.

    Words whose y0 coordinates differ by at most *line_snap* points are grouped
    into the same line.  Lines are returned sorted top-to-bottom.
    """
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (float(w[1]), float(w[0])))
    rows: list[list[tuple]] = []
    for word in sorted_words:
        top = float(word[1])
        for row in reversed(rows):
            if abs(top - float(row[0][1])) <= line_snap:
                row.append(word)
                break
        else:
            rows.append([word])

    lines: list[_TextLine] = []
    for row in rows:
        row_sorted = sorted(row, key=lambda w: float(w[0]))
        text = " ".join(str(w[4]) for w in row_sorted).strip()
        if not text:
            continue
        lines.append(_TextLine(
            text=text,
            bbox=(
                min(float(w[0]) for w in row_sorted),
                min(float(w[1]) for w in row_sorted),
                max(float(w[2]) for w in row_sorted),
                max(float(w[3]) for w in row_sorted),
            ),
            words=row_sorted,
        ))
    return lines


# ── line detection ─────────────────────────────────────────────────────────────

def _get_wide_h_rules(
    page: pymupdf.Page,
    page_width: float,
) -> list[_Rule]:
    """Return wide horizontal rules as ``(y_mid, x0, x1)`` triples, sorted by y.

    A *wide horizontal rule* is any path whose bounding rectangle spans at
    least :data:`_MIN_LINE_WIDTH_RATIO` of the page width and is no taller
    than :data:`_MAX_RULE_HEIGHT_PT`.  This catches both thin filled
    rectangles (LaTeX booktabs rules) and stroked horizontal line paths.
    """
    min_width = page_width * _MIN_LINE_WIDTH_RATIO
    rules: list[_Rule] = []
    for path in page.get_drawings():
        rect = path.get("rect")
        if rect is None:
            continue
        width = float(rect.x1) - float(rect.x0)
        height = abs(float(rect.y1) - float(rect.y0))
        if width >= min_width and height <= _MAX_RULE_HEIGHT_PT:
            y_mid = (float(rect.y0) + float(rect.y1)) / 2.0
            rules.append((y_mid, float(rect.x0), float(rect.x1)))
    rules.sort(key=lambda r: r[0])
    return rules


def _cluster_rules(
    rules: list[_Rule],
) -> list[list[_Rule]]:
    """Group rules into per-table clusters based on vertical proximity.

    Rules within :data:`_LINE_CLUSTER_GAP_PT` points of each other are
    considered part of the same table.  Clusters with fewer than
    :data:`_MIN_LINES_PER_CLUSTER` rules are discarded.
    """
    if not rules:
        return []
    clusters: list[list[_Rule]] = [[rules[0]]]
    for rule in rules[1:]:
        if rule[0] - clusters[-1][-1][0] <= _LINE_CLUSTER_GAP_PT:
            clusters[-1].append(rule)
        else:
            clusters.append([rule])
    return [c for c in clusters if len(c) >= _MIN_LINES_PER_CLUSTER]


# ── column inference ───────────────────────────────────────────────────────────

def _find_col_ranges(
    words: list[tuple],
    min_gap_pt: float = _MIN_COL_GAP_PT,
) -> list[tuple[float, float]]:
    """Infer column x-ranges by merging overlapping word x-spans.

    Words are PyMuPDF "words" tuples: ``(x0, y0, x1, y1, text, ...)``.
    Adjacent spans separated by less than *min_gap_pt* are merged into one
    column; gaps >= *min_gap_pt* start a new column.
    """
    if not words:
        return []
    spans = sorted((float(w[0]), float(w[2])) for w in words)
    merged: list[tuple[float, float]] = []
    cx0, cx1 = spans[0]
    for sx0, sx1 in spans[1:]:
        if sx0 - cx1 < min_gap_pt:
            cx1 = max(cx1, sx1)
        else:
            merged.append((cx0, cx1))
            cx0, cx1 = sx0, sx1
    merged.append((cx0, cx1))
    return merged


def _assign_col(word: tuple, col_ranges: list[tuple[float, float]]) -> int:
    """Return the column index whose range best contains the word's x-centre.

    Falls back to the nearest column by edge distance when the centre lies
    outside all ranges.
    """
    cx = (float(word[0]) + float(word[2])) / 2.0
    for idx, (cx0, cx1) in enumerate(col_ranges):
        if cx0 <= cx <= cx1:
            return idx
    return min(
        range(len(col_ranges)),
        key=lambda i: min(abs(cx - col_ranges[i][0]), abs(cx - col_ranges[i][1])),
    )


def _words_to_grid(
    words: list[tuple],
    col_ranges: list[tuple[float, float]],
    row_gap_pt: float = _ROW_GAP_PT,
) -> list[list[str]]:
    """Arrange words into a 2-D cell grid ``[row][col]``.

    Words are sorted top-to-bottom, left-to-right.  A new row starts when
    the next word's y0 exceeds the current row's y0 by more than *row_gap_pt*.
    """
    if not words:
        return []
    n_cols = len(col_ranges)
    sorted_words = sorted(words, key=lambda w: (float(w[1]), float(w[0])))
    row_groups: list[list[tuple]] = []
    current: list[tuple] = [sorted_words[0]]
    current_y = float(sorted_words[0][1])
    for w in sorted_words[1:]:
        if float(w[1]) - current_y > row_gap_pt:
            row_groups.append(current)
            current = [w]
            current_y = float(w[1])
        else:
            current.append(w)
    row_groups.append(current)
    grid: list[list[str]] = []
    for row_words in row_groups:
        cells = [""] * n_cols
        for w in row_words:
            ci = _assign_col(w, col_ranges)
            if 0 <= ci < n_cols:
                cells[ci] = (cells[ci] + " " + str(w[4])).strip()
        grid.append(cells)
    return grid


# ── numeric-error cell parsing ─────────────────────────────────────────────────

def _is_number_token(text: str) -> bool:
    return bool(_NUMBER_TOKEN_RE.match(text.strip()))


def _is_plus_minus_token(text: str) -> bool:
    return text.strip() in _PLUS_MINUS_TEXT


def _group_words_by_row(
    words: list[tuple],
    row_gap_pt: float = _GROUP_ROW_GAP_PT,
) -> list[list[tuple]]:
    """Group word tuples into rows by vertical proximity (strict gap)."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (float(w[1]), float(w[0])))
    rows: list[list[tuple]] = []
    current = [sorted_words[0]]
    current_y = float(sorted_words[0][1])
    for w in sorted_words[1:]:
        y = float(w[1])
        if abs(y - current_y) > row_gap_pt:
            rows.append(sorted(current, key=lambda r: float(r[0])))
            current = [w]
            current_y = y
        else:
            current.append(w)
            current_y = (current_y * (len(current) - 1) + y) / len(current)
    rows.append(sorted(current, key=lambda r: float(r[0])))
    return rows


def _numeric_error_cells(
    row_words: list[tuple],
) -> list[tuple[str, tuple[float, float]]] | None:
    """Detect 'key | mean ± sd | mean ± sd' row structure.

    Returns a list of ``(cell_text, (x0, x1))`` pairs if the row matches, else None.
    """
    if len(row_words) < 2:
        return None
    cells: list[tuple[str, tuple[float, float]]] = []
    idx = 0
    first_text = str(row_words[0][4])
    if first_text and not _is_number_token(first_text) and not _is_plus_minus_token(first_text):
        cells.append((first_text, (float(row_words[0][0]), float(row_words[0][2]))))
        idx = 1
    while idx < len(row_words):
        w = row_words[idx]
        text = str(w[4])
        if (
            _is_number_token(text)
            and idx + 2 < len(row_words)
            and _is_plus_minus_token(str(row_words[idx + 1][4]))
            and _is_number_token(str(row_words[idx + 2][4]))
        ):
            x0 = float(row_words[idx][0])
            x1 = float(row_words[idx + 2][2])
            cell_text = f"{text} {row_words[idx + 1][4]} {row_words[idx + 2][4]}"
            cells.append((cell_text, (x0, x1)))
            idx += 3
            continue
        if _is_number_token(text):
            cells.append((text, (float(w[0]), float(w[2]))))
        idx += 1

    numeric_count = sum(
        1 for cell_text, _span in cells
        if any(
            _is_number_token(part)
            for part in cell_text.replace("±", " ± ").replace("卤", " ± ").split()
        )
    )
    return cells if len(cells) >= _MIN_COLS and numeric_count >= 1 else None


def _extract_numeric_error_table(
    header_words: list[tuple],
    data_words: list[tuple],
) -> list[list[str]]:
    """Try to parse a 'key | mean±sd | mean±sd …' scientific table.

    Returns ``[header_row, *data_rows]`` if the pattern matches, else ``[]``.
    """
    row_cells = [
        cells
        for row in _group_words_by_row(data_words)
        if (cells := _numeric_error_cells(row)) is not None
    ]
    if len(row_cells) < _MIN_DATA_ROWS:
        return []
    widths = [len(cells) for cells in row_cells]
    mode_width = max(set(widths), key=lambda w: (widths.count(w), w))
    usable_rows = [cells for cells in row_cells if len(cells) == mode_width]
    if len(usable_rows) < _MIN_DATA_ROWS or mode_width < _MIN_COLS:
        return []
    col_ranges: list[tuple[float, float]] = []
    for col_idx in range(mode_width):
        x0s = [cells[col_idx][1][0] for cells in usable_rows]
        x1s = [cells[col_idx][1][1] for cells in usable_rows]
        col_ranges.append((float(median(x0s)), float(median(x1s))))
    header_row = [""] * mode_width
    for w in header_words:
        cx = (float(w[0]) + float(w[2])) / 2.0
        best_idx, best_dist = 0, float("inf")
        for i, (cx0, cx1) in enumerate(col_ranges):
            if cx0 <= cx <= cx1:
                best_idx = i
                break
            dist = min(abs(cx - cx0), abs(cx - cx1))
            if dist < best_dist:
                best_dist, best_idx = dist, i
        header_row[best_idx] = (header_row[best_idx] + " " + str(w[4])).strip()
    if sum(1 for cell in header_row if cell) < 2:
        return []
    return [header_row] + [[cell_text for cell_text, _span in cells] for cells in usable_rows]


# ── multiline cell merging ─────────────────────────────────────────────────────

def merge_multiline_cells(data_rows: list[list[str]]) -> list[list[str]]:
    """Merge continuation rows into their anchor rows.

    For even-column tables with ≥ 4 columns (dual-column layout), the left
    and right halves are independently merged.  For other tables, rows with an
    empty first cell are appended to the previous row's cells.
    """
    if not data_rows:
        return data_rows
    n_cols = len(data_rows[0]) if data_rows else 0

    if n_cols >= 4 and n_cols % 2 == 0:
        half = n_cols // 2
        result: list[list[str]] = []
        left_key_idx = right_key_idx = -1
        left_last_key = right_last_key = ""

        def _is_pseudo(key: str, char: str, source: str, prev_key: str) -> bool:
            if char:
                return False
            if key.startswith("("):
                return True
            if prev_key.endswith("-"):
                return True
            if source.startswith("("):
                return True
            return False

        def _apply_pseudo(
            prev_row: list[str],
            offset: int,
            key: str,
            char: str,
            source: str,
            prev_key: str,
        ) -> None:
            if key.startswith("("):
                if source:
                    prev_row[offset + 2] = (prev_row[offset + 2] + " " + source).strip()
            elif prev_key.endswith("-"):
                prev_row[offset] = (prev_key + key).strip()
                if char:
                    prev_row[offset + 1] = (prev_row[offset + 1] + " " + char).strip()
                if source:
                    prev_row[offset + 2] = (prev_row[offset + 2] + " " + source).strip()
            else:
                if key:
                    prev_row[offset + 1] = (prev_row[offset + 1] + " " + key).strip()
                if source:
                    prev_row[offset + 2] = (prev_row[offset + 2] + " " + source).strip()

        for row in data_rows:
            left, right = row[:half], row[half:]
            left_key, right_key = left[0].strip(), right[0].strip()
            left_char = left[1].strip() if half > 1 else ""
            right_char = right[1].strip() if half > 1 else ""
            left_source = left[2].strip() if half > 2 else ""
            right_source = right[2].strip() if half > 2 else ""
            left_has = any(cell.strip() for cell in left)
            right_has = any(cell.strip() for cell in right)

            left_pseudo = bool(left_key) and _is_pseudo(left_key, left_char, left_source, left_last_key)
            right_pseudo = bool(right_key) and _is_pseudo(right_key, right_char, right_source, right_last_key)
            left_is_cont = not left_key or left_pseudo
            right_is_cont = not right_key or right_pseudo

            if not left_is_cont and not right_is_cont:
                result.append(list(row))
                left_key_idx = right_key_idx = len(result) - 1
                left_last_key = left_key
                right_last_key = right_key
            elif not left_is_cont and right_is_cont:
                result.append(list(left) + [""] * half)
                left_key_idx = len(result) - 1
                left_last_key = left_key
                if right_has and right_key_idx >= 0:
                    prev = result[right_key_idx]
                    if right_pseudo:
                        _apply_pseudo(prev, half, right_key, right_char, right_source, right_last_key)
                        right_last_key = prev[half]
                    else:
                        for i, cell in enumerate(right):
                            if cell.strip():
                                prev[half + i] = (prev[half + i] + " " + cell).strip()
            elif left_is_cont and not right_is_cont:
                if left_has and left_key_idx >= 0:
                    prev = result[left_key_idx]
                    if left_pseudo:
                        _apply_pseudo(prev, 0, left_key, left_char, left_source, left_last_key)
                        left_last_key = prev[0]
                    else:
                        for i, cell in enumerate(left):
                            if cell.strip():
                                prev[i] = (prev[i] + " " + cell).strip()
                result.append([""] * half + list(right))
                right_key_idx = len(result) - 1
                right_last_key = right_key
            else:
                if left_has and left_key_idx >= 0:
                    prev = result[left_key_idx]
                    if left_pseudo:
                        _apply_pseudo(prev, 0, left_key, left_char, left_source, left_last_key)
                        left_last_key = prev[0]
                    else:
                        for i, cell in enumerate(left):
                            if cell.strip():
                                prev[i] = (prev[i] + " " + cell).strip()
                if right_has and right_key_idx >= 0:
                    prev = result[right_key_idx]
                    if right_pseudo:
                        _apply_pseudo(prev, half, right_key, right_char, right_source, right_last_key)
                        right_last_key = prev[half]
                    else:
                        for i, cell in enumerate(right):
                            if cell.strip():
                                prev[half + i] = (prev[half + i] + " " + cell).strip()
        return result

    # Simple single-column-layout merging: rows with empty first cell continue previous row.
    result2: list[list[str]] = []
    for row in data_rows:
        if not row[0].strip() and result2:
            prev = result2[-1]
            for i, cell in enumerate(row):
                if cell.strip() and i < len(prev):
                    prev[i] = (prev[i] + " " + cell).strip()
        else:
            result2.append(list(row))
    return result2


# ── markdown rendering ─────────────────────────────────────────────────────────

def _to_markdown(header: list[str], data_rows: list[list[str]]) -> str:
    """Render header + data rows as a GitHub-flavoured markdown table."""

    def _fmt(row: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in row) + " |"

    sep = ["---"] * len(header)
    lines = [_fmt(header), "| " + " | ".join(sep) + " |"] + [
        _fmt(row) for row in data_rows
    ]
    return "\n".join(lines)


# ── evidence scoring helpers ───────────────────────────────────────────────────

def _line_alignment_score(line: _TextLine) -> int:
    """Count distinct x-position bins (12 pt grid) occupied by words on this line."""
    if len(line.words) < 2:
        return 0
    bins = Counter(round(float(w[0]) / 12) for w in line.words)
    return len(bins)


def _is_table_like_line(line: _TextLine) -> bool:
    """True if the line has table-like structure (aligned words, short tokens, or numbers)."""
    if _line_alignment_score(line) >= _ALIGNMENT_MIN_BINS:
        return True
    tokens = line.text.split()
    if not tokens:
        return False
    numeric_ratio = sum(1 for t in tokens if _NUMBER_RE.search(t)) / len(tokens)
    short_ratio = sum(1 for t in tokens if len(t.strip(".,;:()[]")) <= 12) / len(tokens)
    return numeric_ratio >= 0.20 or (short_ratio >= 0.70 and len(tokens) >= 3)


def _is_prose_heavy_line(line: _TextLine) -> bool:
    """True if the line is probably running prose (long, many common words)."""
    tokens = [t.strip(".,;:()[]").lower() for t in line.text.split()]
    if len(tokens) < 14:
        return False
    return sum(1 for t in tokens if t in _PROSE_WORDS) >= 5


def _is_table_data_row(line: _TextLine) -> bool:
    """True if the line matches 'RowLabel number number …' pattern."""
    tokens = [t.strip(".,;:()[]") for t in line.text.split() if t.strip(".,;:()[]")]
    if len(tokens) < 2:
        return False
    numeric_count = sum(1 for t in tokens[1:] if _NUMBER_RE.search(t))
    return bool(_ROW_LABEL_RE.match(tokens[0])) and numeric_count >= 2


def _stable_row_bounds(lines: list[_TextLine]) -> tuple[float, float] | None:
    """Return stable (x0, x1) from data rows, trimmed to avoid double-column spill."""
    bounds: list[tuple[float, float]] = []
    for line in lines:
        if not _is_table_data_row(line):
            continue
        words = sorted(line.words, key=lambda w: float(w[0]))
        if not words:
            continue
        kept: list[tuple] = []
        numeric_seen = 0
        for i, w in enumerate(words):
            text = str(w[4]).strip(".,;:()[]")
            if _NUMBER_RE.search(text):
                numeric_seen += 1
            kept.append(w)
            if i + 1 < len(words):
                nxt = words[i + 1]
                gap = float(nxt[0]) - float(w[2])
                if (
                    numeric_seen >= 4
                    and gap >= 16
                    and str(nxt[4]).strip(".,;:()[]").lower() in _PROSE_WORDS
                ):
                    break
        if len(kept) >= 2:
            bounds.append((
                min(float(w[0]) for w in kept),
                max(float(w[2]) for w in kept),
            ))
    if len(bounds) < 2:
        return None
    x0s = sorted(b[0] for b in bounds)
    x1s = sorted(b[1] for b in bounds)
    x0 = x0s[max(0, len(x0s) // 4)]
    x1 = x1s[min(len(x1s) - 1, int(len(x1s) * 0.75))]
    return (x0, x1) if x1 > x0 else None


def _score_region(
    lines: list[_TextLine],
    bbox: _BBox,
    rules: list[_Rule],
    page_width: float,
) -> float:
    """Compute evidence-based confidence for a candidate bbox (0.0 – 1.0)."""
    in_lines = [
        line for line in lines
        if (
            _iou(line.bbox, bbox) > 0
            or (
                bbox[0] <= (line.bbox[0] + line.bbox[2]) / 2 <= bbox[2]
                and bbox[1] <= (line.bbox[1] + line.bbox[3]) / 2 <= bbox[3]
            )
        )
    ]
    score = 0.0

    # Caption near/above this region → strong evidence
    caption_lines = [
        line for line in lines
        if TABLE_CAPTION_RE.search(line.text.strip())
        and line.bbox[3] <= bbox[3]
        and abs(line.bbox[3] - bbox[1]) <= 180
    ]
    if caption_lines:
        score += 0.35

    # Wide horizontal rules within the bbox
    wide_in = [
        r for r in rules
        if bbox[1] - _CLIP_MARGIN_PT <= r[0] <= bbox[3] + _CLIP_MARGIN_PT
    ]
    if len(wide_in) >= 3:
        score += 0.30
    elif len(wide_in) >= 2:
        score += 0.20

    # Aligned text lines
    aligned = [line for line in in_lines if _line_alignment_score(line) >= _ALIGNMENT_MIN_BINS]
    if len(aligned) >= 3:
        score += 0.30
    elif len(aligned) >= 2:
        score += 0.15

    # Numeric density and short tokens
    text = " ".join(line.text for line in in_lines)
    tokens = text.split()
    if tokens:
        numeric_ratio = sum(1 for t in tokens if _NUMBER_RE.search(t)) / len(tokens)
        short_ratio = sum(1 for t in tokens if len(t.strip(".,;:()[]")) <= 12) / len(tokens)
        if numeric_ratio >= 0.20:
            score += 0.15
        if short_ratio >= 0.70 and len(tokens) >= 8:
            score += 0.10

    # Negative: long prose lines (likely body text, not a table)
    long_lines = [line for line in in_lines if len(line.text.split()) >= 18]
    if len(long_lines) >= 3:
        score -= 0.30

    # Negative: section-heading words at the start of lines
    first_words = {
        line.text.strip().split()[0].strip(":.").lower()
        for line in in_lines
        if line.text.strip()
    }
    if first_words & _BODY_WORDS:
        score -= 0.20

    return max(0.0, min(1.0, score))


# ── detection strategy: caption anchoring ─────────────────────────────────────

def _caption_based_regions(
    lines: list[_TextLine],
    rules: list[_Rule],
    page_width: float,
    page_height: float,
) -> list[_BBox]:
    """PRIMARY strategy: anchor table candidates on caption text.

    Searches for "Table N" / "表N" lines, then extends the bbox downward
    to capture the table body using wide rules and table-like text lines.
    """
    candidates: list[_BBox] = []
    for cap in lines:
        if not TABLE_CAPTION_RE.search(cap.text.strip()):
            continue

        # Collect table-like lines below the caption
        nearby = [
            line for line in lines
            if line.bbox[1] > cap.bbox[1]
            and line.bbox[1] <= cap.bbox[3] + _CAPTION_SCAN_DEPTH
        ]
        table_lines: list[_TextLine] = []
        last_bottom = cap.bbox[3]
        for line in nearby:
            gap = line.bbox[1] - last_bottom
            is_data = _is_table_data_row(line)
            if table_lines and gap > 95 and not is_data:
                break
            if is_data or (_is_table_like_line(line) and not _is_prose_heavy_line(line)):
                table_lines.append(line)
                last_bottom = line.bbox[3]
                continue
            first_word = line.text.strip().split()[0].strip(":.").lower() if line.text.strip() else ""
            if table_lines and (first_word in _BODY_WORDS or _is_prose_heavy_line(line)):
                break

        # Collect wide rules below the caption
        wide_rules = [
            r for r in rules
            if cap.bbox[3] - 8 <= r[0] <= min(page_height, cap.bbox[3] + _CAPTION_SCAN_DEPTH)
        ]

        # Derive content x-extent
        row_bounds = _stable_row_bounds(table_lines)
        if row_bounds:
            content_x0 = min(cap.bbox[0], row_bounds[0])
            content_x1 = max(cap.bbox[2], row_bounds[1])
            if wide_rules:
                edge_x0 = min(r[1] for r in wide_rules)
                edge_x1 = max(r[2] for r in wide_rules)
                row_width = max(1.0, row_bounds[1] - row_bounds[0])
                edge_width = max(1.0, edge_x1 - edge_x0)
                content_x0 = min(content_x0, edge_x0)
                if edge_width <= row_width * 1.35:
                    content_x1 = max(content_x1, edge_x1)
        elif wide_rules:
            content_x0 = min([cap.bbox[0]] + [r[1] for r in wide_rules])
            content_x1 = max([cap.bbox[2]] + [r[2] for r in wide_rules])
        elif table_lines:
            content_x0 = min(cap.bbox[0], min(line.bbox[0] for line in table_lines))
            content_x1 = max(cap.bbox[2], max(line.bbox[2] for line in table_lines), page_width * 0.55)
        else:
            continue  # caption with no nearby table content

        # Derive bottom y
        bottoms = [cap.bbox[3] + 120]
        if table_lines:
            bottoms.append(max(line.bbox[3] for line in table_lines) + 24)
        if wide_rules:
            bottoms.append(max(r[0] for r in wide_rules) + 60)

        candidates.append((
            max(0.0, content_x0 - 8),
            max(0.0, cap.bbox[1] - 4),
            min(page_width, content_x1 + 8),
            min(page_height, max(bottoms)),
        ))
    return candidates


# ── detection strategy: wide horizontal-rule clusters ─────────────────────────

def _line_based_regions(
    rules: list[_Rule],
    page_width: float,
    page_height: float,
) -> list[_BBox]:
    """SECONDARY strategy: generate candidates from clusters of wide horizontal rules."""
    candidates: list[_BBox] = []
    for cluster in _cluster_rules(rules):
        x0 = max(0.0, min(r[1] for r in cluster) - 4)
        x1 = min(page_width, max(r[2] for r in cluster) + 4)
        y_top = min(r[0] for r in cluster)
        y_bottom = max(r[0] for r in cluster)
        y0 = max(0.0, y_top - 16)
        y1 = min(page_height, y_bottom + 100)
        candidates.append((x0, y0, x1, y1))
    return candidates


# ── detection strategy: text-alignment runs ───────────────────────────────────

def _alignment_based_regions(
    lines: list[_TextLine],
    page_width: float,
    page_height: float,
) -> list[_BBox]:
    """FALLBACK strategy: detect tables by repeated x-alignment in text lines.

    Finds runs of ≥ :data:`_ALIGNMENT_MIN_RUNS` consecutive lines with
    ≥ :data:`_ALIGNMENT_MIN_BINS` distinct x-position bins.
    """
    runs: list[list[_TextLine]] = []
    current: list[_TextLine] = []
    for line in lines:
        if _line_alignment_score(line) >= _ALIGNMENT_MIN_BINS:
            current.append(line)
        else:
            if len(current) >= _ALIGNMENT_MIN_RUNS:
                runs.append(current)
            current = []
    if len(current) >= _ALIGNMENT_MIN_RUNS:
        runs.append(current)

    candidates: list[_BBox] = []
    for run in runs:
        row_bounds = _stable_row_bounds(run)
        if row_bounds:
            raw_x0, raw_x1 = row_bounds
        else:
            raw_x0 = min(line.bbox[0] for line in run)
            raw_x1 = max(line.bbox[2] for line in run)
        candidates.append((
            max(0.0, raw_x0 - 8),
            max(0.0, min(line.bbox[1] for line in run) - 8),
            min(page_width, raw_x1 + 8),
            min(page_height, max(line.bbox[3] for line in run) + 8),
        ))
    return candidates


# ── rotated table text extraction ─────────────────────────────────────────────

def _extract_rotated_cells_from_chars(
    rotated_chars: list[dict],
) -> list[list[str]]:
    """Extract a structured cell grid from 90°-rotated characters.

    Ported from PaperSort's ``detect_rotated_table`` with all pdfplumber-specific
    APIs replaced by the unified ``rotated_chars`` format
    (keys: ``text``, ``x0``, ``top``, ``x1``, ``bottom``).

    Each "column group" (characters sharing the same x0 bin) represents one row
    of the original upright table; intra-group character gaps mark cell boundaries.

    Algorithm:

    1. Group characters by ``round(x0 * 2) / 2`` (0.5 pt bins, stable across calls).
    2. Identify the **title** group (leftmost two groups checked against
       ``TABLE_CAPTION_RE`` / ``startswith("table")``).
    3. Identify the **header** group (has ≥ 3 domain keywords, or fallback to
       first non-title group with ≥ 2 chars).
    4. Infer column count from intra-header-group character gaps
       (gap > :data:`_ROTATED_HEADER_GAP_PT` → new cell).
    5. Derive cell-boundary y-positions via gap analysis on data groups
       (gap > :data:`_ROTATED_EXTRACT_GAP_PT`), then cluster the boundaries.
    6. Assemble ``[header_row, *data_rows]``.

    Returns an empty list when the structure cannot be inferred.
    """
    if len(rotated_chars) < _ROTATED_MIN_CHARS:
        return []

    # Group by x0 using 0.5 pt bins — stable, matches PaperSort's approach.
    groups: dict[float, list[dict]] = defaultdict(list)
    for char in rotated_chars:
        key = round(char["x0"] * 2) / 2
        groups[key].append(char)
    sorted_keys = sorted(groups.keys())
    if len(sorted_keys) < 3:
        return []

    # ── locate title group (leftmost 2 groups checked) ────────────────────────
    title_key: float | None = None
    for key in sorted_keys[:2]:
        text = "".join(
            c["text"] for c in sorted(groups[key], key=lambda c: -c["top"])
        ).strip()
        if TABLE_CAPTION_RE.search(text) or text.lower().startswith("table"):
            title_key = key
            break

    # ── locate header group (keyword match; fallback = first non-title group) ─
    header_key: float | None = None
    for key in sorted_keys:
        if key == title_key:
            continue
        kw_text = "".join(
            c["text"] for c in sorted(groups[key], key=lambda c: -c["top"])
        ).lower()
        if sum(1 for kw in _ROTATED_HEADER_KEYWORDS if kw in kw_text) >= 3:
            header_key = key
            break
    if header_key is None:
        for key in sorted_keys:
            if key != title_key and len(groups[key]) >= 2:
                header_key = key
                break
    if header_key is None:
        return []

    # ── build header row: gaps > _ROTATED_HEADER_GAP_PT mark new cells ────────
    hdr_sorted = sorted(groups[header_key], key=lambda c: -c["top"])
    header_col_groups: list[list[dict]] = []
    current: list[dict] = [hdr_sorted[0]]
    for i in range(1, len(hdr_sorted)):
        gap = hdr_sorted[i - 1]["top"] - hdr_sorted[i]["top"]
        if gap > _ROTATED_HEADER_GAP_PT:
            header_col_groups.append(current)
            current = []
        current.append(hdr_sorted[i])
    header_col_groups.append(current)

    n_cols = len(header_col_groups)
    if n_cols < 2:
        return []

    header_row = [
        "".join(c["text"] for c in grp)
        for grp in header_col_groups
    ]

    # ── split remaining keys into main and continuation data groups ───────────
    data_keys = [k for k in sorted_keys if k not in (title_key, header_key)]
    if not data_keys:
        return []

    last_col_max_top = max(c["top"] for c in header_col_groups[-1])
    main_threshold = last_col_max_top + 10.0
    main_keys = [k for k in data_keys if min(c["top"] for c in groups[k]) <= main_threshold]
    continuation_keys = [k for k in data_keys if k not in main_keys]
    if not main_keys:
        return []

    row_groups: dict[float, list[float]] = {k: [k] for k in main_keys}
    for cont in continuation_keys:
        nearest = min(main_keys, key=lambda mk: abs(mk - cont))
        row_groups[nearest].append(cont)

    # ── gap analysis: collect cell-boundary positions from data groups ─────────
    gap_pairs: list[tuple[float, float]] = []
    for main_key in main_keys:
        chars_sorted = sorted(groups[main_key], key=lambda c: -c["top"])
        tops = [c["top"] for c in chars_sorted]
        for i in range(len(tops) - 1):
            if tops[i] - tops[i + 1] > _ROTATED_EXTRACT_GAP_PT:
                gap_pairs.append((tops[i], tops[i + 1]))

    if len(gap_pairs) < n_cols - 1:
        return []

    # ── cluster gap pairs into shared cell-boundary positions ──────────────────
    used: set[int] = set()
    clusters: list[tuple[float, list[float], list[float]]] = []
    for i, (before_top, after_top) in enumerate(gap_pairs):
        if i in used:
            continue
        cluster_before = [before_top]
        cluster_after = [after_top]
        for j in range(i + 1, len(gap_pairs)):
            if j in used:
                continue
            next_before, next_after = gap_pairs[j]
            if abs(after_top - next_after) <= _ROTATED_EXTRACT_CLUSTER_TOL:
                cluster_before.append(next_before)
                cluster_after.append(next_after)
                used.add(j)
        used.add(i)
        boundary = (min(cluster_before) + max(cluster_after)) / 2.0
        clusters.append((boundary, cluster_before, cluster_after))

    clusters.sort(key=lambda cl: -cl[0])
    if len(clusters) < n_cols - 1:
        return []

    boundaries = sorted(
        [cl[0] for cl in clusters[: n_cols - 1]],
        reverse=True,
    )

    def _top_to_col(top: float) -> int:
        return sum(1 for b in boundaries if top <= b)

    # ── assemble data rows ─────────────────────────────────────────────────────
    result: list[list[str]] = [header_row]
    for main_key in sorted(main_keys):
        col_subrows: list[list[list[dict]]] = [[] for _ in range(n_cols)]
        for key in sorted(row_groups[main_key]):
            subrows: list[list[dict]] = [[] for _ in range(n_cols)]
            for char in groups[key]:
                col_idx = _top_to_col(char["top"])
                if 0 <= col_idx < n_cols:
                    subrows[col_idx].append(char)
            for col_idx in range(n_cols):
                if subrows[col_idx]:
                    col_subrows[col_idx].append(subrows[col_idx])

        cols = [""] * n_cols
        for col_idx, sub_list in enumerate(col_subrows):
            if not sub_list:
                continue
            if len(sub_list) == 1:
                cols[col_idx] = "".join(
                    c["text"] for c in sorted(sub_list[0], key=lambda c: -c["top"])
                )
                continue
            # Multiple sub-groups: check for wrap-around (overlapping y-ranges).
            ranges = [
                (min(c["top"] for c in sr), max(c["top"] for c in sr))
                for sr in sub_list
            ]
            has_wrap = any(
                sr_a[1] > sr_b[0] - 5 and sr_b[1] > sr_a[0] - 5
                for idx_a, sr_a in enumerate(ranges)
                for sr_b in ranges[idx_a + 1:]
            )
            if has_wrap:
                cols[col_idx] = "".join(
                    "".join(c["text"] for c in sorted(sr, key=lambda c: -c["top"]))
                    for sr in sub_list
                )
            else:
                all_c = [c for sr in sub_list for c in sr]
                cols[col_idx] = "".join(
                    c["text"] for c in sorted(all_c, key=lambda c: -c["top"])
                )

        if any(col.strip() for col in cols):
            result.append(cols)

    return result if len(result) >= 2 else []


# ── detection strategy: rotated tables ────────────────────────────────────────

def _rotated_regions(
    page: pymupdf.Page,
    page_num: int,
    page_width: float,
    page_height: float,
) -> list[tuple[_BBox, float, list[dict]]]:
    """SPECIAL CASE: detect 90°-rotated tables via character direction vectors.

    Returns a list of ``(bbox, confidence, rotated_chars)`` triples.  The
    ``rotated_chars`` are all the rotated characters in the detected cluster;
    they are passed to :func:`_extract_rotated_cells_from_chars` for structured
    text extraction.  These candidates bypass the normal scoring pipeline and
    are merged into the final scored list directly.

    Uses PyMuPDF's ``get_text("rawdict")`` to find lines whose direction
    vector has a non-zero sin component (i.e. the text is not horizontal).
    """
    rotated_chars: list[dict] = []
    try:
        rawdict = page.get_text("rawdict")
    except Exception:
        logger.debug("Page %d: get_text('rawdict') failed.", page_num + 1, exc_info=True)
        return []

    for block in rawdict.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block
            continue
        for line in block.get("lines", []):
            dir_vec = line.get("dir", (1.0, 0.0))
            dir_x, dir_y = float(dir_vec[0]), float(dir_vec[1])
            # Skip horizontal lines (dir ≈ (1, 0))
            if abs(dir_y) <= 0.1 and abs(dir_x - 1.0) <= 0.1:
                continue
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    bbox = char.get("bbox", (0, 0, 0, 0))
                    rotated_chars.append({
                        "text": char.get("c", ""),
                        "x0": float(bbox[0]),
                        "top": float(bbox[1]),
                        "x1": float(bbox[2]),
                        "bottom": float(bbox[3]),
                    })

    if len(rotated_chars) < _ROTATED_MIN_CHARS:
        return []

    # Group characters by x-position
    sorted_chars = sorted(rotated_chars, key=lambda c: c["x0"])
    groups: list[list[dict]] = []
    for char in sorted_chars:
        x = char["x0"]
        for group in reversed(groups):
            group_x = sum(c["x0"] for c in group) / len(group)
            if abs(x - group_x) <= _ROTATED_CLUSTER_TOL:
                group.append(char)
                break
        else:
            groups.append([char])

    # Find groups containing a table caption
    caption_group_indices: list[int] = []
    for i, group in enumerate(groups):
        text = "".join(c["text"] for c in sorted(group, key=lambda c: -c["top"])).strip()
        if TABLE_CAPTION_RE.search(text):
            caption_group_indices.append(i)
    if not caption_group_indices:
        return []

    results: list[tuple[_BBox, float, list[dict]]] = []
    for cap_idx in caption_group_indices:
        cluster_indices: set[int] = {cap_idx}
        for direction in (-1, 1):
            idx, prev = cap_idx + direction, cap_idx
            while 0 <= idx < len(groups):
                gap = abs(
                    sum(c["x0"] for c in groups[idx]) / len(groups[idx])
                    - sum(c["x0"] for c in groups[prev]) / len(groups[prev])
                )
                if gap > _ROTATED_MAX_GAP:
                    break
                if len(groups[idx]) >= 2:
                    cluster_indices.add(idx)
                    prev = idx
                idx += direction

        total_chars = sum(len(groups[i]) for i in cluster_indices)
        if total_chars < _ROTATED_MIN_CHARS:
            continue
        all_chars = [c for i in cluster_indices for c in groups[i]]
        x0 = max(0.0, min(c["x0"] for c in all_chars) - 8.0)
        y0 = max(0.0, min(c["top"] for c in all_chars) - 8.0)
        x1 = min(page_width, max(c["x1"] for c in all_chars) + 8.0)
        y1 = min(page_height, max(c["bottom"] for c in all_chars) + 8.0)
        if x1 <= x0 or y1 <= y0:
            continue
        # Pass cluster chars to extraction; bbox_chars includes all detected cluster chars.
        # Caption (0.35) + rotation evidence (0.30) = 0.65
        results.append(((x0, y0, x1, y1), 0.65, all_chars))
        logger.debug(
            "Page %d: rotated table candidate at bbox=(%.1f,%.1f,%.1f,%.1f), chars=%d.",
            page_num + 1, x0, y0, x1, y1, total_chars,
        )
    return results


# ── candidate management ───────────────────────────────────────────────────────

def _merge_candidates(candidates: list[_BBox]) -> list[_BBox]:
    """Merge overlapping candidates by expanding to their union bbox."""
    merged: list[_BBox] = []
    for bbox in candidates:
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        for i, existing in enumerate(merged):
            mutual = min(_overlap_ratio(bbox, existing), _overlap_ratio(existing, bbox))
            if _iou(bbox, existing) >= _REGION_MERGE_IOU or mutual >= _REGION_MERGE_IOU:
                merged[i] = (
                    min(existing[0], bbox[0]),
                    min(existing[1], bbox[1]),
                    max(existing[2], bbox[2]),
                    max(existing[3], bbox[3]),
                )
                break
        else:
            merged.append(bbox)
    return merged


# ── table cell extraction ──────────────────────────────────────────────────────

def _extract_table_cells(
    page: pymupdf.Page,
    bbox: _BBox,
    rules: list[_Rule],
) -> tuple[list[str], list[list[str]]]:
    """Extract header and data rows from a candidate table bbox.

    Returns ``(header_row, data_rows)``.  Returns ``([], [])`` when the
    region contains too few usable cells.

    Header / data split logic:
    - ≥ 3 rules inside bbox → use the second rule (midrule) as separator.
    - 2 rules → 25 % heuristic from the top.
    - 0–1 rules → treat the first word-row as the header.

    When rules are present the clip top is clamped to the first rule so that
    any caption text sitting above the toprule is excluded from word extraction.
    """
    import pymupdf as _pymupdf

    x0, y_top, x1, y_bottom = bbox

    # Identify rules that fall inside this bbox first (no word extraction needed).
    in_rules = sorted(
        [r for r in rules if y_top - _CLIP_MARGIN_PT <= r[0] <= y_bottom + _CLIP_MARGIN_PT],
        key=lambda r: r[0],
    )

    # When rules are present, clip from the first rule downward so that
    # caption text above the toprule does not pollute the header zone.
    clip_top = (in_rules[0][0] - _CLIP_MARGIN_PT) if in_rules else (y_top - _CLIP_MARGIN_PT)

    clip = _pymupdf.Rect(
        x0 - _CLIP_MARGIN_PT, clip_top,
        x1 + _CLIP_MARGIN_PT, y_bottom + _CLIP_MARGIN_PT,
    )
    all_words: list[tuple] = page.get_text("words", clip=clip) or []
    if not all_words:
        return [], []

    if len(in_rules) >= 3:
        sep_y = in_rules[1][0]
    elif len(in_rules) == 2:
        sep_y = y_top + (y_bottom - y_top) * 0.25
    else:
        rows = _group_words_by_row(all_words)
        if len(rows) < 2:
            return [], []
        sep_y = float(rows[0][-1][3]) + _CLIP_MARGIN_PT

    header_words = [w for w in all_words if float(w[3]) <= sep_y + _CLIP_MARGIN_PT]
    data_words = [w for w in all_words if float(w[1]) > sep_y - _CLIP_MARGIN_PT]

    if not header_words:
        return [], []

    # Try numeric-error table (mean±sd) pattern first
    numeric = _extract_numeric_error_table(header_words, data_words)
    if numeric:
        return numeric[0], numeric[1:]

    # Standard column inference
    col_ranges = _find_col_ranges(header_words)
    if len(col_ranges) < _MIN_COLS:
        col_ranges = _find_col_ranges(all_words)
    if len(col_ranges) < _MIN_COLS:
        return [], []

    n_cols = len(col_ranges)
    flat_header = [""] * n_cols
    for row in _words_to_grid(header_words, col_ranges):
        for ci, cell in enumerate(row):
            if cell:
                flat_header[ci] = (flat_header[ci] + " " + cell).strip()

    data_grid = _words_to_grid(data_words, col_ranges)
    data_grid = merge_multiline_cells(data_grid)
    return flat_header, data_grid


# ── main entry point ───────────────────────────────────────────────────────────

def detect_borderless_tables(
    page: pymupdf.Page,
    page_num: int,
    page_width: float,
    dpi: float | None,
    pymupdf_pixmap_attrs: set[str],
    already_detected_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[ParsedMedia]:
    """Detect borderless (three-line / booktabs-style) tables missed by ``find_tables()``.

    Applies four strategies — caption anchoring, wide horizontal-rule clustering,
    text-alignment runs, and rotated-text detection — scores each candidate by
    multi-evidence confidence, and emits one :class:`~paperqa.types.ParsedMedia`
    entry per table found (``type="table", detection_method="borderless"``).

    Candidates whose bounding box substantially overlaps an already-detected
    table (IoU >= :data:`_OVERLAP_IOU_THRESHOLD`) are skipped to avoid
    re-emitting tables that ``find_tables()`` already found.

    Parameters
    ----------
    page:
        An open PyMuPDF page.
    page_num:
        Zero-indexed page number; ``page_num + 1`` is stored in metadata.
    page_width:
        Page width in points.
    dpi:
        Optional rendering DPI; forwarded to ``page.get_pixmap``.
    pymupdf_pixmap_attrs:
        Attribute names to copy from the rendered pixmap into ``info``.
    already_detected_bboxes:
        Bboxes of tables already detected by ``find_tables()``.
    """
    import pymupdf as _pymupdf

    already: list[_BBox] = list(already_detected_bboxes or [])
    page_height = float(page.rect.height)

    # ── shared inputs ──────────────────────────────────────────────────────────
    all_page_words: list[tuple] = page.get_text("words") or []
    lines = _build_text_lines(all_page_words)
    rules = _get_wide_h_rules(page, page_width)

    # ── gather candidates from deterministic strategies ────────────────────────
    candidates: list[_BBox] = []
    candidates.extend(_caption_based_regions(lines, rules, page_width, page_height))
    candidates.extend(_line_based_regions(rules, page_width, page_height))
    candidates.extend(_alignment_based_regions(lines, page_width, page_height))

    # Clip to page bounds and merge overlapping candidates
    clipped: list[_BBox] = [
        c for bbox in candidates
        if (c := _clip_to_page(bbox, page_width, page_height)) is not None
    ]
    clipped = _merge_candidates(clipped)

    # ── score and filter candidates ────────────────────────────────────────────
    # Third element: pre-extracted cell grid for rotated tables (None for normal tables).
    scored: list[tuple[_BBox, float, list[list[str]] | None]] = []
    for bbox in clipped:
        s = _score_region(lines, bbox, rules, page_width)
        if s >= _REGION_MIN_KEEP_SCORE:
            scored.append((bbox, s, None))

    # ── add rotated candidates (pre-scored; bypass normal threshold) ───────────
    for bbox, rot_score, rot_chars in _rotated_regions(page, page_num, page_width, page_height):
        c = _clip_to_page(bbox, page_width, page_height)
        if c is not None:
            cells = _extract_rotated_cells_from_chars(rot_chars)
            scored.append((c, rot_score, cells if len(cells) >= 2 else None))

    # Sort by confidence descending for consistent output order
    scored.sort(key=lambda item: -item[1])

    # ── emit ParsedMedia for each surviving candidate ──────────────────────────
    results: list[ParsedMedia] = []
    emitted_bboxes: list[_BBox] = list(already)

    for bbox, _score, pre_cells in scored:
        if _overlaps_any(bbox, emitted_bboxes):
            logger.debug(
                "Page %d: borderless candidate at y=[%.1f, %.1f] overlaps an "
                "already-detected table; skipping.",
                page_num + 1, bbox[1], bbox[3],
            )
            continue
        if bbox[3] - bbox[1] < _MIN_TABLE_HEIGHT_PT:
            continue

        # Rotated tables supply pre-extracted cells; others use standard extraction.
        if pre_cells is not None and len(pre_cells) >= 2:
            header_row, data_rows = pre_cells[0], pre_cells[1:]
        else:
            header_row, data_rows = _extract_table_cells(page, bbox, rules)

        if not header_row or len(data_rows) < _MIN_DATA_ROWS:
            logger.debug(
                "Page %d: borderless candidate at y=[%.1f, %.1f] yielded no "
                "usable cells; skipping.",
                page_num + 1, bbox[1], bbox[3],
            )
            continue

        clip = _pymupdf.Rect(
            bbox[0] - _CLIP_MARGIN_PT, bbox[1] - _CLIP_MARGIN_PT,
            bbox[2] + _CLIP_MARGIN_PT, bbox[3] + _CLIP_MARGIN_PT,
        )
        pix = page.get_pixmap(clip=clip, dpi=dpi)
        pixmap_attrs = {attr: _serialize_info_value(getattr(pix, attr)) for attr in pymupdf_pixmap_attrs}
        media_info: dict = {
            "bbox": (clip.x0, clip.y0, clip.x1, clip.y1),
            "type": "table",
            "detection_method": "borderless",
        } | pixmap_attrs
        media_info["info_hashable"] = json.dumps(
            {
                k: (tuple(int(round(v)) for v in val) if k == "bbox" else val)
                for k, val in media_info.items()
                if isinstance(val, (int, float, str, bool, list, tuple)) or val is None
            },
            sort_keys=True,
        )
        media_info["page_num"] = page_num + 1

        results.append(
            ParsedMedia(
                index=len(already) + len(results),
                data=pix.tobytes(),
                text=clean_invalid_unicode(_to_markdown(header_row, data_rows)),
                info=media_info,
            )
        )
        emitted_bboxes.append(bbox)
        logger.debug(
            "Page %d: detected borderless table at y=[%.1f, %.1f], "
            "%d col(s), %d data row(s).",
            page_num + 1, bbox[1], bbox[3], len(header_row), len(data_rows),
        )

    return results
