"""
Detection of borderless (three-line / booktabs-style) academic tables.

Supplements PyMuPDF's ``find_tables()`` with two strategies that catch
tables missed because they have no vertical border lines:

1. **Caption anchoring** — searches for "Table N" / "表N" text and extends
   a bounding box downward to capture the table body.
2. **Wide H-rule clustering** — groups drawing-layer horizontal rules
   spanning ≥ 20 % of the page width into candidate table bands.

Ported from PaperSort's ``TableRegionDetector`` / ``extract_threeline``
with all pdfplumber-specific APIs replaced by PyMuPDF equivalents.
"""
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pymupdf
from paperqa.types import ParsedMedia
from paperqa.utils import clean_invalid_unicode

logger = logging.getLogger(__name__)

# ── tunable constants ──────────────────────────────────────────────────────────
_MIN_LINE_WIDTH_RATIO: float = 0.20   # rule must span ≥ this fraction of page width
_MAX_RULE_HEIGHT_PT: float = 3.0      # taller paths are not horizontal rules
_LINE_CLUSTER_GAP_PT: float = 200.0   # max vertical gap between rules in one cluster
_MIN_LINES_PER_CLUSTER: int = 2       # cluster needs at least this many rules
_MIN_TABLE_HEIGHT_PT: float = 20.0    # minimum bbox height to keep a candidate
_MIN_COL_GAP_PT: float = 8.0          # whitespace gap to start a new column
_MIN_COLS: int = 2                    # reject tables with fewer columns
_MIN_DATA_ROWS: int = 1               # reject tables with fewer data rows
_CLIP_MARGIN_PT: float = 3.0          # expand clip rect by this on all sides
_ROW_GAP_PT: float = 6.0              # y-gap to start a new word-row
_OVERLAP_IOU_THRESHOLD: float = 0.30  # IoU threshold for duplicate suppression
_REGION_MIN_KEEP_SCORE: float = 0.25  # discard candidates below this score
_REGION_MERGE_IOU: float = 0.80       # merge candidates with IoU above this
_REGION_LINE_SNAP: float = 3.0        # y-snap for grouping words into text lines
_CAPTION_SCAN_DEPTH: float = 620.0    # max pts below caption to scan for table
_ALIGNMENT_MIN_BINS: int = 2          # min x-bins on a line to be "table-like"

TABLE_CAPTION_RE = re.compile(
    r"^(?:\d+\s+)?(?:table|tab\.)\s*s?\d+(?=\b|[A-Z])"
    r"|^(?:\d+\s+)?supplementary\s*table\s*s?\d+(?=\b|[A-Z])"
    r"|^(?:\d+\s+)?table\s*\d+[.:]"
    r"|^(?:\d+\s+)?表\s*[S号]?\s*\d+\b"
    r"|^(?:\d+\s+)?附表\s*[S号]?\s*\d+\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"[-+]?\d+(?:\.\d+)?%?|[<>]=?\s*\d+|p\s*[<=>]\s*0?\.\d+", re.IGNORECASE
)
_BODY_WORDS: frozenset[str] = frozenset({
    "abstract", "introduction", "methods", "results",
    "discussion", "conclusion", "references",
})

_BBox = tuple[float, float, float, float]
_Rule = tuple[float, float, float]  # (y_mid, x0, x1)


@dataclass(frozen=True)
class _TextLine:
    """A single line of text built from PyMuPDF word tuples."""

    text: str
    bbox: _BBox
    words: list[tuple]


# ── geometry helpers ───────────────────────────────────────────────────────────


def _iou(a: _BBox, b: _BBox) -> float:
    """Return intersection-over-union for two axis-aligned bounding boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _overlap_ratio(inner: _BBox, outer: _BBox) -> float:
    """Return the fraction of *inner*'s area that is covered by *outer*."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


def _overlaps_any(
    bbox: _BBox,
    others: list[_BBox],
    threshold: float = _OVERLAP_IOU_THRESHOLD,
) -> bool:
    """Return True if *bbox* substantially overlaps any region in *others*."""
    return any(_iou(bbox, other) >= threshold for other in others)


def _clip_to_page(
    bbox: _BBox, page_width: float, page_height: float
) -> _BBox | None:
    """Clip *bbox* to page bounds; return None if the result is empty."""
    x0, y0 = max(0.0, bbox[0]), max(0.0, bbox[1])
    x1, y1 = min(page_width, bbox[2]), min(page_height, bbox[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _serialize_info_value(v: Any) -> Any:
    """Convert PyMuPDF geometry types (IRect, Rect) to JSON-serializable values."""
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    try:
        return list(v)
    except TypeError:
        return str(v)


# ── text-line building ─────────────────────────────────────────────────────────


def _build_text_lines(
    words: list[tuple], line_snap: float = _REGION_LINE_SNAP
) -> list[_TextLine]:
    """Group PyMuPDF word tuples into text lines.

    Args:
        words: Word tuples from ``page.get_text("words")``.
        line_snap: Maximum y-difference to group words onto the same line.

    Returns:
        Text lines sorted top-to-bottom.
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
        if text:
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


# ── rule detection ─────────────────────────────────────────────────────────────


def _get_wide_h_rules(page: pymupdf.Page, page_width: float) -> list[_Rule]:
    """Return wide horizontal rules as ``(y_mid, x0, x1)`` triples, sorted by y.

    A wide rule is any path whose bounding rectangle spans at least
    ``_MIN_LINE_WIDTH_RATIO`` of the page width and is no taller than
    ``_MAX_RULE_HEIGHT_PT``.

    Args:
        page: An open PyMuPDF page.
        page_width: Page width in points, used to compute the minimum span.

    Returns:
        Rules sorted by y-coordinate.
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
            rules.append(
                ((float(rect.y0) + float(rect.y1)) / 2, float(rect.x0), float(rect.x1))
            )
    rules.sort(key=lambda r: r[0])
    return rules


def _cluster_rules(rules: list[_Rule]) -> list[list[_Rule]]:
    """Group rules into per-table clusters based on vertical proximity.

    Args:
        rules: Wide horizontal rules sorted by y-coordinate.

    Returns:
        Clusters of at least ``_MIN_LINES_PER_CLUSTER`` rules each.
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
    words: list[tuple], min_gap_pt: float = _MIN_COL_GAP_PT
) -> list[tuple[float, float]]:
    """Infer column x-ranges by merging overlapping word x-spans.

    Args:
        words: PyMuPDF word tuples ``(x0, y0, x1, y1, text, …)``.
        min_gap_pt: Minimum whitespace gap to start a new column.

    Returns:
        Non-overlapping ``(x0, x1)`` column ranges sorted left-to-right.
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
    """Return the column index best containing the word's x-centre.

    Falls back to the nearest column by edge distance when the centre lies
    outside all ranges.
    """
    cx = (float(word[0]) + float(word[2])) / 2
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

    Args:
        words: PyMuPDF word tuples.
        col_ranges: Column ``(x0, x1)`` ranges from ``_find_col_ranges``.
        row_gap_pt: y-gap threshold to start a new row.

    Returns:
        Grid of cell strings, one list per row.
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


def merge_multiline_cells(data_rows: list[list[str]]) -> list[list[str]]:
    """Merge continuation rows into the previous anchor row.

    A continuation row is one whose first cell is empty.  Its non-empty
    cells are appended to the corresponding cells of the preceding row.

    Args:
        data_rows: Grid of cell strings, one list per row.

    Returns:
        Merged data rows.
    """
    result: list[list[str]] = []
    for row in data_rows:
        if not row[0].strip() and result:
            for i, cell in enumerate(row):
                if cell.strip() and i < len(result[-1]):
                    result[-1][i] = (result[-1][i] + " " + cell).strip()
        else:
            result.append(list(row))
    return result


# ── markdown rendering ─────────────────────────────────────────────────────────


def _to_markdown(header: list[str], data_rows: list[list[str]]) -> str:
    """Render a cell grid as a GitHub-flavoured markdown table.

    Note: ``pymupdf.table.Table.to_markdown()`` only works on PyMuPDF Table
    objects.  Since our detected tables are plain ``list[list[str]]`` grids,
    this minimal equivalent is used instead.

    Args:
        header: Column header cells.
        data_rows: Data rows, each a list of cell strings.

    Returns:
        A GitHub-flavoured markdown table string.
    """
    def _row(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    return "\n".join([_row(header), sep, *(_row(r) for r in data_rows)])


# ── evidence scoring ───────────────────────────────────────────────────────────


def _score_region(
    lines: list[_TextLine],
    bbox: _BBox,
    rules: list[_Rule],
) -> float:
    """Compute an evidence-based confidence score for a candidate bbox.

    Args:
        lines: All text lines on the page.
        bbox: Candidate table bounding box.
        rules: Wide horizontal rules on the page.

    Returns:
        A score in ``[0.0, 1.0]``.
    """
    in_lines = [
        ln for ln in lines
        if bbox[0] <= (ln.bbox[0] + ln.bbox[2]) / 2 <= bbox[2]
        and bbox[1] <= (ln.bbox[1] + ln.bbox[3]) / 2 <= bbox[3]
    ]
    score = 0.0

    if any(
        TABLE_CAPTION_RE.search(ln.text.strip())
        and ln.bbox[3] <= bbox[3]
        and abs(ln.bbox[3] - bbox[1]) <= 180
        for ln in lines
    ):
        score += 0.35

    wide_in = [
        r for r in rules if bbox[1] - _CLIP_MARGIN_PT <= r[0] <= bbox[3] + _CLIP_MARGIN_PT
    ]
    score += 0.30 if len(wide_in) >= 3 else 0.20 if len(wide_in) >= 2 else 0.0

    aligned = [
        ln for ln in in_lines
        if len(Counter(round(float(w[0]) / 12) for w in ln.words)) >= _ALIGNMENT_MIN_BINS
    ]
    score += 0.30 if len(aligned) >= 3 else 0.15 if len(aligned) >= 2 else 0.0

    tokens = " ".join(ln.text for ln in in_lines).split()
    if tokens:
        if sum(1 for t in tokens if _NUMBER_RE.search(t)) / len(tokens) >= 0.20:
            score += 0.15
        if (
            sum(1 for t in tokens if len(t.strip(".,;:()[]")) <= 12) / len(tokens) >= 0.70
            and len(tokens) >= 8
        ):
            score += 0.10

    if sum(1 for ln in in_lines if len(ln.text.split()) >= 18) >= 3:
        score -= 0.30
    first_words = {
        ln.text.strip().split()[0].strip(":.").lower()
        for ln in in_lines if ln.text.strip()
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
    """Detect table regions by anchoring on caption text.

    Searches for "Table N" / "表N" lines, then extends a bounding box
    downward to capture the table body using wide rules and table-like
    text lines.

    Args:
        lines: Text lines on the page.
        rules: Wide horizontal rules on the page.
        page_width: Page width in points.
        page_height: Page height in points.

    Returns:
        Candidate bounding boxes.
    """
    candidates: list[_BBox] = []
    for cap in lines:
        if not TABLE_CAPTION_RE.search(cap.text.strip()):
            continue
        scan_bottom = min(page_height, cap.bbox[3] + _CAPTION_SCAN_DEPTH)

        table_lines: list[_TextLine] = []
        last_bottom = cap.bbox[3]
        for ln in lines:
            if ln.bbox[1] <= cap.bbox[3] or ln.bbox[1] > scan_bottom:
                continue
            if table_lines and ln.bbox[1] - last_bottom > 95:
                break
            n_tokens = ln.text.split()
            num_ratio = sum(1 for t in n_tokens if _NUMBER_RE.search(t)) / max(1, len(n_tokens))
            n_x_bins = len(Counter(round(float(w[0]) / 12) for w in ln.words))
            first_word = n_tokens[0].strip(":.").lower() if n_tokens else ""
            is_table_like = (n_x_bins >= _ALIGNMENT_MIN_BINS or num_ratio >= 0.15)
            if is_table_like and first_word not in _BODY_WORDS:
                table_lines.append(ln)
                last_bottom = ln.bbox[3]
            elif table_lines and first_word in _BODY_WORDS:
                break

        wide_rules = [r for r in rules if cap.bbox[3] - 8 <= r[0] <= scan_bottom]

        if not wide_rules and not table_lines:
            continue

        if wide_rules:
            x0 = min(cap.bbox[0], min(r[1] for r in wide_rules)) - 8
            x1 = max(cap.bbox[2], max(r[2] for r in wide_rules)) + 8
        else:
            x0 = min(cap.bbox[0], min(ln.bbox[0] for ln in table_lines)) - 8
            x1 = max(cap.bbox[2], max(ln.bbox[2] for ln in table_lines), page_width * 0.55) + 8

        bottom = max(
            cap.bbox[3] + 120,
            max((ln.bbox[3] for ln in table_lines), default=0.0) + 24,
            max((r[0] for r in wide_rules), default=0.0) + 60,
        )
        candidates.append((
            max(0.0, x0), max(0.0, cap.bbox[1] - 4),
            min(page_width, x1), min(page_height, bottom),
        ))
    return candidates


# ── detection strategy: wide h-rule clusters ──────────────────────────────────


def _line_based_regions(
    rules: list[_Rule], page_width: float, page_height: float
) -> list[_BBox]:
    """Detect table regions from clusters of wide horizontal rules.

    Args:
        rules: Wide horizontal rules on the page.
        page_width: Page width in points.
        page_height: Page height in points.

    Returns:
        Candidate bounding boxes, one per rule cluster.
    """
    return [
        (
            max(0.0, min(r[1] for r in c) - 4),
            max(0.0, min(r[0] for r in c) - 16),
            min(page_width, max(r[2] for r in c) + 4),
            min(page_height, max(r[0] for r in c) + 100),
        )
        for c in _cluster_rules(rules)
    ]


# ── candidate management ───────────────────────────────────────────────────────


def _merge_candidates(candidates: list[_BBox]) -> list[_BBox]:
    """Merge substantially overlapping candidates by expanding to their union."""
    merged: list[_BBox] = []
    for bbox in candidates:
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        for i, existing in enumerate(merged):
            mutual = min(_overlap_ratio(bbox, existing), _overlap_ratio(existing, bbox))
            if _iou(bbox, existing) >= _REGION_MERGE_IOU or mutual >= _REGION_MERGE_IOU:
                merged[i] = (
                    min(existing[0], bbox[0]), min(existing[1], bbox[1]),
                    max(existing[2], bbox[2]), max(existing[3], bbox[3]),
                )
                break
        else:
            merged.append(bbox)
    return merged


# ── cell extraction ────────────────────────────────────────────────────────────


def _extract_table_cells(
    page: pymupdf.Page,
    bbox: _BBox,
    rules: list[_Rule],
) -> tuple[list[str], list[list[str]]]:
    """Extract header and data rows from a candidate table bbox.

    Header/data split logic:

    - ≥ 3 rules inside bbox → second rule (midrule) is the separator.
    - 2 rules → 25 % heuristic from the top.
    - 0–1 rules → midpoint between first and second word-row tops.

    When rules are present the clip top is clamped to the first rule so
    that caption text above the toprule is excluded from word extraction.

    Args:
        page: An open PyMuPDF page.
        bbox: Candidate table bounding box ``(x0, y0, x1, y1)``.
        rules: Wide horizontal rules on the page.

    Returns:
        A ``(header_row, data_rows)`` pair; both are empty on failure.
    """
    x0, y_top, x1, y_bottom = bbox
    in_rules = sorted(
        [r for r in rules if y_top - _CLIP_MARGIN_PT <= r[0] <= y_bottom + _CLIP_MARGIN_PT],
        key=lambda r: r[0],
    )
    clip_top = (in_rules[0][0] - _CLIP_MARGIN_PT) if in_rules else (y_top - _CLIP_MARGIN_PT)
    clip = pymupdf.Rect(
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
        y_tops = sorted({round(float(w[1])) for w in all_words})
        if len(y_tops) < 2:
            return [], []
        sep_y = (float(y_tops[0]) + float(y_tops[1])) / 2

    header_words = [w for w in all_words if float(w[3]) <= sep_y + _CLIP_MARGIN_PT]
    data_words = [w for w in all_words if float(w[1]) > sep_y - _CLIP_MARGIN_PT]
    if not header_words:
        return [], []

    col_ranges = _find_col_ranges(header_words) or _find_col_ranges(all_words)
    if len(col_ranges) < _MIN_COLS:
        return [], []

    n_cols = len(col_ranges)
    flat_header = [""] * n_cols
    for row in _words_to_grid(header_words, col_ranges):
        for ci, cell in enumerate(row):
            if cell:
                flat_header[ci] = (flat_header[ci] + " " + cell).strip()

    data_rows = _words_to_grid(data_words, col_ranges)
    return flat_header, merge_multiline_cells(data_rows)


# ── main entry point ───────────────────────────────────────────────────────────


def detect_borderless_tables(
    page: pymupdf.Page,
    page_num: int,
    page_width: float,
    dpi: float | None,
    pymupdf_pixmap_attrs: set[str],
    already_detected_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[ParsedMedia]:
    """Detect borderless (three-line / booktabs-style) tables.

    Applies two strategies — caption anchoring and wide horizontal-rule
    clustering — scores each candidate by multi-evidence confidence, and
    emits one ``ParsedMedia`` entry per table found
    (``type="table", detection_method="borderless"``).

    Candidates whose bounding box substantially overlaps an already-detected
    table (IoU ≥ ``_OVERLAP_IOU_THRESHOLD``) are skipped to avoid
    re-emitting tables that ``find_tables()`` already found.

    Args:
        page: An open PyMuPDF page.
        page_num: Zero-indexed page number; ``page_num + 1`` is stored in
            metadata.
        page_width: Page width in points.
        dpi: Optional rendering DPI forwarded to ``page.get_pixmap``.
        pymupdf_pixmap_attrs: Attribute names to copy from the rendered
            pixmap into ``info``.
        already_detected_bboxes: Bboxes of tables already detected by
            ``find_tables()``.

    Returns:
        A list of ``ParsedMedia`` entries for each detected borderless table.
    """
    already: list[_BBox] = list(already_detected_bboxes or [])
    page_height = float(page.rect.height)
    all_page_words: list[tuple] = page.get_text("words") or []
    lines = _build_text_lines(all_page_words)
    rules = _get_wide_h_rules(page, page_width)

    candidates: list[_BBox] = [
        *_caption_based_regions(lines, rules, page_width, page_height),
        *_line_based_regions(rules, page_width, page_height),
    ]
    clipped = [
        c for bbox in candidates
        if (c := _clip_to_page(bbox, page_width, page_height)) is not None
    ]
    all_scored = [
        (bbox, _score_region(lines, bbox, rules))
        for bbox in _merge_candidates(clipped)
    ]
    scored = sorted(
        [(b, s) for b, s in all_scored if s >= _REGION_MIN_KEEP_SCORE],
        key=lambda x: -x[1],
    )

    results: list[ParsedMedia] = []
    emitted: list[_BBox] = list(already)

    for bbox, _ in scored:
        if _overlaps_any(bbox, emitted) or bbox[3] - bbox[1] < _MIN_TABLE_HEIGHT_PT:
            continue
        header_row, data_rows = _extract_table_cells(page, bbox, rules)
        if not header_row or len(data_rows) < _MIN_DATA_ROWS:
            logger.debug(
                "Page %d: borderless candidate at y=[%.1f, %.1f] yielded no usable cells.",
                page_num + 1, bbox[1], bbox[3],
            )
            continue

        clip = pymupdf.Rect(
            bbox[0] - _CLIP_MARGIN_PT, bbox[1] - _CLIP_MARGIN_PT,
            bbox[2] + _CLIP_MARGIN_PT, bbox[3] + _CLIP_MARGIN_PT,
        )
        pix = page.get_pixmap(clip=clip, dpi=dpi)
        media_info: dict = {
            "bbox": (clip.x0, clip.y0, clip.x1, clip.y1),
            "type": "table",
            "detection_method": "borderless",
        } | {attr: _serialize_info_value(getattr(pix, attr)) for attr in pymupdf_pixmap_attrs}
        media_info["info_hashable"] = json.dumps(
            {
                k: (tuple(int(round(v)) for v in val) if k == "bbox" else val)
                for k, val in media_info.items()
                if isinstance(val, (int, float, str, bool, list, tuple)) or val is None
            },
            sort_keys=True,
        )
        media_info["page_num"] = page_num + 1
        results.append(ParsedMedia(
            index=len(already) + len(results),
            data=pix.tobytes(),
            text=clean_invalid_unicode(_to_markdown(header_row, data_rows)),
            info=media_info,
        ))
        emitted.append(bbox)
        logger.debug(
            "Page %d: detected borderless table at y=[%.1f, %.1f], %d col(s), %d data row(s).",
            page_num + 1, bbox[1], bbox[3], len(header_row), len(data_rows),
        )

    return results
