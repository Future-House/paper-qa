from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest
from pydantic import ValidationError

from paperqa_nemotron.api import (
    NemotronParseAnnotatedBBox,
    NemotronParseBBox,
    NemotronParseClassification,
    NemotronParseMarkdown,
    NemotronParseMarkdownBBox,
    _call_nvidia_api,
    _call_sagemaker_api,
)

REPO_ROOT = Path(__file__).parents[3]
STUB_DATA_DIR = REPO_ROOT / "tests" / "stub_data"


class TestNemotronParseBBox:
    def test_bbox_validation(self) -> None:
        bbox = NemotronParseBBox(xmin=0.1, xmax=0.9, ymin=0.2, ymax=0.8)
        assert bbox.xmin == 0.1
        assert bbox.xmax == 0.9
        assert bbox.ymin == 0.2
        assert bbox.ymax == 0.8

        bbox_full = NemotronParseBBox(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        assert bbox_full.xmin == 0.0
        assert bbox_full.xmax == 1.0
        assert bbox_full.ymin == 0.0
        assert bbox_full.ymax == 1.0

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            NemotronParseBBox(xmin=-0.1, xmax=0.5, ymin=0.2, ymax=0.8)

        with pytest.raises(ValidationError, match="less than or equal to 1"):
            NemotronParseBBox(xmin=1.1, xmax=1.5, ymin=0.2, ymax=0.8)

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            NemotronParseBBox(xmin=0.1, xmax=-0.5, ymin=0.2, ymax=0.8)

        with pytest.raises(ValidationError, match="less than or equal to 1"):
            NemotronParseBBox(xmin=0.1, xmax=1.5, ymin=0.2, ymax=0.8)

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=-0.2, ymax=0.8)

        with pytest.raises(ValidationError, match="less than or equal to 1"):
            NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=1.2, ymax=1.8)

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.2, ymax=-0.8)

        with pytest.raises(ValidationError, match="less than or equal to 1"):
            NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.2, ymax=1.8)

        with pytest.raises(ValidationError, match="xmin must be less than xmax"):
            NemotronParseBBox(xmin=0.5, xmax=0.5, ymin=0.2, ymax=0.8)

        with pytest.raises(ValidationError, match="xmin must be less than xmax"):
            NemotronParseBBox(xmin=0.7, xmax=0.3, ymin=0.2, ymax=0.8)

        with pytest.raises(ValidationError, match="ymin must be less than ymax"):
            NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.5, ymax=0.5)

        with pytest.raises(ValidationError, match="ymin must be less than ymax"):
            NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.8, ymax=0.2)

    def test_bbox_to_page_coordinates(self) -> None:
        bbox = NemotronParseBBox(xmin=0.1, xmax=0.9, ymin=0.2, ymax=0.8)
        assert bbox.to_page_coordinates(height=1000, width=800) == pytest.approx(
            (80.0, 200.0, 720.0, 800.0)
        )
        # Also check different page dimensions
        assert bbox.to_page_coordinates(height=100, width=200) == pytest.approx(
            (20.0, 20.0, 180.0, 80.0)
        )

    def test_iou_identical_boxes(self) -> None:
        bbox1 = NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.2, ymax=0.6)
        bbox2 = NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.2, ymax=0.6)
        assert bbox1.iou(bbox2) == pytest.approx(1.0)

    def test_iou_no_overlap(self) -> None:
        bbox1 = NemotronParseBBox(xmin=0.0, xmax=0.2, ymin=0.0, ymax=0.2)
        bbox2 = NemotronParseBBox(xmin=0.5, xmax=0.7, ymin=0.5, ymax=0.7)
        assert bbox1.iou(bbox2) == pytest.approx(0.0)

    def test_iou_partial_overlap(self) -> None:
        # Two boxes with 50% horizontal overlap
        bbox1 = NemotronParseBBox(xmin=0.0, xmax=0.4, ymin=0.0, ymax=0.4)
        bbox2 = NemotronParseBBox(xmin=0.2, xmax=0.6, ymin=0.0, ymax=0.4)
        # Intersection: 0.2 * 0.4 = 0.08
        # Union: 0.16 + 0.16 - 0.08 = 0.24
        # IoU: 0.08 / 0.24 = 1/3
        assert bbox1.iou(bbox2) == pytest.approx(1 / 3)

    def test_iou_high_overlap(self) -> None:
        # Two boxes with very high overlap (simulating detection_only refinement)
        bbox1 = NemotronParseBBox(xmin=0.10, xmax=0.90, ymin=0.20, ymax=0.80)
        bbox2 = NemotronParseBBox(xmin=0.11, xmax=0.89, ymin=0.21, ymax=0.79)
        # Should be high IoU (> 0.90)
        assert bbox1.iou(bbox2) > 0.90

    def test_union_creates_superset(self) -> None:
        bbox1 = NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.2, ymax=0.6)
        bbox2 = NemotronParseBBox(xmin=0.3, xmax=0.7, ymin=0.1, ymax=0.5)
        union = bbox1.union(bbox2)
        # Union should be the smallest box containing both
        assert union.xmin == 0.1  # min of 0.1, 0.3
        assert union.xmax == 0.7  # max of 0.5, 0.7
        assert union.ymin == 0.1  # min of 0.2, 0.1
        assert union.ymax == 0.6  # max of 0.6, 0.5

    def test_union_identical_boxes(self) -> None:
        bbox1 = NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.2, ymax=0.6)
        bbox2 = NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.2, ymax=0.6)
        union = bbox1.union(bbox2)
        assert union == bbox1


class TestMergeWithDetection:
    def test_merge_empty_detection(self) -> None:
        markdown_results = [
            NemotronParseMarkdownBBox(
                bbox=NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.2, ymax=0.6),
                type=NemotronParseClassification.TEXT,
                text="Sample text",
            )
        ]
        merged = NemotronParseMarkdownBBox.merge_with_detection(
            markdown_results, [], iou_threshold=0.975
        )
        assert merged == markdown_results

    def test_merge_no_matching_type(self) -> None:
        markdown_results = [
            NemotronParseMarkdownBBox(
                bbox=NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.2, ymax=0.6),
                type=NemotronParseClassification.TEXT,
                text="Sample text",
            )
        ]
        detection_results = [
            NemotronParseAnnotatedBBox(
                bbox=NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.2, ymax=0.6),
                type=NemotronParseClassification.TABLE,  # Different type
            )
        ]
        merged = NemotronParseMarkdownBBox.merge_with_detection(
            markdown_results, detection_results, iou_threshold=0.975
        )
        assert merged[0].bbox == markdown_results[0].bbox

    def test_merge_below_threshold(self) -> None:
        markdown_results = [
            NemotronParseMarkdownBBox(
                bbox=NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.2, ymax=0.6),
                type=NemotronParseClassification.TEXT,
                text="Sample text",
            )
        ]
        detection_results = [
            NemotronParseAnnotatedBBox(
                bbox=NemotronParseBBox(xmin=0.3, xmax=0.7, ymin=0.4, ymax=0.8),
                type=NemotronParseClassification.TEXT,
            )
        ]
        merged = NemotronParseMarkdownBBox.merge_with_detection(
            markdown_results, detection_results, iou_threshold=0.975
        )
        # IoU is low, should keep original bbox
        assert merged[0].bbox == markdown_results[0].bbox

    def test_merge_above_threshold(self) -> None:
        markdown_bbox = NemotronParseBBox(xmin=0.10, xmax=0.90, ymin=0.20, ymax=0.80)
        detection_bbox = NemotronParseBBox(xmin=0.10, xmax=0.90, ymin=0.20, ymax=0.80)

        markdown_results = [
            NemotronParseMarkdownBBox(
                bbox=markdown_bbox,
                type=NemotronParseClassification.TEXT,
                text="Sample text",
            )
        ]
        detection_results = [
            NemotronParseAnnotatedBBox(
                bbox=detection_bbox, type=NemotronParseClassification.TEXT
            )
        ]
        merged = NemotronParseMarkdownBBox.merge_with_detection(
            markdown_results, detection_results, iou_threshold=0.975
        )
        # IoU is 1.0, superset bbox equals both (identical boxes)
        assert merged[0].bbox == markdown_bbox.union(detection_bbox)
        assert merged[0].text == "Sample text"

    def test_merge_creates_superset_bbox(self) -> None:
        # Markdown bbox is slightly smaller on one side
        markdown_bbox = NemotronParseBBox(xmin=0.11, xmax=0.90, ymin=0.20, ymax=0.80)
        # Detection bbox is slightly larger on one side
        detection_bbox = NemotronParseBBox(xmin=0.10, xmax=0.89, ymin=0.20, ymax=0.80)

        markdown_results = [
            NemotronParseMarkdownBBox(
                bbox=markdown_bbox,
                type=NemotronParseClassification.TEXT,
                text="Sample text",
            )
        ]
        detection_results = [
            NemotronParseAnnotatedBBox(
                bbox=detection_bbox, type=NemotronParseClassification.TEXT
            )
        ]
        # Verify IoU is above threshold
        assert markdown_bbox.iou(detection_bbox) > 0.95

        merged = NemotronParseMarkdownBBox.merge_with_detection(
            markdown_results, detection_results, iou_threshold=0.95
        )
        # Should create superset containing both bboxes
        expected_bbox = NemotronParseBBox(xmin=0.10, xmax=0.90, ymin=0.20, ymax=0.80)
        assert merged[0].bbox == expected_bbox
        assert merged[0].text == "Sample text"

    def test_merge_preserves_text(self) -> None:
        markdown_results = [
            NemotronParseMarkdownBBox(
                bbox=NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.2, ymax=0.6),
                type=NemotronParseClassification.TABLE,
                text="| col1 | col2 |\n|------|------|\n| a | b |",
            )
        ]
        detection_results = [
            NemotronParseAnnotatedBBox(
                bbox=NemotronParseBBox(xmin=0.1, xmax=0.5, ymin=0.2, ymax=0.6),
                type=NemotronParseClassification.TABLE,
            )
        ]
        merged = NemotronParseMarkdownBBox.merge_with_detection(
            markdown_results, detection_results, iou_threshold=0.975
        )
        # Text should be preserved from markdown_results
        assert merged[0].text == "| col1 | col2 |\n|------|------|\n| a | b |"

    def test_merge_multiple_items(self) -> None:
        markdown_results = [
            NemotronParseMarkdownBBox(
                bbox=NemotronParseBBox(xmin=0.0, xmax=0.4, ymin=0.0, ymax=0.3),
                type=NemotronParseClassification.TITLE,
                text="Title",
            ),
            NemotronParseMarkdownBBox(
                bbox=NemotronParseBBox(xmin=0.0, xmax=0.9, ymin=0.35, ymax=0.65),
                type=NemotronParseClassification.TEXT,
                text="Body text",
            ),
            NemotronParseMarkdownBBox(
                bbox=NemotronParseBBox(xmin=0.2, xmax=0.8, ymin=0.7, ymax=0.95),
                type=NemotronParseClassification.TABLE,
                text="Table content",
            ),
        ]
        # Detection results: only provide refinement for TEXT and TABLE
        detection_results = [
            NemotronParseAnnotatedBBox(
                bbox=NemotronParseBBox(xmin=0.0, xmax=0.9, ymin=0.35, ymax=0.65),
                type=NemotronParseClassification.TEXT,
            ),
            # TABLE bbox with very high IoU (near identical)
            NemotronParseAnnotatedBBox(
                bbox=NemotronParseBBox(xmin=0.2, xmax=0.8, ymin=0.7, ymax=0.95),
                type=NemotronParseClassification.TABLE,
            ),
        ]
        merged = NemotronParseMarkdownBBox.merge_with_detection(
            markdown_results, detection_results, iou_threshold=0.90
        )

        assert len(merged) == 3
        # Title: no matching detection, keeps original
        assert merged[0].bbox == markdown_results[0].bbox
        assert merged[0].text == "Title"
        # TEXT: perfect match, superset bbox (identical in this case)
        assert merged[1].bbox == markdown_results[1].bbox.union(
            detection_results[0].bbox
        )
        assert merged[1].text == "Body text"
        # TABLE: perfect IoU match, superset bbox (identical in this case)
        assert merged[2].bbox == markdown_results[2].bbox.union(
            detection_results[1].bbox
        )
        assert merged[2].text == "Table content"


@pytest.fixture(name="pdf_page_np")
def fixture_pdf_page_np() -> np.ndarray:
    pdf_doc = pdfium.PdfDocument(STUB_DATA_DIR / "pasa.pdf")
    # nemotron-parse's markdown_no_bbox tool will start
    # babbling \\n if using default scale=1
    page_np = pdf_doc[0].render(scale=2).to_numpy()
    assert page_np.shape == (1684, 1191, 3), "Expected particular page size"
    return page_np


class TestNvidiaAPI:

    @pytest.mark.vcr
    @pytest.mark.parametrize("temperature", [0, 1])
    @pytest.mark.asyncio
    async def test_markdown_bbox(
        self, pdf_page_np: np.ndarray, temperature: float
    ) -> None:
        response = await _call_nvidia_api(
            pdf_page_np, tool_name="markdown_bbox", temperature=temperature
        )
        assert response
        for r in response:
            assert isinstance(r, NemotronParseMarkdownBBox)
            assert isinstance(r.bbox, NemotronParseBBox)
            assert r.type
            assert r.text

    @pytest.mark.vcr
    @pytest.mark.parametrize("temperature", [0, 1])
    @pytest.mark.asyncio
    async def test_markdown_no_bbox(
        self, pdf_page_np: np.ndarray, temperature: float
    ) -> None:
        response = await _call_nvidia_api(
            pdf_page_np, tool_name="markdown_no_bbox", temperature=temperature
        )
        assert response
        for r in response:
            assert isinstance(r, NemotronParseMarkdown)
            assert r.text

    @pytest.mark.vcr
    @pytest.mark.parametrize("temperature", [0, 1])
    @pytest.mark.asyncio
    async def test_detection_only(
        self, pdf_page_np: np.ndarray, temperature: float
    ) -> None:
        response = await _call_nvidia_api(
            pdf_page_np, tool_name="detection_only", temperature=temperature
        )
        assert response
        for r in response:
            assert isinstance(r, NemotronParseAnnotatedBBox)
            assert isinstance(r.bbox, NemotronParseBBox)
            assert r.type


@pytest.mark.skip(reason="Uncomment to test with AWS SageMaker")
class TestSageMakerAPI:

    @pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
    @pytest.mark.parametrize("temperature", [0, 1])
    @pytest.mark.asyncio
    async def test_markdown_bbox(
        self, pdf_page_np: np.ndarray, temperature: float
    ) -> None:
        response = await _call_sagemaker_api(
            pdf_page_np, tool_name="markdown_bbox", temperature=temperature
        )
        assert response
        for r in response:
            assert isinstance(r, NemotronParseMarkdownBBox)
            assert isinstance(r.bbox, NemotronParseBBox)
            assert r.type
            assert r.text

    @pytest.mark.parametrize("temperature", [0, 1])
    @pytest.mark.asyncio
    async def test_markdown_no_bbox(
        self, pdf_page_np: np.ndarray, temperature: float
    ) -> None:
        response = await _call_sagemaker_api(
            pdf_page_np, tool_name="markdown_no_bbox", temperature=temperature
        )
        assert response
        for r in response:
            assert isinstance(r, NemotronParseMarkdown)
            assert r.text

    @pytest.mark.parametrize("temperature", [0, 1])
    @pytest.mark.asyncio
    async def test_detection_only(
        self, pdf_page_np: np.ndarray, temperature: float
    ) -> None:
        response = await _call_sagemaker_api(
            pdf_page_np, tool_name="detection_only", temperature=temperature
        )
        assert response
        for r in response:
            assert isinstance(r, NemotronParseAnnotatedBBox)
            assert isinstance(r.bbox, NemotronParseBBox)
            assert r.type
