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

    @pytest.mark.parametrize(
        ("bbox1_coords", "bbox2_coords", "expected_iou"),
        [
            # Identical boxes -> IoU = 1.0
            ((0.1, 0.5, 0.2, 0.6), (0.1, 0.5, 0.2, 0.6), 1.0),
            # No overlap -> IoU = 0.0
            ((0.0, 0.2, 0.0, 0.2), (0.5, 0.7, 0.5, 0.7), 0.0),
            # Partial overlap (50% horizontal) -> IoU = 1/3
            ((0.0, 0.4, 0.0, 0.4), (0.2, 0.6, 0.0, 0.4), 1 / 3),
        ],
        ids=["identical", "no_overlap", "partial_overlap"],
    )
    def test_iou(
        self,
        bbox1_coords: tuple[float, float, float, float],
        bbox2_coords: tuple[float, float, float, float],
        expected_iou: float,
    ) -> None:
        bbox1 = NemotronParseBBox.from_coordinates(bbox1_coords)
        bbox2 = NemotronParseBBox.from_coordinates(bbox2_coords)
        assert bbox1.iou(bbox2) == pytest.approx(expected_iou)

    def test_union(self) -> None:
        # Test superset creation with different boxes
        bbox1 = NemotronParseBBox.from_coordinates((0.1, 0.5, 0.2, 0.6))
        bbox2 = NemotronParseBBox.from_coordinates((0.3, 0.7, 0.1, 0.5))
        union = bbox1.union(bbox2)
        assert (union.xmin, union.xmax, union.ymin, union.ymax) == (0.1, 0.7, 0.1, 0.6)

        # Test identical boxes return same bbox
        bbox3 = NemotronParseBBox.from_coordinates((0.1, 0.5, 0.2, 0.6))
        assert bbox1.union(bbox3) == bbox1


class TestMergeWithDetection:
    def test_merge_empty_inputs(self) -> None:
        """Test edge cases with empty markdown or detection results."""
        sample_bbox = NemotronParseBBox.from_coordinates((0.1, 0.5, 0.2, 0.6))

        # Empty detection -> returns markdown as-is
        markdown_results = [
            NemotronParseMarkdownBBox(
                bbox=sample_bbox, type=NemotronParseClassification.TEXT, text="Sample"
            )
        ]
        merged = NemotronParseMarkdownBBox.merge_with_detection(
            markdown_results, [], iou_threshold=0.975
        )
        assert merged == markdown_results

        # Empty markdown -> detection included with text=None
        detection_results = [
            NemotronParseAnnotatedBBox(
                bbox=sample_bbox, type=NemotronParseClassification.TEXT
            )
        ]
        merged = NemotronParseMarkdownBBox.merge_with_detection(
            [], detection_results, iou_threshold=0.975
        )
        assert len(merged) == 1
        assert merged[0].bbox == sample_bbox
        assert merged[0].text is None

    def test_merge_unmatched_detection(self) -> None:
        """Test that unmatched detections are added (different location, type, or low IoU)."""
        markdown_bbox = NemotronParseBBox.from_coordinates((0.1, 0.5, 0.2, 0.6))
        markdown_results = [
            NemotronParseMarkdownBBox(
                bbox=markdown_bbox,
                type=NemotronParseClassification.TEXT,
                text="Sample text",
            )
        ]

        # Different location (no overlap)
        detection_far = NemotronParseAnnotatedBBox(
            bbox=NemotronParseBBox.from_coordinates((0.6, 0.9, 0.7, 0.95)),
            type=NemotronParseClassification.PICTURE,
        )
        merged = NemotronParseMarkdownBBox.merge_with_detection(
            markdown_results, [detection_far], iou_threshold=0.975
        )
        assert len(merged) == 2
        assert merged[0].text == "Sample text"
        assert merged[1].bbox == detection_far.bbox
        assert merged[1].text is None

        # Same bbox but different type
        detection_wrong_type = NemotronParseAnnotatedBBox(
            bbox=markdown_bbox, type=NemotronParseClassification.TABLE
        )
        merged = NemotronParseMarkdownBBox.merge_with_detection(
            markdown_results, [detection_wrong_type], iou_threshold=0.975
        )
        assert len(merged) == 2
        assert merged[1].type == NemotronParseClassification.TABLE
        assert merged[1].text is None

        # Same type but IoU below threshold
        detection_low_iou = NemotronParseAnnotatedBBox(
            bbox=NemotronParseBBox.from_coordinates((0.3, 0.7, 0.4, 0.8)),
            type=NemotronParseClassification.TEXT,
        )
        merged = NemotronParseMarkdownBBox.merge_with_detection(
            markdown_results, [detection_low_iou], iou_threshold=0.975
        )
        assert len(merged) == 2
        assert merged[0].bbox == markdown_bbox  # Original preserved
        assert merged[1].text is None  # Unmatched detection

    def test_merge_matched_detection(self) -> None:
        """Test successful merging: superset bbox, text preserved."""
        # Slightly different bboxes that should merge
        markdown_bbox = NemotronParseBBox.from_coordinates((0.11, 0.90, 0.20, 0.80))
        detection_bbox = NemotronParseBBox.from_coordinates((0.10, 0.89, 0.20, 0.80))
        assert markdown_bbox.iou(detection_bbox) > 0.95  # Verify high IoU

        markdown_results = [
            NemotronParseMarkdownBBox(
                bbox=markdown_bbox,
                type=NemotronParseClassification.TABLE,
                text="| col1 | col2 |\n| a | b |",
            )
        ]
        detection_results = [
            NemotronParseAnnotatedBBox(
                bbox=detection_bbox, type=NemotronParseClassification.TABLE
            )
        ]

        merged = NemotronParseMarkdownBBox.merge_with_detection(
            markdown_results, detection_results, iou_threshold=0.95
        )
        assert len(merged) == 1
        # Superset bbox contains both
        assert merged[0].bbox == NemotronParseBBox.from_coordinates(
            (0.10, 0.90, 0.20, 0.80)
        )
        # Text preserved from markdown
        assert merged[0].text == "| col1 | col2 |\n| a | b |"

    def test_merge_multiple_items(self) -> None:
        markdown_results = [
            NemotronParseMarkdownBBox(
                bbox=NemotronParseBBox.from_coordinates((0.0, 0.4, 0.0, 0.3)),
                type=NemotronParseClassification.TITLE,
                text="Title",
            ),
            NemotronParseMarkdownBBox(
                bbox=NemotronParseBBox.from_coordinates((0.0, 0.9, 0.35, 0.65)),
                type=NemotronParseClassification.TEXT,
                text="Body text",
            ),
            NemotronParseMarkdownBBox(
                bbox=NemotronParseBBox.from_coordinates((0.2, 0.8, 0.7, 0.95)),
                type=NemotronParseClassification.TABLE,
                text="Table content",
            ),
        ]
        # Detection results: only provide refinement for TEXT and TABLE
        detection_results = [
            NemotronParseAnnotatedBBox(
                bbox=NemotronParseBBox.from_coordinates((0.0, 0.9, 0.35, 0.65)),
                type=NemotronParseClassification.TEXT,
            ),
            # TABLE bbox with very high IoU (near identical)
            NemotronParseAnnotatedBBox(
                bbox=NemotronParseBBox.from_coordinates((0.2, 0.8, 0.7, 0.95)),
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
