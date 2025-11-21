from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest
from pydantic import ValidationError

from paperqa_nemotron.api import (
    NemotronParseAnnotatedBBox,
    NemotronParseBBox,
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
