import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pypdfium2 as pdfium
import pytest
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import SegmentedPdfPage
from PIL import Image
from pytest_subtests import SubTests

from paperqa_docling.nemotron_parse import (
    NemotronParseDocumentDetectionOnlyBackend,
    NemotronParseDocumentMarkdownBackend,
    NemotronParseDocumentMarkdownBBoxBackend,
    NemotronParsePageBackend,
    NvidiaAnnotatedBBox,
    NvidiaBBox,
    NvidiaClassification,
    NvidiaMarkdown,
    NvidiaMarkdownBBox,
    _call_nemotron_parse_api,
)
from paperqa_docling.reader import parse_pdf_to_pages

REPO_ROOT = Path(__file__).parents[3]
STUB_DATA_DIR = REPO_ROOT / "tests" / "stub_data"


@pytest.fixture
def sample_pdf_document() -> pdfium.PdfDocument:
    """Fixture providing a sample PDF document."""
    return pdfium.PdfDocument(STUB_DATA_DIR / "pasa.pdf")


@pytest.fixture
def sample_page_backend_markdown_bbox(
    sample_pdf_document: pdfium.PdfDocument,
) -> NemotronParsePageBackend:
    """Fixture providing a page backend with markdown_bbox tool."""
    return NemotronParsePageBackend(
        ppage=sample_pdf_document[0], tool_name="markdown_bbox"
    )


@pytest.fixture
def sample_page_backend_markdown_no_bbox(
    sample_pdf_document: pdfium.PdfDocument,
) -> NemotronParsePageBackend:
    """Fixture providing a page backend with markdown_no_bbox tool."""
    return NemotronParsePageBackend(
        ppage=sample_pdf_document[0], tool_name="markdown_no_bbox"
    )


@pytest.fixture
def sample_page_backend_detection_only(
    sample_pdf_document: pdfium.PdfDocument,
) -> NemotronParsePageBackend:
    """Fixture providing a page backend with detection_only tool."""
    return NemotronParsePageBackend(
        ppage=sample_pdf_document[0], tool_name="detection_only"
    )


class TestPageBackendBasics:
    """Test basic page backend functionality."""

    def test_init_markdown_bbox(
        self, sample_page_backend_markdown_bbox: NemotronParsePageBackend
    ) -> None:
        """Test page backend initialization with markdown_bbox."""
        assert sample_page_backend_markdown_bbox.valid
        assert sample_page_backend_markdown_bbox._tool_name == "markdown_bbox"
        assert sample_page_backend_markdown_bbox._dpage is None

    def test_init_markdown_no_bbox(
        self, sample_page_backend_markdown_no_bbox: NemotronParsePageBackend
    ) -> None:
        """Test page backend initialization with markdown_no_bbox."""
        assert sample_page_backend_markdown_no_bbox.valid
        assert sample_page_backend_markdown_no_bbox._tool_name == "markdown_no_bbox"
        assert sample_page_backend_markdown_no_bbox._dpage is None

    def test_init_detection_only(
        self, sample_page_backend_detection_only: NemotronParsePageBackend
    ) -> None:
        """Test page backend initialization with detection_only."""
        assert sample_page_backend_detection_only.valid
        assert sample_page_backend_detection_only._tool_name == "detection_only"
        assert sample_page_backend_detection_only._dpage is None

    def test_is_valid(
        self, sample_page_backend_markdown_bbox: NemotronParsePageBackend
    ) -> None:
        """Test page backend is_valid method."""
        assert sample_page_backend_markdown_bbox.is_valid()

    def test_get_size(
        self, sample_page_backend_markdown_bbox: NemotronParsePageBackend
    ) -> None:
        """Test page backend get_size method."""
        size = sample_page_backend_markdown_bbox.get_size()
        assert isinstance(size, Size)
        assert size.width > 0
        assert size.height > 0

    def test_get_page_image(
        self, sample_page_backend_markdown_bbox: NemotronParsePageBackend
    ) -> None:
        """Test page backend get_page_image method."""
        image = sample_page_backend_markdown_bbox.get_page_image()
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"

    def test_get_page_image_with_cropbox(
        self, sample_page_backend_markdown_bbox: NemotronParsePageBackend
    ) -> None:
        """Test page backend get_page_image with cropbox."""
        cropbox = BoundingBox(
            l=0,
            r=100,
            t=0,
            b=100,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        image = sample_page_backend_markdown_bbox.get_page_image(
            scale=1.0, cropbox=cropbox
        )
        assert isinstance(image, Image.Image)

    def test_unload(
        self, sample_page_backend_markdown_bbox: NemotronParsePageBackend
    ) -> None:
        """Test page backend unload."""
        sample_page_backend_markdown_bbox.unload()
        assert sample_page_backend_markdown_bbox._ppage is None
        assert sample_page_backend_markdown_bbox._unloaded


class TestParseMarkdownBBox:
    """Test parsing with markdown_bbox tool."""

    @patch("paperqa_docling.nemotron_parse._call_nemotron_parse_api")
    def test_ensure_parsed(
        self,
        mock_api_call: MagicMock,
        sample_page_backend_markdown_bbox: NemotronParsePageBackend,
    ) -> None:
        """Test page backend _ensure_parsed method with markdown_bbox."""
        # Mock API response with proper format
        mock_api_call.return_value = [
            NvidiaMarkdownBBox(
                bbox=NvidiaBBox(xmin=10, xmax=100, ymin=10, ymax=30),
                type=NvidiaClassification.TITLE,
                text="# Title",
            ),
            NvidiaMarkdownBBox(
                bbox=NvidiaBBox(xmin=10, xmax=100, ymin=40, ymax=60),
                type=NvidiaClassification.SECTION_HEADER,
                text="Content here",
            ),
        ]

        sample_page_backend_markdown_bbox._ensure_parsed()

        assert sample_page_backend_markdown_bbox._dpage is not None
        assert isinstance(sample_page_backend_markdown_bbox._dpage, SegmentedPdfPage)
        assert len(sample_page_backend_markdown_bbox._dpage.textline_cells) == 2
        # Check markdown is preserved
        assert (
            "# Title" in sample_page_backend_markdown_bbox._dpage.textline_cells[0].text
        )

    @patch("paperqa_docling.nemotron_parse._call_nemotron_parse_api")
    def test_get_segmented_page(
        self,
        mock_api_call: MagicMock,
        sample_page_backend_markdown_bbox: NemotronParsePageBackend,
    ) -> None:
        """Test getting segmented page."""
        mock_api_call.return_value = [
            NvidiaMarkdownBBox(
                bbox=NvidiaBBox(xmin=10, xmax=100, ymin=10, ymax=30),
                type=NvidiaClassification.TITLE,
                text="# Test",
            ),
        ]

        result = sample_page_backend_markdown_bbox.get_segmented_page()

        assert result is not None
        assert isinstance(result, SegmentedPdfPage)

    @patch("paperqa_docling.nemotron_parse._call_nemotron_parse_api")
    def test_get_text_cells(
        self,
        mock_api_call: MagicMock,
        sample_page_backend_markdown_bbox: NemotronParsePageBackend,
    ) -> None:
        """Test getting text cells."""
        mock_api_call.return_value = [
            NvidiaMarkdownBBox(
                bbox=NvidiaBBox(xmin=10, xmax=100, ymin=10, ymax=30),
                type=NvidiaClassification.TITLE,
                text="# Test",
            ),
            NvidiaMarkdownBBox(
                bbox=NvidiaBBox(xmin=10, xmax=100, ymin=40, ymax=60),
                type=NvidiaClassification.SECTION_HEADER,
                text="Content",
            ),
        ]

        cells = list(sample_page_backend_markdown_bbox.get_text_cells())

        assert len(cells) == 2

    @patch("paperqa_docling.nemotron_parse._call_nemotron_parse_api")
    def test_get_text_in_rect(
        self,
        mock_api_call: MagicMock,
        sample_page_backend_markdown_bbox: NemotronParsePageBackend,
    ) -> None:
        """Test getting text in rectangle."""
        mock_api_call.return_value = [
            NvidiaMarkdownBBox(
                bbox=NvidiaBBox(xmin=0, xmax=500, ymin=0, ymax=100),
                type=NvidiaClassification.TITLE,
                text="# Test",
            ),
            NvidiaMarkdownBBox(
                bbox=NvidiaBBox(xmin=0, xmax=500, ymin=110, ymax=200),
                type=NvidiaClassification.SECTION_HEADER,
                text="Content here",
            ),
        ]

        bbox = BoundingBox(
            l=0,
            r=500,
            t=0,
            b=100,
            coord_origin=CoordOrigin.TOPLEFT,
        )

        text = sample_page_backend_markdown_bbox.get_text_in_rect(bbox)
        assert isinstance(text, str)
        assert "# Test" in text


class TestParseMarkdownNoBBox:
    """Test parsing with markdown_no_bbox tool."""

    @patch("paperqa_docling.nemotron_parse._call_nemotron_parse_api")
    def test_ensure_parsed(
        self,
        mock_api_call: MagicMock,
        sample_page_backend_markdown_no_bbox: NemotronParsePageBackend,
    ) -> None:
        """Test page backend _ensure_parsed method with markdown_no_bbox."""
        mock_api_call.return_value = [
            NvidiaMarkdown(text="# Title"),
            NvidiaMarkdown(text="Content here"),
        ]

        sample_page_backend_markdown_no_bbox._ensure_parsed()

        assert sample_page_backend_markdown_no_bbox._dpage is not None
        assert isinstance(sample_page_backend_markdown_no_bbox._dpage, SegmentedPdfPage)
        assert len(sample_page_backend_markdown_no_bbox._dpage.textline_cells) == 2

    @patch("paperqa_docling.nemotron_parse._call_nemotron_parse_api")
    def test_empty_response(
        self,
        mock_api_call: MagicMock,
        sample_page_backend_markdown_no_bbox: NemotronParsePageBackend,
    ) -> None:
        """Test handling empty response."""
        mock_api_call.return_value = []

        sample_page_backend_markdown_no_bbox._ensure_parsed()

        assert sample_page_backend_markdown_no_bbox._dpage is not None
        assert len(sample_page_backend_markdown_no_bbox._dpage.textline_cells) == 0

    @patch("paperqa_docling.nemotron_parse._call_nemotron_parse_api")
    def test_skip_empty_text(
        self,
        mock_api_call: MagicMock,
        sample_page_backend_markdown_no_bbox: NemotronParsePageBackend,
    ) -> None:
        """Test skipping empty text items."""
        mock_api_call.return_value = [
            NvidiaMarkdown(text="# Title"),
            NvidiaMarkdown(text=""),
            NvidiaMarkdown(text="   "),
            NvidiaMarkdown(text="Content here"),
        ]

        sample_page_backend_markdown_no_bbox._ensure_parsed()

        # Type narrowing for mypy
        assert sample_page_backend_markdown_no_bbox._dpage is not None
        assert len(sample_page_backend_markdown_no_bbox._dpage.textline_cells) == 2


class TestParseDetectionOnly:
    """Test parsing with detection_only tool."""

    @patch("paperqa_docling.nemotron_parse._call_nemotron_parse_api")
    def test_ensure_parsed(
        self,
        mock_api_call: MagicMock,
        sample_page_backend_detection_only: NemotronParsePageBackend,
    ) -> None:
        """Test page backend _ensure_parsed method with detection_only."""
        mock_api_call.return_value = [
            NvidiaAnnotatedBBox(
                bbox=NvidiaBBox(xmin=10, xmax=100, ymin=10, ymax=30),
                type=NvidiaClassification.TITLE,
            ),
            NvidiaAnnotatedBBox(
                bbox=NvidiaBBox(xmin=10, xmax=100, ymin=40, ymax=60),
                type=NvidiaClassification.TABLE,
            ),
        ]

        sample_page_backend_detection_only._ensure_parsed()

        assert sample_page_backend_detection_only._dpage is not None
        assert isinstance(sample_page_backend_detection_only._dpage, SegmentedPdfPage)
        assert len(sample_page_backend_detection_only._dpage.textline_cells) == 2
        # Check that classification types are in the text
        assert (
            "[Title]"
            in sample_page_backend_detection_only._dpage.textline_cells[0].text
        )
        assert (
            "[Table]"
            in sample_page_backend_detection_only._dpage.textline_cells[1].text
        )

    @patch("paperqa_docling.nemotron_parse._call_nemotron_parse_api")
    def test_get_bitmap_rects(
        self,
        mock_api_call: MagicMock,
        sample_page_backend_detection_only: NemotronParsePageBackend,
    ) -> None:
        """Test getting bitmap rectangles."""
        mock_api_call.return_value = []
        sample_page_backend_detection_only._ensure_parsed()
        rects = list(sample_page_backend_detection_only.get_bitmap_rects())
        assert not rects


class TestDocumentBackends:
    """Test document-level backends."""

    def test_backend_markdown_bbox(self, subtests: SubTests) -> None:
        """Test markdown_bbox backend."""
        in_doc = InputDocument(
            path_or_stream=STUB_DATA_DIR / "pasa.pdf",
            format=InputFormat.PDF,
            backend=NemotronParseDocumentMarkdownBBoxBackend,
        )
        backend = in_doc._backend
        # Type narrowing for mypy
        assert isinstance(backend, NemotronParseDocumentMarkdownBBoxBackend)

        with subtests.test("validity"):
            assert backend.page_count() > 0
            assert backend.is_valid()

        with subtests.test("load"):
            page_backend = backend.load_page(0)
            assert isinstance(page_backend, NemotronParsePageBackend)
            assert page_backend._tool_name == "markdown_bbox"
            assert page_backend.valid

    def test_backend_markdown_no_bbox(self, subtests: SubTests) -> None:
        """Test markdown_no_bbox backend."""
        in_doc = InputDocument(
            path_or_stream=STUB_DATA_DIR / "pasa.pdf",
            format=InputFormat.PDF,
            backend=NemotronParseDocumentMarkdownBackend,
        )
        backend = in_doc._backend
        assert isinstance(backend, NemotronParseDocumentMarkdownBackend)

        with subtests.test("validity"):
            assert backend.page_count() > 0
            assert backend.is_valid()

        with subtests.test("load"):
            page_backend = backend.load_page(0)
            assert isinstance(page_backend, NemotronParsePageBackend)
            assert page_backend._tool_name == "markdown_no_bbox"
            assert page_backend.valid

    def test_backend_detection_only(self, subtests: SubTests) -> None:
        """Test detection_only backend."""
        in_doc = InputDocument(
            path_or_stream=STUB_DATA_DIR / "pasa.pdf",
            format=InputFormat.PDF,
            backend=NemotronParseDocumentDetectionOnlyBackend,
        )
        backend = in_doc._backend
        assert isinstance(backend, NemotronParseDocumentDetectionOnlyBackend)

        with subtests.test("validity"):
            assert backend.page_count() > 0
            assert backend.is_valid()

        with subtests.test("load"):
            page_backend = backend.load_page(0)
            assert isinstance(page_backend, NemotronParsePageBackend)
            assert page_backend._tool_name == "detection_only"
            assert page_backend.valid

    def test_rejects_bad_pdf(self, tmp_path) -> None:
        """Test rejection of invalid PDF."""
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_text("not a pdf")

        # Docling catches the PdfiumError and logs it, so the document
        # should be created but be in an invalid state
        doc = InputDocument(
            path_or_stream=bad_pdf,
            format=InputFormat.PDF,
            backend=NemotronParseDocumentMarkdownBBoxBackend,
        )
        # The backend should be None or the document should have page_count=0
        assert doc.page_count == 0 or not doc._backend.is_valid()


# Tests that actually call the API
class TestRealAPICall:
    """Tests that actually invoke the Nvidia API."""

    @pytest.mark.skipif(
        "NVIDIA_API_KEY" not in os.environ,
        reason="NVIDIA_API_KEY not set",
    )
    def test_call_api_markdown_bbox(self) -> None:
        """Test calling the API with markdown_bbox tool."""
        # Create a simple test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = _call_nemotron_parse_api(test_image, tool_name="markdown_bbox")

        assert isinstance(result, list)
        # Result should be a list of NvidiaMarkdownBBox objects
        for item in result:
            assert isinstance(item, NvidiaMarkdownBBox)
            assert hasattr(item, "bbox")
            assert hasattr(item, "text")
            assert hasattr(item, "type")

    @pytest.mark.skipif(
        "NVIDIA_API_KEY" not in os.environ,
        reason="NVIDIA_API_KEY not set",
    )
    def test_call_api_markdown_no_bbox(self) -> None:
        """Test calling the API with markdown_no_bbox tool."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = _call_nemotron_parse_api(test_image, tool_name="markdown_no_bbox")

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, NvidiaMarkdown)
            assert hasattr(item, "text")

    @pytest.mark.skipif(
        "NVIDIA_API_KEY" not in os.environ,
        reason="NVIDIA_API_KEY not set",
    )
    def test_call_api_detection_only(self) -> None:
        """Test calling the API with detection_only tool."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = _call_nemotron_parse_api(test_image, tool_name="detection_only")

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, NvidiaAnnotatedBBox)
            assert hasattr(item, "bbox")
            assert hasattr(item, "type")

    @pytest.mark.skipif(
        "NVIDIA_API_KEY" not in os.environ,
        reason="NVIDIA_API_KEY not set",
    )
    def test_real_pdf_page_markdown_bbox(
        self, sample_pdf_document: pdfium.PdfDocument
    ) -> None:
        """Test parsing a real PDF page with markdown_bbox."""
        backend = NemotronParsePageBackend(
            ppage=sample_pdf_document[0], tool_name="markdown_bbox"
        )

        backend._ensure_parsed()

        assert backend._dpage is not None
        assert isinstance(backend._dpage, SegmentedPdfPage)
        # Should have some text cells
        assert len(backend._dpage.textline_cells) > 0

    @pytest.mark.skipif(
        "NVIDIA_API_KEY" not in os.environ,
        reason="NVIDIA_API_KEY not set",
    )
    def test_real_pdf_page_markdown_no_bbox(
        self, sample_pdf_document: pdfium.PdfDocument
    ) -> None:
        """Test parsing a real PDF page with markdown_no_bbox."""
        backend = NemotronParsePageBackend(
            ppage=sample_pdf_document[0], tool_name="markdown_no_bbox"
        )

        backend._ensure_parsed()

        assert backend._dpage is not None
        assert isinstance(backend._dpage, SegmentedPdfPage)
        assert len(backend._dpage.textline_cells) > 0

    @pytest.mark.skipif(
        "NVIDIA_API_KEY" not in os.environ,
        reason="NVIDIA_API_KEY not set",
    )
    def test_real_pdf_page_detection_only(
        self, sample_pdf_document: pdfium.PdfDocument
    ) -> None:
        """Test parsing a real PDF page with detection_only."""
        backend = NemotronParsePageBackend(
            ppage=sample_pdf_document[0], tool_name="detection_only"
        )

        backend._ensure_parsed()

        assert backend._dpage is not None
        assert isinstance(backend._dpage, SegmentedPdfPage)
        # detection_only should have some results
        assert len(backend._dpage.textline_cells) > 0


@patch("paperqa_docling.nemotron_parse._call_nemotron_parse_api")
def test_nemotron_parse_integration_reader(
    mock_api_call: MagicMock,
) -> None:
    """Test integration with reader.parse_pdf_to_pages."""
    # Mock API response
    mock_api_call.return_value = [
        NvidiaMarkdownBBox(
            bbox=NvidiaBBox(xmin=10, xmax=100, ymin=10, ymax=30),
            type=NvidiaClassification.TITLE,
            text="# Test Document\n\nContent here",
        ),
    ]

    parsed_text = parse_pdf_to_pages(
        STUB_DATA_DIR / "pasa.pdf",
        backend=NemotronParseDocumentMarkdownBBoxBackend,
        parse_media=False,  # Skip media parsing for faster test
    )

    assert parsed_text.metadata.parsing_libraries
    # Just check that docling is present; the backend name is in the metadata.name
    assert any("docling" in lib for lib in parsed_text.metadata.parsing_libraries)
    assert parsed_text.metadata.name is not None
    assert "NemotronParseDocumentMarkdownBBoxBackend" in parsed_text.metadata.name
