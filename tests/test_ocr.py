"""Tests for the OCR engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from comic_translator.ocr import OCREngine


@pytest.fixture()
def mock_reader():
    """Return a mock easyocr.Reader."""
    reader = MagicMock()
    reader.readtext.return_value = [
        ([[10, 20], [100, 20], [100, 40], [10, 40]], "Hello", 0.99),
        ([[10, 50], [80, 50], [80, 70], [10, 70]], "  ", 0.95),  # whitespace-only
    ]
    return reader


@patch("comic_translator.ocr.easyocr", create=True)
def test_detect_returns_normalised_regions(mock_easyocr, mock_reader):
    mock_easyocr.Reader.return_value = mock_reader

    with patch.dict("sys.modules", {"easyocr": mock_easyocr}):
        engine = OCREngine(languages=["en"], gpu=False)
        engine._reader = mock_reader

    regions = engine.detect("fake_image.jpg")

    # Whitespace-only region should be filtered out.
    assert len(regions) == 1
    region = regions[0]
    assert region["text"] == "Hello"
    assert region["confidence"] == pytest.approx(0.99)
    assert region["bbox"] == (10, 20, 100, 40)


def test_parse_region_converts_polygon():
    raw = ([[5, 10], [50, 10], [50, 30], [5, 30]], "World", 0.85)
    region = OCREngine._parse_region(raw)
    assert region["bbox"] == (5, 10, 50, 30)
    assert region["text"] == "World"
    assert region["confidence"] == pytest.approx(0.85)


def test_parse_region_strips_whitespace():
    raw = ([[0, 0], [10, 0], [10, 10], [0, 10]], "  hi  ", 0.7)
    region = OCREngine._parse_region(raw)
    assert region["text"] == "hi"


def test_parse_region_handles_non_axis_aligned_polygon():
    # Rotated bounding box.
    raw = ([[10, 30], [30, 10], [50, 30], [30, 50]], "Ok", 0.9)
    region = OCREngine._parse_region(raw)
    assert region["bbox"] == (10, 10, 50, 50)
