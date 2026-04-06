"""Tests for the end-to-end ComicTranslator pipeline."""

from __future__ import annotations

import shutil
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


@pytest.fixture()
def sample_image(tmp_path):
    """A small white PNG image."""
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    path = str(tmp_path / "sample.png")
    img.save(path)
    return path


def _make_pipeline(ocr_results=None, translation_fn=None):
    """Build a ComicTranslator with all external dependencies mocked."""
    from comic_translator.pipeline import ComicTranslator

    pipeline = ComicTranslator.__new__(ComicTranslator)

    mock_ocr = MagicMock()
    mock_ocr.detect.return_value = ocr_results or []

    mock_translator = MagicMock()
    if translation_fn:
        mock_translator.translate_batch.side_effect = lambda texts: [
            translation_fn(t) for t in texts
        ]
    else:
        mock_translator.translate_batch.return_value = []

    from comic_translator.image_processor import ImageProcessor

    pipeline._ocr = mock_ocr
    pipeline._translator = mock_translator
    pipeline._processor = ImageProcessor()
    return pipeline


class TestTranslateImage:
    def test_returns_empty_list_when_no_text_detected(self, sample_image, tmp_path):
        pipeline = _make_pipeline(ocr_results=[])
        out = str(tmp_path / "out.png")
        results = pipeline.translate_image(sample_image, out)
        assert results == []
        # Image should still be copied.
        import os

        assert os.path.exists(out)

    def test_copies_image_when_no_text_detected(self, sample_image, tmp_path):
        pipeline = _make_pipeline(ocr_results=[])
        out = str(tmp_path / "out.png")
        pipeline.translate_image(sample_image, out)
        original = Image.open(sample_image)
        result = Image.open(out)
        assert original.size == result.size

    def test_returns_translation_results(self, sample_image, tmp_path):
        regions = [
            {"bbox": (10, 10, 100, 40), "text": "Hello", "confidence": 0.99},
            {"bbox": (10, 50, 100, 80), "text": "World", "confidence": 0.95},
        ]
        pipeline = _make_pipeline(
            ocr_results=regions,
            translation_fn=lambda t: t.upper(),
        )
        pipeline._translator.translate_batch.return_value = ["HELLO", "WORLD"]
        out = str(tmp_path / "out.png")
        results = pipeline.translate_image(sample_image, out)

        assert len(results) == 2
        assert results[0]["original"] == "Hello"
        assert results[0]["translated"] == "HELLO"
        assert results[0]["bbox"] == (10, 10, 100, 40)
        assert results[1]["original"] == "World"
        assert results[1]["translated"] == "WORLD"

    def test_output_image_saved(self, sample_image, tmp_path):
        import os

        regions = [{"bbox": (10, 10, 190, 50), "text": "Comic", "confidence": 0.98}]
        pipeline = _make_pipeline(ocr_results=regions)
        pipeline._translator.translate_batch.return_value = ["Cómic"]
        out = str(tmp_path / "translated.png")
        pipeline.translate_image(sample_image, out)
        assert os.path.exists(out)

    def test_ocr_called_with_input_path(self, sample_image, tmp_path):
        pipeline = _make_pipeline(ocr_results=[])
        out = str(tmp_path / "out.png")
        pipeline.translate_image(sample_image, out)
        pipeline._ocr.detect.assert_called_once_with(sample_image)
