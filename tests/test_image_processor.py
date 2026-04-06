"""Tests for the ImageProcessor."""

from __future__ import annotations

import os
import tempfile

import pytest
from PIL import Image

from comic_translator.image_processor import ImageProcessor


@pytest.fixture()
def small_white_image(tmp_path):
    """Create a small white PNG image for testing."""
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    path = str(tmp_path / "test_image.png")
    img.save(path)
    return path


@pytest.fixture()
def processor():
    return ImageProcessor()


class TestSampleBackground:
    def test_white_image_returns_white(self, processor):
        img = Image.new("RGB", (100, 50), (255, 255, 255))
        color = processor._sample_background(img, (10, 10, 90, 40))
        assert color == (255, 255, 255)

    def test_black_image_returns_black(self, processor):
        img = Image.new("RGB", (100, 50), (0, 0, 0))
        color = processor._sample_background(img, (10, 10, 90, 40))
        assert color == (0, 0, 0)

    def test_single_pixel_bbox(self, processor):
        img = Image.new("RGB", (100, 50), (128, 64, 32))
        # Should not raise for tiny bounding box.
        color = processor._sample_background(img, (5, 5, 6, 6))
        assert len(color) == 3


class TestContrastingColor:
    def test_white_bg_gives_black_text(self):
        assert ImageProcessor._contrasting_color((255, 255, 255)) == (0, 0, 0)

    def test_black_bg_gives_white_text(self):
        assert ImageProcessor._contrasting_color((0, 0, 0)) == (255, 255, 255)

    def test_mid_grey_gives_black_text(self):
        assert ImageProcessor._contrasting_color((200, 200, 200)) == (0, 0, 0)


class TestReplaceTextRegions:
    def test_replaces_text_in_image(self, processor, small_white_image, tmp_path):
        out = str(tmp_path / "out.png")
        regions = [{"bbox": (10, 10, 190, 50)}]
        processor.replace_text_regions(
            small_white_image, regions, ["Hola mundo"], out
        )
        assert os.path.exists(out)
        result = Image.open(out)
        assert result.size == (200, 100)

    def test_mismatched_regions_and_texts_raises(self, processor, small_white_image, tmp_path):
        out = str(tmp_path / "out.png")
        regions = [{"bbox": (10, 10, 90, 50)}, {"bbox": (10, 60, 90, 90)}]
        with pytest.raises(ValueError, match="must match"):
            processor.replace_text_regions(
                small_white_image, regions, ["Solo uno"], out
            )

    def test_creates_output_directory(self, processor, small_white_image, tmp_path):
        out = str(tmp_path / "sub" / "dir" / "out.png")
        processor.replace_text_regions(
            small_white_image, [{"bbox": (0, 0, 50, 30)}], ["Hi"], out
        )
        assert os.path.exists(out)

    def test_no_regions_saves_unchanged_image(self, processor, small_white_image, tmp_path):
        out = str(tmp_path / "unchanged.png")
        processor.replace_text_regions(small_white_image, [], [], out)
        assert os.path.exists(out)
        result = Image.open(out)
        assert result.size == (200, 100)

    def test_long_text_does_not_raise(self, processor, small_white_image, tmp_path):
        out = str(tmp_path / "long.png")
        long_text = "This is a very long translated sentence that may not fit."
        processor.replace_text_regions(
            small_white_image, [{"bbox": (0, 0, 60, 20)}], [long_text], out
        )
        assert os.path.exists(out)
