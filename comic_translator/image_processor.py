"""
Image-processing module — erase original text regions and render translated text.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

# Maximum font size tried when fitting text into a bounding box.
_MAX_FONT_SIZE = 40
# Minimum font size before giving up on fitting.
_MIN_FONT_SIZE = 6
# Fraction of the bounding-box border used to sample background colour.
_BORDER_SAMPLE_FRACTION = 0.15
# Candidate system font paths tried in order (cross-platform friendly).
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arial.ttf",
]


class ImageProcessor:
    """Erases text regions in an image and redraws them with translated text."""

    def __init__(self, font_path: str | None = None) -> None:
        """
        Args:
            font_path: Optional explicit path to a TrueType font file.  When
                       ``None`` the processor searches common system locations
                       and falls back to the built-in PIL bitmap font.
        """
        self._font_path = font_path or self._find_system_font()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def replace_text_regions(
        self,
        image_path: str,
        regions: list[dict],
        translated_texts: Sequence[str],
        output_path: str,
    ) -> None:
        """Overwrite each detected text region with its translation.

        Args:
            image_path: Path to the source image file.
            regions: List of region dicts (``{"bbox": (x1,y1,x2,y2), ...}``).
            translated_texts: Translations corresponding to *regions*.
            output_path: Where the processed image will be saved.
        """
        if len(regions) != len(translated_texts):
            raise ValueError(
                f"Number of regions ({len(regions)}) must match number of "
                f"translated texts ({len(translated_texts)})."
            )

        image = Image.open(image_path).convert("RGB")

        for region, translation in zip(regions, translated_texts):
            image = self._replace_single_region(image, region["bbox"], translation)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    # ------------------------------------------------------------------
    # Per-region helpers
    # ------------------------------------------------------------------

    def _replace_single_region(
        self,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        text: str,
    ) -> Image.Image:
        """Return a copy of *image* with *bbox* filled and *text* drawn."""
        x1, y1, x2, y2 = bbox
        bg_color = self._sample_background(image, bbox)

        draw = ImageDraw.Draw(image)
        # Erase original text by filling the bounding box with background colour.
        draw.rectangle([x1, y1, x2, y2], fill=bg_color)

        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        lines, font = self._fit_text(draw, text, box_w, box_h)

        # Choose a contrasting text colour.
        text_color = self._contrasting_color(bg_color)

        # Draw lines centred inside the bounding box.
        line_height = self._line_height(draw, font)
        total_text_h = line_height * len(lines)
        y_start = y1 + max((box_h - total_text_h) // 2, 0)

        for i, line in enumerate(lines):
            try:
                line_w = draw.textlength(line, font=font)
            except AttributeError:
                line_w = draw.textsize(line, font=font)[0]  # type: ignore[attr-defined]
            x_start = x1 + max((box_w - int(line_w)) // 2, 0)
            draw.text((x_start, y_start + i * line_height), line, font=font, fill=text_color)

        return image

    # ------------------------------------------------------------------
    # Text fitting
    # ------------------------------------------------------------------

    def _fit_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        box_w: int,
        box_h: int,
    ) -> tuple[list[str], ImageFont.ImageFont | ImageFont.FreeTypeFont]:
        """Return (wrapped_lines, font) such that the text fits in the box."""
        for font_size in range(_MAX_FONT_SIZE, _MIN_FONT_SIZE - 1, -1):
            font = self._load_font(font_size)
            lh = self._line_height(draw, font)
            if lh <= 0:
                continue
            max_lines = max(box_h // lh, 1)
            # Try progressively wider wrapping until the text fits vertically.
            for width_chars in range(max(len(text), 1), 0, -1):
                lines = textwrap.wrap(text, width=width_chars) or [text]
                if len(lines) <= max_lines and self._lines_fit_width(draw, lines, font, box_w):
                    return lines, font
        # Final fallback: single line at minimum size.
        return [text], self._load_font(_MIN_FONT_SIZE)

    @staticmethod
    def _lines_fit_width(
        draw: ImageDraw.ImageDraw,
        lines: list[str],
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        box_w: int,
    ) -> bool:
        for line in lines:
            try:
                w = draw.textlength(line, font=font)
            except AttributeError:
                w = draw.textsize(line, font=font)[0]  # type: ignore[attr-defined]
            if w > box_w:
                return False
        return True

    @staticmethod
    def _line_height(
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    ) -> int:
        try:
            bb = font.getbbox("Ag")
            return bb[3] - bb[1] + 2
        except AttributeError:
            pass
        try:
            _, h = draw.textsize("Ag", font=font)  # type: ignore[attr-defined]
            return h + 2
        except Exception:
            return 12

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_background(
        image: Image.Image,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int]:
        """Return the dominant colour along the border of *bbox*."""
        x1, y1, x2, y2 = bbox
        w = image.width
        h = image.height
        bx1 = max(x1, 0)
        by1 = max(y1, 0)
        bx2 = min(x2, w - 1)
        by2 = min(y2, h - 1)

        border_w = max(int((bx2 - bx1) * _BORDER_SAMPLE_FRACTION), 1)
        border_h = max(int((by2 - by1) * _BORDER_SAMPLE_FRACTION), 1)

        pixels: list[tuple[int, int, int]] = []
        region = image.crop((bx1, by1, bx2, by2)).convert("RGB")

        # Top and bottom strips.
        for strip_y in range(min(border_h, region.height)):
            for px in range(region.width):
                pixels.append(region.getpixel((px, strip_y)))  # type: ignore[arg-type]
        bottom_start = max(region.height - border_h, 0)
        for strip_y in range(bottom_start, region.height):
            for px in range(region.width):
                pixels.append(region.getpixel((px, strip_y)))  # type: ignore[arg-type]

        # Left and right strips.
        for px in range(min(border_w, region.width)):
            for py in range(region.height):
                pixels.append(region.getpixel((px, py)))  # type: ignore[arg-type]
        right_start = max(region.width - border_w, 0)
        for px in range(right_start, region.width):
            for py in range(region.height):
                pixels.append(region.getpixel((px, py)))  # type: ignore[arg-type]

        if not pixels:
            return (255, 255, 255)

        r = int(sum(p[0] for p in pixels) / len(pixels))
        g = int(sum(p[1] for p in pixels) / len(pixels))
        b = int(sum(p[2] for p in pixels) / len(pixels))
        return (r, g, b)

    @staticmethod
    def _contrasting_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
        """Return black or white depending on which contrasts more with *bg*."""
        luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        return (0, 0, 0) if luminance > 128 else (255, 255, 255)

    # ------------------------------------------------------------------
    # Font loading
    # ------------------------------------------------------------------

    def _load_font(
        self, size: int
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if self._font_path:
            try:
                return ImageFont.truetype(self._font_path, size)
            except (OSError, IOError):
                pass
        return ImageFont.load_default()

    @staticmethod
    def _find_system_font() -> str | None:
        for path in _FONT_CANDIDATES:
            if Path(path).exists():
                return path
        return None
