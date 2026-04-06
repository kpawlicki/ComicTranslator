"""
OCR module — detect text regions in images using EasyOCR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

TextRegion = dict  # {"bbox": (x1, y1, x2, y2), "text": str, "confidence": float}


class OCREngine:
    """Wraps EasyOCR to detect text regions in an image."""

    def __init__(self, languages: list[str] | None = None, gpu: bool = False) -> None:
        """
        Args:
            languages: List of language codes to detect (e.g. ``["en"]``).
                       Defaults to English only.
            gpu: Whether to use a GPU for inference.
        """
        if languages is None:
            languages = ["en"]
        import easyocr  # imported lazily so tests can mock it easily

        self._reader = easyocr.Reader(languages, gpu=gpu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image_path: str) -> list[TextRegion]:
        """Return text regions found in *image_path*.

        Each region is a dict with keys:
        - ``bbox``: ``(x1, y1, x2, y2)`` in pixel coordinates.
        - ``text``: The detected text string.
        - ``confidence``: Detection confidence in ``[0, 1]``.
        """
        raw = self._reader.readtext(image_path)
        return [self._parse_region(r) for r in raw if r[1].strip()]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_region(raw_result: tuple) -> TextRegion:
        """Convert an EasyOCR result tuple into a normalised dict."""
        polygon, text, confidence = raw_result
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        return {"bbox": bbox, "text": text.strip(), "confidence": float(confidence)}
