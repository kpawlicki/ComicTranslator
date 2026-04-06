"""
Pipeline module — orchestrates OCR → translation → image replacement.
"""

from __future__ import annotations

from .image_processor import ImageProcessor
from .ocr import OCREngine
from .translator import TextTranslator


class ComicTranslator:
    """End-to-end pipeline: detect text, translate it, and redraw the image.

    Example::

        translator = ComicTranslator(source="en", target="es")
        translator.translate_image("cover.jpg", "cover_es.jpg")
    """

    def __init__(
        self,
        source: str = "auto",
        target: str = "en",
        ocr_languages: list[str] | None = None,
        gpu: bool = False,
        font_path: str | None = None,
    ) -> None:
        """
        Args:
            source: Source language code for translation (``"auto"`` to detect).
            target: Target language code for translation.
            ocr_languages: Languages passed to EasyOCR for text detection.
                           Defaults to ``["en"]``.
            gpu: Whether EasyOCR should use a GPU.
            font_path: Optional path to a TrueType font used when rendering
                       translated text.
        """
        if ocr_languages is None:
            ocr_languages = ["en"]
        self._ocr = OCREngine(languages=ocr_languages, gpu=gpu)
        self._translator = TextTranslator(source=source, target=target)
        self._processor = ImageProcessor(font_path=font_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate_image(self, input_path: str, output_path: str) -> list[dict]:
        """Translate all text found in *input_path* and save to *output_path*.

        Returns:
            A list of dicts with keys ``"original"``, ``"translated"``, and
            ``"bbox"`` for each detected text region, so callers can inspect
            what was changed.
        """
        regions = self._ocr.detect(input_path)
        if not regions:
            # Nothing to do — copy the image unchanged.
            import shutil

            shutil.copy2(input_path, output_path)
            return []

        original_texts = [r["text"] for r in regions]
        translated_texts = self._translator.translate_batch(original_texts)

        self._processor.replace_text_regions(
            image_path=input_path,
            regions=regions,
            translated_texts=translated_texts,
            output_path=output_path,
        )

        return [
            {"original": orig, "translated": trans, "bbox": region["bbox"]}
            for orig, trans, region in zip(original_texts, translated_texts, regions)
        ]
