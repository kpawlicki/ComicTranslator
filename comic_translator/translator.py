"""
Translation module — translate text strings using deep-translator.
"""

from __future__ import annotations


class TextTranslator:
    """Wraps GoogleTranslator from deep-translator for text translation."""

    def __init__(self, source: str = "auto", target: str = "en") -> None:
        """
        Args:
            source: BCP-47 language code of the source language, or ``"auto"``
                    to detect it automatically.
            target: BCP-47 language code of the desired output language.
        """
        from deep_translator import GoogleTranslator  # lazy import for testability

        self._translator = GoogleTranslator(source=source, target=target)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str) -> str:
        """Return *text* translated into the target language.

        Returns the original *text* unchanged when it is empty or contains
        only whitespace.
        """
        if not text or not text.strip():
            return text
        result = self._translator.translate(text)
        return result if result is not None else text

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Return each string in *texts* translated into the target language."""
        return [self.translate(t) for t in texts]
