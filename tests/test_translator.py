"""Tests for the TextTranslator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from comic_translator.translator import TextTranslator


@pytest.fixture()
def mock_google_translator():
    gt = MagicMock()
    gt.translate.side_effect = lambda text: f"[translated]{text}"
    return gt


def _make_translator(mock_gt) -> TextTranslator:
    with patch("comic_translator.translator.GoogleTranslator", create=True, return_value=mock_gt):
        import comic_translator.translator as mod

        with patch.dict(
            "sys.modules",
            {"deep_translator": MagicMock(GoogleTranslator=MagicMock(return_value=mock_gt))},
        ):
            t = TextTranslator.__new__(TextTranslator)
            t._translator = mock_gt
    return t


def test_translate_returns_translation(mock_google_translator):
    t = _make_translator(mock_google_translator)
    result = t.translate("Hello")
    assert result == "[translated]Hello"
    mock_google_translator.translate.assert_called_once_with("Hello")


def test_translate_empty_string_unchanged(mock_google_translator):
    t = _make_translator(mock_google_translator)
    result = t.translate("")
    assert result == ""
    mock_google_translator.translate.assert_not_called()


def test_translate_whitespace_only_unchanged(mock_google_translator):
    t = _make_translator(mock_google_translator)
    result = t.translate("   ")
    assert result == "   "
    mock_google_translator.translate.assert_not_called()


def test_translate_returns_original_when_none_returned():
    gt = MagicMock()
    gt.translate.return_value = None
    t = _make_translator(gt)
    result = t.translate("Hello")
    assert result == "Hello"


def test_translate_batch(mock_google_translator):
    t = _make_translator(mock_google_translator)
    results = t.translate_batch(["foo", "bar"])
    assert results == ["[translated]foo", "[translated]bar"]
    assert mock_google_translator.translate.call_count == 2
