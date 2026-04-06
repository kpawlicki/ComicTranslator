"""
CLI entry point for ComicTranslator.

Usage::

    python main.py --input cover.jpg --output cover_es.jpg \
        --source en --target es

Run ``python main.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import sys

from comic_translator.pipeline import ComicTranslator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comic-translator",
        description=(
            "Detect text in an image, translate it into another language, "
            "and save the result as a new image."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Path to the source image (JPEG, PNG, …).",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Where the translated image will be saved.",
    )
    parser.add_argument(
        "--source",
        default="auto",
        metavar="LANG",
        help=(
            'Source language code, e.g. "en" or "ja". '
            'Use "auto" to detect automatically (default).'
        ),
    )
    parser.add_argument(
        "--target",
        default="en",
        metavar="LANG",
        help='Target language code, e.g. "es" or "fr" (default: "en").',
    )
    parser.add_argument(
        "--ocr-lang",
        action="append",
        dest="ocr_languages",
        metavar="LANG",
        default=None,
        help=(
            "Language(s) to pass to the OCR engine. "
            "Can be repeated for multilingual images. "
            'Default: ["en"].'
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU for OCR inference (requires CUDA).",
    )
    parser.add_argument(
        "--font",
        default=None,
        metavar="PATH",
        help="Path to a TrueType font file used when rendering translated text.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print each detected/translated text region to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print(f"[comic-translator] Loading models for OCR …")
    translator = ComicTranslator(
        source=args.source,
        target=args.target,
        ocr_languages=args.ocr_languages,
        gpu=args.gpu,
        font_path=args.font,
    )

    print(f"[comic-translator] Processing '{args.input}' → '{args.output}' …")
    results = translator.translate_image(args.input, args.output)

    if not results:
        print("[comic-translator] No text regions detected in the image.")
    else:
        print(f"[comic-translator] Replaced {len(results)} text region(s).")
        if args.verbose:
            for item in results:
                bbox_str = "({},{})–({},{})".format(*item["bbox"])
                print(f"  {bbox_str}  {item['original']!r}  →  {item['translated']!r}")

    print(f"[comic-translator] Saved translated image to '{args.output}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
