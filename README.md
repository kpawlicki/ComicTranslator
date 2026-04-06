# ComicTranslator

A Python tool that detects text in images and replaces it with text in another language.  
It is designed for translating comics, manga, and any other image-based content where text is embedded directly in the artwork.

---

## How it works

1. **OCR** — [EasyOCR](https://github.com/JaidedAI/EasyOCR) locates every text region in the image and returns its bounding box and content.
2. **Translation** — [deep-translator](https://github.com/nidhaloff/deep-translator) (Google Translate back-end) translates each detected string into the target language.
3. **Image rendering** — [Pillow](https://python-pillow.org/) erases the original text by filling the bounding box with the sampled background colour, then draws the translated text scaled to fit the box.

---

## Requirements

- Python 3.9+
- The packages listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

> **Note:** EasyOCR downloads its language model on first use (~100 MB for English).

---

## Usage

### Command-line interface

```bash
python main.py --input cover.jpg --output cover_es.jpg --source en --target es
```

All options:

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | *(required)* | Source image (JPEG, PNG, …) |
| `--output PATH` | *(required)* | Destination path for the translated image |
| `--source LANG` | `auto` | Source language code or `auto` for auto-detection |
| `--target LANG` | `en` | Target language code |
| `--ocr-lang LANG` | `en` | Language(s) passed to EasyOCR (repeatable) |
| `--gpu` | off | Use a CUDA GPU for OCR inference |
| `--font PATH` | *(auto)* | Path to a TrueType font file for rendering |
| `--verbose` | off | Print each detected/translated region |

### Python API

```python
from comic_translator import ComicTranslator

translator = ComicTranslator(source="en", target="es")
results = translator.translate_image("page.png", "page_es.png")

for item in results:
    print(item["bbox"], repr(item["original"]), "→", repr(item["translated"]))
```

---

## Project structure

```
comic_translator/
    __init__.py          # Public API (ComicTranslator)
    ocr.py               # Text detection via EasyOCR
    translator.py        # Language translation via deep-translator
    image_processor.py   # Erase + redraw text regions using Pillow
    pipeline.py          # Orchestrates OCR → translate → render
main.py                  # CLI entry point
tests/
    test_ocr.py
    test_translator.py
    test_image_processor.py
    test_pipeline.py
requirements.txt
```

---

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```
