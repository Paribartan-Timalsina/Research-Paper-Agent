from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from app.core.exceptions import InvalidFileError, PDFParseError
from app.services.llm_service import llm


@dataclass
class ExtractedPDF:
    title: str
    text: str
    # (page_number_1_indexed, description) — populated only when with_figures=True
    figures: list[tuple[int, str]] = field(default_factory=list)


_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_MIN_IMG_BYTES = 5_000  # skip tiny logos / decorative glyphs

_FIGURE_PROMPT = (
    "This is a figure from a research paper. In 1-2 sentences, describe "
    "what it shows (e.g. architecture diagram, results chart, sample outputs). "
    "Be concrete. If it's a plot, name the axes and the trend."
)


def _clean(text: str) -> str:
    # Postgres TEXT columns reject NUL bytes; some PDFs embed them in text streams.
    text = text.replace("\x00", "")
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _image_bytes(doc: fitz.Document, xref: int) -> bytes | None:
    try:
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha >= 4:  # CMYK -> RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
        return pix.tobytes("png")
    except Exception:
        return None


def _describe_page_figures(doc: fitz.Document, page: fitz.Page) -> list[str]:
    descs: list[str] = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        data = _image_bytes(doc, xref)
        if not data or len(data) < _MIN_IMG_BYTES:
            continue
        try:
            caption = llm.describe_image(data, _FIGURE_PROMPT)
        except NotImplementedError:
            return []
        except Exception:
            continue
        if caption:
            descs.append(caption.strip())
    return descs


def extract_pdf(
    content: bytes,
    fallback_title: str,
    with_figures: bool = False,
) -> ExtractedPDF:
    """Extract text + best-guess title from PDF bytes.

    When `with_figures` is True, embedded images are sent to the configured
    VLM and the descriptions are inlined as `[FIGURE on page N]: ...`.

    Raises:
        InvalidFileError: empty payload.
        PDFParseError: PyMuPDF failed or no text was extracted.
    """
    if not content:
        raise InvalidFileError("Empty file payload")

    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        raise PDFParseError(f"PyMuPDF failed to open: {e}") from e

    pages: list[str] = []
    figures: list[tuple[int, str]] = []
    guessed_title: str | None = None

    try:
        with doc:
            for i in range(doc.page_count):
                page = doc.load_page(i)
                page_text: str = page.get_text("text") or ""
                if i == 0:
                    for line in page_text.splitlines():
                        line = line.strip()
                        if len(line) > 8 and not line.lower().startswith(("abstract", "http")):
                            guessed_title = line
                            break
                if with_figures:
                    for desc in _describe_page_figures(doc, page):
                        page_text += f"\n\n[FIGURE on page {i+1}]: {desc}"
                        figures.append((i + 1, desc))
                pages.append(page_text)
    except Exception as e:
        raise PDFParseError(f"PyMuPDF failed: {e}") from e

    cleaned = _clean("\n\n".join(pages))
    if not cleaned:
        raise PDFParseError("No text could be extracted from the PDF")

    return ExtractedPDF(
        title=(guessed_title or fallback_title)[:500],
        text=cleaned,
        figures=figures,
    )


def chunk_text(text: str, max_chars: int = 4000, overlap: int = 200) -> list[str]:
    """Simple character chunker with overlap. Good enough for MVP retrieval."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks
