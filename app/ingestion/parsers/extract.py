from dataclasses import dataclass
from io import BytesIO

import pymupdf
from bs4 import BeautifulSoup
from docx import Document as DocxDocument


class ExtractionError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail = detail


@dataclass(frozen=True)
class ExtractedPage:
    number: int
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    pages: tuple[ExtractedPage, ...]


def extract_document(file_type: str, content: bytes) -> ExtractedDocument:
    parsers = {"pdf": extract_pdf, "docx": extract_docx, "html": extract_html}
    try:
        parser = parsers[file_type]
    except KeyError as exc:
        raise ExtractionError(f"unsupported file type: {file_type}") from exc
    return parser(content)


def extract_html(content: bytes) -> ExtractedDocument:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return _from_pages((ExtractedPage(1, text),))


def extract_docx(content: bytes) -> ExtractedDocument:
    document = DocxDocument(BytesIO(content))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return _from_pages((ExtractedPage(1, "\n".join(paragraphs)),))


def extract_pdf(content: bytes) -> ExtractedDocument:
    pdf = pymupdf.open(stream=content, filetype="pdf")
    try:
        pages = tuple(ExtractedPage(index, page.get_text().strip()) for index, page in enumerate(pdf, start=1))
    finally:
        pdf.close()
    return _from_pages(pages)


def _from_pages(pages: tuple[ExtractedPage, ...]) -> ExtractedDocument:
    text = "\n\n".join(page.text for page in pages if page.text)
    if not text:
        raise ExtractionError("no text extracted")
    return ExtractedDocument(text=text, pages=pages)
