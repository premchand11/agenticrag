from io import BytesIO

import pymupdf
import pytest
from app.ingestion.parsers.extract import ExtractionError, extract_document
from docx import Document as DocxDocument


def test_extract_html_skips_script_and_style():
    content = b"<html><style>p{}</style><script>secret</script><p>Redis caches data</p></html>"
    extracted = extract_document("html", content)
    assert extracted.text == "Redis caches data"
    assert "secret" not in extracted.text
    assert extracted.pages[0].number == 1


def test_extract_docx_reads_paragraphs():
    document = DocxDocument()
    document.add_paragraph("Redis caches data")
    buffer = BytesIO()
    document.save(buffer)
    extracted = extract_document("docx", buffer.getvalue())
    assert extracted.text == "Redis caches data"


def test_extract_pdf_reads_each_page():
    pdf = pymupdf.open()
    try:
        pdf.new_page().insert_text((72, 72), "Page one")
        pdf.new_page().insert_text((72, 72), "Page two")
        content = pdf.tobytes()
    finally:
        pdf.close()
    extracted = extract_document("pdf", content)
    assert [page.text for page in extracted.pages] == ["Page one", "Page two"]
    assert "Page one" in extracted.text and "Page two" in extracted.text


def test_extract_html_rejects_a_blank_document():
    with pytest.raises(ExtractionError):
        extract_document("html", b"<html><style>p{}</style></html>")
