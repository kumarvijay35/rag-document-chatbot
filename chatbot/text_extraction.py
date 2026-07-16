"""
chatbot/text_extraction.py

Text extraction now lives on the Django side (it's file handling, which is
Django's responsibility). The FastAPI service receives plain text only —
it never touches uploaded files.

requirements: pypdf  (you almost certainly already have it via LangChain's
PyPDFLoader; add `pypdf` explicitly to requirements.txt anyway)
"""

import os

from pypdf import PdfReader


class TextExtractionError(Exception):
    pass


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    elif ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise TextExtractionError(f"Could not read PDF: {exc}") from exc

    else:
        raise TextExtractionError(f"Unsupported file type: {ext}")

    text = text.strip()
    if not text:
        raise TextExtractionError(
            "No extractable text found (scanned/image-only PDFs are not supported)"
        )
    return text
