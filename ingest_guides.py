"""Chunk data/telecom_guide.pdf (600 chars, 100 overlap) into the `guides` collection (FR-16)."""
import re

from langchain_core.documents import Document
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, rebuild_collection

COLLECTION = "guides"


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Fixed-size character chunks with overlap, nudged back to a word boundary."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_docs():
    docs, ids = [], []
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        reader = PdfReader(pdf)
        n = 0
        for page_no, page in enumerate(reader.pages, start=1):
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            for chunk in chunk_text(text):
                n += 1
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": "GUIDE",
                            "ref": f"{pdf.stem} p.{page_no} #{n}",
                            "title": chunk[:80].rsplit(" ", 1)[0] + "…",
                            "page": page_no,
                        },
                    )
                )
                ids.append(f"guide-{pdf.stem}-{n}")
    return docs, ids


def main():
    docs, ids = load_docs()
    rebuild_collection(COLLECTION, docs, ids)
    print(f"[guides] ingested {len(docs)} guide chunks")


if __name__ == "__main__":
    main()
