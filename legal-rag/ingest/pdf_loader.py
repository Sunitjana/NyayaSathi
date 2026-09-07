"""Extract text from PDFs using PyMuPDF (fitz) and chunk by tokens."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypedDict

import fitz  # PyMuPDF
import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PDF_DIR, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


class Chunk(TypedDict):
    text: str
    metadata: dict
    chunk_hash: str


_enc = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_enc.encode(text))


def _split_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(_enc.decode(chunk_tokens))
        start += max_tokens - overlap_tokens
    return chunks


def _make_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_pdf_chunks(pdf_path: str) -> list[Chunk]:
    filename = Path(pdf_path).name
    doc = fitz.open(pdf_path)
    chunks: list[Chunk] = []
    global_chunk_id = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if not text:
            continue

        sub_chunks = _split_text(text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)

        for text_chunk in sub_chunks:
            chunk_hash = _make_hash(text_chunk)
            chunks.append(
                Chunk(
                    text=text_chunk,
                    metadata={
                        "source": filename,
                        "page": page_num + 1,
                        "chunk_id": global_chunk_id,
                    },
                    chunk_hash=chunk_hash,
                )
            )
            global_chunk_id += 1

    doc.close()
    return chunks


def load_all_pdfs(pdf_dir: str | None = None) -> list[Chunk]:
    directory = Path(pdf_dir or PDF_DIR)
    if not directory.exists():
        raise FileNotFoundError(f"PDF directory not found: {directory}")

    all_chunks: list[Chunk] = []
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return all_chunks

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        try:
            chunks = extract_pdf_chunks(str(pdf_path))
            print(f"  → {len(chunks)} chunks extracted")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ⚠ Skipping {pdf_path.name}: {e}")

    return all_chunks


if __name__ == "__main__":
    results = load_all_pdfs()
    print(f"\nTotal PDF chunks: {len(results)}")
    if results:
        print(f"Sample chunk:\n{results[0]['text'][:200]}...")
