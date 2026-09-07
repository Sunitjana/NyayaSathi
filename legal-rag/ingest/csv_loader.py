"""Load and chunk the Indian Law QA CSV dataset."""

from __future__ import annotations

import hashlib
from typing import TypedDict

import pandas as pd
import tiktoken

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CSV_PATH, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


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


def load_csv_chunks(csv_path: str | None = None) -> list[Chunk]:
    path = csv_path or CSV_PATH
    df = pd.read_csv(path)

    required = {"Instruction", "Response"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")

    chunks: list[Chunk] = []

    for idx, row in df.iterrows():
        instruction = str(row["Instruction"]).strip()
        response = str(row["Response"]).strip()
        if not instruction or not response:
            continue

        combined = f"Question: {instruction}\nAnswer: {response}"

        sub_chunks = _split_text(combined, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)

        for i, text in enumerate(sub_chunks):
            chunk_hash = _make_hash(text)
            chunks.append(
                Chunk(
                    text=text,
                    metadata={
                        "source": "indian-law-dataset",
                        "row_id": int(idx),
                        "instruction": instruction[:200],
                        "chunk_part": i,
                    },
                    chunk_hash=chunk_hash,
                )
            )

    return chunks


if __name__ == "__main__":
    results = load_csv_chunks()
    print(f"Loaded {len(results)} chunks from CSV")
    if results:
        print(f"Sample chunk:\n{results[0]['text'][:200]}...")
