"""Load and chunk JSONL datasets (case law, legal acts, Q&A)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict

import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


class Chunk(TypedDict):
    text: str
    metadata: dict
    chunk_hash: str


_enc = tiktoken.get_encoding("cl100k_base")


def _split_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunks.append(_enc.decode(tokens[start:end]))
        start += max_tokens - overlap_tokens
    return chunks


def _make_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_text(record: dict) -> str:
    """Extract the most useful text from a JSONL record regardless of schema."""
    # Try common field combinations
    parts: list[str] = []

    # Case law: Text + Summary
    if "Text" in record and "Summary" in record:
        parts.append(f"Judgment:\n{record['Text']}")
        if record["Summary"].strip():
            parts.append(f"Summary:\n{record['Summary']}")
        return "\n\n".join(parts)

    # Legal acts: act_title + section + law
    if "act_title" in record and "law" in record:
        title = record.get("act_title", "")
        section = record.get("section", "")
        law = record.get("law", "")
        return f"{title} - Section {section}\n{law}"

    # Supreme court chunked: just text
    if "text" in record:
        return record["text"]

    # Court cases with petitioner/respondent
    if "petitioner" in record or "case_name" in record:
        parts = []
        for key in ["case_name", "petitioner", "respondent", "bench", "facts", "decision"]:
            if key in record and record[key]:
                parts.append(f"{key.replace('_', ' ').title()}: {record[key]}")
        return "\n".join(parts)

    # Q&A format: instruction/question + response/answer
    for q_key in ["instruction", "question", "input", "query", "premise"]:
        if q_key in record:
            q = record[q_key]
            for a_key in ["response", "answer", "output", "hypothesis", "label"]:
                if a_key in record:
                    return f"Question: {q}\nAnswer: {record[a_key]}"
            return q

    # Fallback: concatenate all string values
    text_parts = []
    for k, v in record.items():
        if isinstance(v, str) and len(v) > 10:
            text_parts.append(v)
    return "\n".join(text_parts)


def load_jsonl_chunks(
    file_path: str,
    source_name: str | None = None,
    max_records: int | None = None,
) -> list[Chunk]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    src = source_name or path.stem
    chunks: list[Chunk] = []
    record_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = _extract_text(record)
            if not text or len(text.strip()) < 20:
                continue

            sub_chunks = _split_text(text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
            for i, chunk_text in enumerate(sub_chunks):
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            "source": src,
                            "record_id": line_num,
                            "chunk_part": i,
                        },
                        chunk_hash=_make_hash(chunk_text),
                    )
                )

            record_count += 1
            if max_records and record_count >= max_records:
                break

    return chunks


def load_all_jsonl(directory: str, max_records_per_file: int | None = None) -> list[Chunk]:
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    all_chunks: list[Chunk] = []
    for jsonl_file in sorted(dir_path.glob("*.jsonl")):
        print(f"  Processing: {jsonl_file.name}")
        chunks = load_jsonl_chunks(str(jsonl_file), max_records=max_records_per_file)
        print(f"    → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    import sys
    base = Path(__file__).resolve().parent.parent.parent
    case_dir = base / "data" / "raw" / "case_law"
    qa_dir = base / "data" / "raw" / "qa_datasets"

    print("Case Law:")
    case_chunks = load_all_jsonl(str(case_dir), max_records_per_file=5)
    print(f"Total: {len(case_chunks)} chunks\n")

    print("QA Datasets:")
    qa_chunks = load_all_jsonl(str(qa_dir), max_records_per_file=5)
    print(f"Total: {len(qa_chunks)} chunks")
