"""Embed text chunks and store them in Neon pgvector via batch PostgreSQL inserts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

from config import BASE_DIR, DATABASE_URL, EMBEDDING_MODEL
from ingest.csv_loader import load_csv_chunks
from ingest.pdf_loader import load_all_pdfs
from ingest.jsonl_loader import load_all_jsonl

CASE_LAW_DIR = str(BASE_DIR / "data" / "raw" / "case_law")
QA_DATASETS_DIR = str(BASE_DIR / "data" / "raw" / "qa_datasets")


def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def get_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not set in .env file.")
    return psycopg2.connect(DATABASE_URL)


def _get_existing_hashes(conn) -> set[str]:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_hash FROM legal_chunks")
            return {row[0] for row in cur.fetchall()}
    except Exception:
        conn.rollback()
        return set()


def ingest_chunks(
    conn,
    chunks: list[dict],
    model: SentenceTransformer,
    collection_name: str,
    batch_size: int = 500,
) -> int:
    existing_hashes = _get_existing_hashes(conn)
    new_chunks = [c for c in chunks if c["chunk_hash"] not in existing_hashes]

    if not new_chunks:
        print(f"  No new chunks to ingest (all {len(chunks)} already exist)")
        return 0

    total = len(new_chunks)
    ingested = 0

    for i in range(0, total, batch_size):
        batch = new_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=128).tolist()

        rows = []
        for chunk, emb in zip(batch, embeddings):
            meta = {**chunk["metadata"]}
            for k, v in meta.items():
                if isinstance(v, (list, dict)):
                    meta[k] = str(v)

            rows.append((
                chunk["text"],
                str(emb),
                meta.get("source", collection_name),
                collection_name,
                chunk["chunk_hash"],
                json.dumps(meta),
            ))

        with conn.cursor() as cur:
            execute_values(
                cur,
                """INSERT INTO legal_chunks (text, embedding, source, collection, chunk_hash, metadata)
                   VALUES %s
                   ON CONFLICT (chunk_hash) DO NOTHING""",
                rows,
                page_size=500,
            )
        conn.commit()
        ingested += len(batch)
        pct = int(ingested / total * 100)
        print(f"  [{pct:3d}%] Ingested {ingested:,}/{total:,} chunks")

    return ingested


def run_ingestion():
    print("=" * 60)
    print("LEGAL RAG — FULL DATA INGESTION (Neon pgvector)")
    print("=" * 60)

    model = get_embedding_model()
    conn = get_db_connection()

    results = {}

    try:
        print("\n[1/4] Loading CSV dataset...")
        csv_chunks = load_csv_chunks()
        print(f"  Total CSV chunks: {len(csv_chunks)}")
        print("  Embedding and storing...")
        results["csv"] = ingest_chunks(conn, csv_chunks, model, "legal_qa")

        print("\n[2/4] Loading PDF documents...")
        pdf_chunks = load_all_pdfs()
        print(f"  Total PDF chunks: {len(pdf_chunks)}")
        print("  Embedding and storing...")
        results["pdf"] = ingest_chunks(conn, pdf_chunks, model, "legal_docs")

        print("\n[3/4] Loading case law datasets...")
        case_chunks = load_all_jsonl(CASE_LAW_DIR)
        print(f"  Total case law chunks: {len(case_chunks)}")
        if case_chunks:
            print("  Embedding and storing...")
            results["case_law"] = ingest_chunks(conn, case_chunks, model, "legal_case_law")
        else:
            results["case_law"] = 0

        print("\n[4/4] Loading QA & statutes datasets...")
        statute_chunks = load_all_jsonl(QA_DATASETS_DIR)
        print(f"  Total statute/QA chunks: {len(statute_chunks)}")
        if statute_chunks:
            print("  Embedding and storing...")
            results["statutes"] = ingest_chunks(conn, statute_chunks, model, "legal_statutes")
        else:
            results["statutes"] = 0

        print("\n" + "=" * 60)
        print("INGESTION SUMMARY")
        total_new = sum(results.values())
        for source, count in results.items():
            print(f"  {source:15s}: {count:,} new chunks")
        print(f"  {'TOTAL':15s}: {total_new:,} new chunks")

        print(f"\n  Collection counts:")
        with conn.cursor() as cur:
            for coll_name in ["legal_qa", "legal_docs", "legal_case_law", "legal_statutes"]:
                cur.execute(
                    "SELECT COUNT(*) FROM legal_chunks WHERE collection = %s",
                    (coll_name,),
                )
                count = cur.fetchone()[0]
                print(f"    {coll_name:20s}: {count:,} total chunks")
        print("=" * 60)

    finally:
        conn.close()


if __name__ == "__main__":
    run_ingestion()
