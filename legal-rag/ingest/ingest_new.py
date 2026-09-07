"""Fast ingestion for new PDFs and JSONL datasets via direct PostgreSQL. Skips already-populated collections."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer

from config import BASE_DIR, EMBEDDING_MODEL
from ingest.embedder import get_db_connection, ingest_chunks
from ingest.pdf_loader import load_all_pdfs
from ingest.jsonl_loader import load_all_jsonl

CASE_LAW_DIR = str(BASE_DIR / "data" / "raw" / "case_law")
QA_DATASETS_DIR = str(BASE_DIR / "data" / "raw" / "qa_datasets")


def _collection_count(conn, collection_name: str) -> int:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM legal_chunks WHERE collection = %s",
                (collection_name,),
            )
            return cur.fetchone()[0]
    except Exception:
        conn.rollback()
        return 0


def main():
    print("=" * 60)
    print("LEGAL RAG — NEW DATA INGESTION (Supabase pgvector)")
    print("=" * 60)

    model = SentenceTransformer(EMBEDDING_MODEL)
    conn = get_db_connection()
    results = {}

    try:
        print("\n[1/3] PDF documents...")
        existing_pdf = _collection_count(conn, "legal_docs")
        print(f"  Existing in 'legal_docs': {existing_pdf:,} chunks")

        pdf_chunks = load_all_pdfs()
        if existing_pdf < len(pdf_chunks):
            print(f"  Ingesting {len(pdf_chunks):,} PDF chunks...")
            results["pdf"] = ingest_chunks(conn, pdf_chunks, model, "legal_docs")
        else:
            print("  Skipping — already up to date")
            results["pdf"] = 0

        print("\n[2/3] Case law datasets...")
        case_existing = _collection_count(conn, "legal_case_law")
        if case_existing == 0:
            case_chunks = load_all_jsonl(CASE_LAW_DIR)
            print(f"  Total chunks: {len(case_chunks):,}")
            if case_chunks:
                results["case_law"] = ingest_chunks(conn, case_chunks, model, "legal_case_law")
            else:
                results["case_law"] = 0
        else:
            print(f"  Skipping — already has {case_existing:,} chunks")
            results["case_law"] = 0

        print("\n[3/3] Statutes & QA datasets...")
        stat_existing = _collection_count(conn, "legal_statutes")
        if stat_existing == 0:
            stat_chunks = load_all_jsonl(QA_DATASETS_DIR)
            print(f"  Total chunks: {len(stat_chunks):,}")
            if stat_chunks:
                results["statutes"] = ingest_chunks(conn, stat_chunks, model, "legal_statutes")
            else:
                results["statutes"] = 0
        else:
            print(f"  Skipping — already has {stat_existing:,} chunks")
            results["statutes"] = 0

        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        total_new = sum(results.values())
        for src, count in results.items():
            print(f"  {src:15s}: {count:,} new chunks")
        print(f"  {'TOTAL':15s}: {total_new:,} new chunks")

        print("\n  All collections:")
        for name in ["legal_qa", "legal_docs", "legal_case_law", "legal_statutes"]:
            count = _collection_count(conn, name)
            print(f"    {name:20s}: {count:,}")
        print("=" * 60)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
