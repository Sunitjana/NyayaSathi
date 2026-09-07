"""Advanced retriever: hybrid search (pgvector semantic + BM25) with cross-encoder reranking."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from google import genai

from config import (
    DATABASE_URL,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODEL,
    QUERY_EXPANSION_PROMPT,
    RERANKER_MODEL,
    TOP_K_BM25,
    TOP_K_FINAL,
    TOP_K_PER_COLLECTION,
)

ALL_COLLECTIONS = ["legal_qa", "legal_docs", "legal_case_law", "legal_statutes"]


class SearchResult(TypedDict):
    text: str
    source: str
    metadata: dict
    score: float


_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_bm25_index: dict | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _get_conn():
    return psycopg2.connect(DATABASE_URL)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _build_bm25_index(collection_name: str) -> tuple[BM25Okapi, list[dict]] | None:
    conn = None
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT text, source, metadata FROM legal_chunks WHERE collection = %s",
                (collection_name,),
            )
            rows = cur.fetchall()
    except Exception:
        return None
    finally:
        if conn:
            conn.close()

    if not rows:
        return None

    docs = [r[0] for r in rows]
    corpus = [_tokenize(doc) for doc in docs]
    bm25 = BM25Okapi(corpus)

    doc_records = [
        {
            "text": r[0],
            "metadata": r[2] if isinstance(r[2], dict) else json.loads(r[2]) if r[2] else {},
        }
        for r in rows
    ]

    return bm25, doc_records


def _get_bm25_index() -> dict:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = {}
        for name in ALL_COLLECTIONS:
            result = _build_bm25_index(name)
            if result:
                _bm25_index[name] = result
    return _bm25_index


def _bm25_search(query: str, top_k: int) -> list[SearchResult]:
    index = _get_bm25_index()
    query_tokens = _tokenize(query)
    results: list[SearchResult] = []

    for collection_name, (bm25, doc_records) in index.items():
        scores = bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            rec = doc_records[idx]
            max_score = max(scores) if max(scores) > 0 else 1.0
            normalized = scores[idx] / max_score
            results.append(
                SearchResult(
                    text=rec["text"],
                    source=rec["metadata"].get("source", collection_name),
                    metadata=rec["metadata"],
                    score=round(float(normalized), 4),
                )
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def _semantic_search(query: str, top_k_per_collection: int) -> list[SearchResult]:
    """Vector similarity search via pgvector."""
    model = _get_model()
    query_embedding = model.encode(query).tolist()

    results: list[SearchResult] = []

    conn = None
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = 100")
            for collection_name in ALL_COLLECTIONS:
                cur.execute(
                    """SELECT text, source, metadata,
                              1 - (embedding <=> %s::vector) AS similarity
                       FROM legal_chunks
                       WHERE collection = %s
                       ORDER BY embedding <=> %s::vector
                       LIMIT %s""",
                    (str(query_embedding), collection_name, str(query_embedding), top_k_per_collection),
                )
                rows = cur.fetchall()

                for row in rows:
                    text, source, meta, similarity = row
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    elif meta is None:
                        meta = {}
                    results.append(
                        SearchResult(
                            text=text,
                            source=source or collection_name,
                            metadata=meta,
                            score=round(float(similarity), 4),
                        )
                    )
    except Exception as e:
        print(f"Semantic search error: {e}")
    finally:
        if conn:
            conn.close()

    return results


def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for r in results:
        key = r["text"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _rerank(query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
    if not results:
        return results

    reranker = _get_reranker()
    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)

    for r, s in zip(results, scores):
        r["score"] = round(float(s), 4)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def _expand_query(query: str) -> list[str]:
    """Use Gemini to generate alternative phrasings for better retrieval."""
    try:
        if not GEMINI_API_KEY:
            return [query]

        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        response = client.models.generate_content(
            model=GEMINI_FALLBACK_MODEL,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        alternatives = [
            line.strip()
            for line in (response.text or "").strip().split("\n")
            if line.strip() and len(line.strip()) > 10
        ][:2]
        return [query] + alternatives
    except Exception:
        return [query]


def retrieve(query: str, top_k_per_collection: int | None = None, expand: bool = False) -> list[SearchResult]:
    k = top_k_per_collection or TOP_K_PER_COLLECTION

    queries = _expand_query(query) if expand else [query]

    all_semantic: list[SearchResult] = []
    for q in queries:
        all_semantic.extend(_semantic_search(q, k))

    combined = _deduplicate(all_semantic)
    reranked = _rerank(query, combined, TOP_K_FINAL)

    return reranked


def get_collection_stats() -> dict:
    stats = {}
    conn = None
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            for name in ALL_COLLECTIONS:
                cur.execute(
                    "SELECT COUNT(*) FROM legal_chunks WHERE collection = %s",
                    (name,),
                )
                stats[name] = cur.fetchone()[0]
    except Exception:
        for name in ALL_COLLECTIONS:
            if name not in stats:
                stats[name] = 0
    finally:
        if conn:
            conn.close()
    return stats


if __name__ == "__main__":
    query = "What is the punishment for theft in India?"
    print(f"Query: {query}\n")
    results = retrieve(query)
    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['score']:.4f} | Source: {r['source']}")
        print(f"    {r['text'][:150]}...\n")
