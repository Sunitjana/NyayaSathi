"""Full RAG pipeline: query → retrieve → generate → answer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MAX_INPUT_LENGTH, TOP_K_PER_COLLECTION
from rag.retriever import SearchResult, retrieve
from rag.generator import GenerationResult, generate_answer


class PipelineResult(TypedDict):
    answer: str
    sources: list[str]
    confidence: str
    retrieved_chunks: list[SearchResult]
    related_questions: list[str]


def _extract_related_questions(chunks: list[SearchResult], limit: int = 3) -> list[str]:
    questions: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        instruction = chunk["metadata"].get("instruction", "")
        if instruction and instruction not in seen:
            seen.add(instruction)
            questions.append(instruction)
        if len(questions) >= limit:
            break

    return questions


def run_query(query: str) -> PipelineResult:
    if not query or not query.strip():
        return PipelineResult(
            answer="Please enter a valid legal question.",
            sources=[],
            confidence="low",
            retrieved_chunks=[],
            related_questions=[],
        )

    query = query.strip()
    if len(query) > MAX_INPUT_LENGTH:
        query = query[:MAX_INPUT_LENGTH]

    retrieved = retrieve(query, TOP_K_PER_COLLECTION)

    if not retrieved:
        return PipelineResult(
            answer="I could not find relevant information in my legal database. "
                   "Please try rephrasing your question.",
            sources=[],
            confidence="low",
            retrieved_chunks=[],
            related_questions=[],
        )

    chunk_dicts = [
        {
            "text": r["text"],
            "source": r["source"],
            "metadata": r["metadata"],
            "score": r["score"],
        }
        for r in retrieved
    ]

    result = generate_answer(query, chunk_dicts)
    related = _extract_related_questions(retrieved)

    return PipelineResult(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        retrieved_chunks=retrieved,
        related_questions=related,
    )


if __name__ == "__main__":
    test_query = "What are the fundamental rights in the Indian Constitution?"
    print(f"Query: {test_query}\n")
    result = run_query(test_query)
    print(f"Answer:\n{result['answer']}\n")
    print(f"Sources: {result['sources']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Related questions: {result['related_questions']}")
