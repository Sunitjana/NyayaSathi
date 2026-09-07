"""Personal Legal Assistant — classifies problems and provides structured legal guidance."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_INPUT_LENGTH,
    PROBLEM_CATEGORIES,
    TOP_K_PER_COLLECTION,
)
from rag.retriever import retrieve
from rag.generator import generate_assistant_response, GenerationResult


CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "criminal": [
        "theft", "murder", "assault", "robbery", "fraud", "cheating",
        "kidnapping", "arrest", "bail", "fir", "police", "crime",
        "criminal", "extortion", "forgery", "dowry", "harassment",
        "stalking", "cyber crime", "defamation",
    ],
    "civil": [
        "contract", "agreement", "breach", "damages", "injunction",
        "suit", "decree", "civil", "tort", "negligence", "dispute",
    ],
    "property": [
        "property", "land", "tenant", "landlord", "rent", "lease",
        "eviction", "possession", "registration", "stamp duty",
        "encroachment", "partition", "will", "inheritance", "succession",
        "security deposit", "transfer",
    ],
    "consumer": [
        "consumer", "product", "defective", "refund", "warranty",
        "service", "complaint", "consumer forum", "unfair trade",
        "misleading", "overcharging",
    ],
    "labor": [
        "employment", "salary", "wages", "termination", "dismissal",
        "retrenchment", "labor", "labour", "workplace", "pf",
        "provident fund", "gratuity", "esi", "working hours",
        "maternity", "sexual harassment", "posh",
    ],
    "family": [
        "divorce", "marriage", "custody", "alimony", "maintenance",
        "adoption", "domestic violence", "dowry", "child", "guardian",
        "matrimonial", "hindu marriage", "muslim marriage", "family",
    ],
}


class AssistantResult(TypedDict):
    category: str
    answer: str
    sources: list[str]
    confidence: str


def classify_problem(text: str) -> str:
    text_lower = text.lower()
    scores: dict[str, int] = {cat: 0 for cat in PROBLEM_CATEGORIES}

    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[category] += 1

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] == 0:
        return "civil"
    return best


def _build_category_query(problem: str, category: str) -> str:
    category_terms = {
        "criminal": "criminal law IPC BNS punishment offense",
        "civil": "civil procedure suit damages contract",
        "property": "property law transfer land tenant landlord rent",
        "consumer": "consumer protection rights complaint forum",
        "labor": "labor law employment wages termination workplace",
        "family": "family law marriage divorce custody maintenance",
    }
    suffix = category_terms.get(category, "Indian law")
    return f"{problem} {suffix}"


def run_assistant(
    problem: str,
    category: str | None = None,
) -> AssistantResult:
    if not problem or not problem.strip():
        return AssistantResult(
            category="unknown",
            answer="Please describe your legal problem.",
            sources=[],
            confidence="low",
        )

    problem = problem.strip()
    if len(problem) > MAX_INPUT_LENGTH:
        problem = problem[:MAX_INPUT_LENGTH]

    detected_category = category or classify_problem(problem)
    if detected_category not in PROBLEM_CATEGORIES:
        detected_category = "civil"

    enriched_query = _build_category_query(problem, detected_category)
    retrieved = retrieve(enriched_query, TOP_K_PER_COLLECTION)

    if not retrieved:
        return AssistantResult(
            category=detected_category,
            answer="I could not find relevant legal information for your problem. "
                   "Please consult a qualified lawyer.",
            sources=[],
            confidence="low",
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

    gen_result: GenerationResult = generate_assistant_response(
        problem, detected_category, chunk_dicts
    )

    return AssistantResult(
        category=detected_category,
        answer=gen_result["answer"],
        sources=gen_result["sources"],
        confidence=gen_result["confidence"],
    )


if __name__ == "__main__":
    test_problem = "My landlord won't return my security deposit after I vacated the flat"
    print(f"Problem: {test_problem}\n")
    result = run_assistant(test_problem)
    print(f"Category: {result['category']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Sources: {result['sources']}")
    print(f"\nAnswer:\n{result['answer']}")
