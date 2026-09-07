"""Generate answers using Google Gemini with retrieved legal context."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import tiktoken
from google import genai

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_FALLBACK_MODEL,
    MAX_CONTEXT_TOKENS,
    SYSTEM_PROMPT,
    ASSISTANT_SYSTEM_PROMPT,
)


class GenerationResult(TypedDict):
    answer: str
    sources: list[str]
    confidence: str


_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _build_context(chunks: list[dict], max_tokens: int) -> str:
    context_parts: list[str] = []
    token_count = 0

    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        meta = chunk.get("metadata", {})
        page = meta.get("page", "")
        page_info = f", Page {page}" if page else ""

        entry = f"[Source {i}: {source}{page_info}]\n{chunk['text']}\n"
        entry_tokens = _count_tokens(entry)

        if token_count + entry_tokens > max_tokens:
            break

        context_parts.append(entry)
        token_count += entry_tokens

    return "\n".join(context_parts)


def _assess_confidence(chunks: list[dict], answer: str) -> str:
    if not chunks:
        return "low"

    top_scores = [c.get("score", 0) for c in chunks[:3]]
    avg_score = sum(top_scores) / len(top_scores) if top_scores else 0

    no_info_phrases = [
        "could not find",
        "no relevant information",
        "not available in",
        "cannot find",
    ]
    if any(phrase in answer.lower() for phrase in no_info_phrases):
        return "low"

    if avg_score >= 0.5:
        return "high"
    elif avg_score >= 0.3:
        return "medium"
    return "low"


def _get_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set. Add it to your .env file."
        )
    return genai.Client(api_key=GEMINI_API_KEY)


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    system_prompt: str | None = None,
) -> GenerationResult:
    sys_prompt = system_prompt or SYSTEM_PROMPT

    context = _build_context(retrieved_chunks, MAX_CONTEXT_TOKENS - 2000)

    user_message = (
        f"LEGAL CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        "Provide a comprehensive answer based on the above legal context. "
        "Cite specific sources."
    )

    total_tokens = _count_tokens(sys_prompt) + _count_tokens(user_message)
    if total_tokens > MAX_CONTEXT_TOKENS:
        excess = total_tokens - MAX_CONTEXT_TOKENS + 500
        context_tokens = _enc.encode(context)
        truncated = _enc.decode(context_tokens[:-excess])
        user_message = (
            f"LEGAL CONTEXT:\n{truncated}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            "Provide a comprehensive answer based on the above legal context. "
            "Cite specific sources."
        )

    models_to_try = [GEMINI_MODEL, GEMINI_FALLBACK_MODEL]
    answer = ""
    for model_name in models_to_try:
        success = False
        for attempt in range(2):
            try:
                client = _get_client()
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        {"role": "user", "parts": [{"text": f"{sys_prompt}\n\n{user_message}"}]}
                    ],
                )
                answer = response.text or "No response generated."
                success = True
                break
            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "UNAVAILABLE" in err_str or "overloaded" in err_str.lower()):
                    time.sleep(2 ** (attempt + 1))
                    continue
                answer = f"Error generating response: {err_str}"
                success = True
                break
        if success and not answer.startswith("Error"):
            break

    sources = list(
        dict.fromkeys(
            c.get("source", "Unknown") for c in retrieved_chunks
        )
    )

    confidence = _assess_confidence(retrieved_chunks, answer)

    return GenerationResult(
        answer=answer,
        sources=sources,
        confidence=confidence,
    )


def generate_assistant_response(
    problem: str,
    category: str,
    retrieved_chunks: list[dict],
) -> GenerationResult:
    enriched_prompt = (
        f"{ASSISTANT_SYSTEM_PROMPT}\n\n"
        f"Problem Category: {category}\n"
    )
    return generate_answer(problem, retrieved_chunks, enriched_prompt)


if __name__ == "__main__":
    result = generate_answer(
        "What is the punishment for theft?",
        [
            {
                "text": "Section 379 IPC: Punishment for theft - imprisonment up to 3 years, or fine, or both.",
                "source": "IPC.pdf",
                "metadata": {"page": 100},
                "score": 0.8,
            }
        ],
    )
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Confidence: {result['confidence']}")
