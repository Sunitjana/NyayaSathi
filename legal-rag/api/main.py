"""FastAPI backend for the Legal RAG assistant."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import traceback
import uuid
from typing import AsyncGenerator, Literal

import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from pydantic import BaseModel, Field

from config import (
    ASSISTANT_SYSTEM_PROMPT,
    CORS_ORIGINS,
    DATABASE_URL,
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODEL,
    GEMINI_MODEL,
    MAX_CONTEXT_TOKENS,
    MAX_INPUT_LENGTH,
    PROBLEM_CATEGORIES,
    SYSTEM_PROMPT,
    TOP_K_PER_COLLECTION,
)

from rag.generator import _assess_confidence, _build_context
from rag.pipeline import run_query
from rag.retriever import get_collection_stats, retrieve
from assistant.legal_assistant import (
    _build_category_query,
    classify_problem,
    run_assistant,
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Legal RAG API",
    description="Indian Legal Document Retrieval & Virtual Legal Assistant",
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class HistoryTurn(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)


class AssistantRequest(BaseModel):
    problem: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)
    category: str | None = None


class StreamRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)
    mode: Literal["search", "assistant"] = "search"
    category: str | None = None
    history: list[HistoryTurn] = Field(default_factory=list)


class SessionSave(BaseModel):
    id: str | None = None
    title: str
    mode: str
    messages: list[dict]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok"}


# ============================================================
# DATABASE / RAG STATS
# ============================================================

@app.get("/stats")
async def stats():
    try:
        collection_stats = get_collection_stats()
        total = sum(collection_stats.values())

        return {
            "collections": collection_stats,
            "total_chunks": total,
            "categories": PROBLEM_CATEGORIES,
        }

    except Exception as e:
        print(f"Stats Error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================
# NORMAL QUERY ENDPOINT
# ============================================================

@app.post("/query")
async def query(req: QueryRequest):
    try:
        result = await asyncio.to_thread(run_query, req.query)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "retrieved_chunks": [
                {
                    "text": c["text"][:500],
                    "source": c["source"],
                    "score": c["score"],
                }
                for c in result["retrieved_chunks"]
            ],
            "related_questions": result["related_questions"],
        }

    except Exception as e:
        print("\n========== QUERY ERROR ==========")
        print(str(e))
        traceback.print_exc()
        print("=================================\n")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================
# LEGAL ASSISTANT ENDPOINT
# ============================================================

@app.post("/assistant")
async def assistant(req: AssistantRequest):
    try:
        result = await asyncio.to_thread(run_assistant, req.problem, req.category)
        return {
            "category": result["category"],
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
        }

    except Exception as e:
        print("\n========== ASSISTANT ERROR ==========")
        print(str(e))
        traceback.print_exc()
        print("=====================================\n")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================
# CLASSIFY LEGAL PROBLEM
# ============================================================

@app.post("/classify")
async def classify(req: QueryRequest):
    return {"category": classify_problem(req.query)}


# ============================================================
# BUILD CHAT HISTORY
# ============================================================

def _build_history_context(
    history: list[HistoryTurn],
    max_turns: int = 5,
) -> str:
    if not history:
        return ""

    recent = history[-(max_turns * 2) :]
    parts = []

    for turn in recent:
        label = "User" if turn.role == "user" else "Assistant"
        parts.append(f"{label}: {turn.content[:500]}")

    return "CONVERSATION HISTORY:\n" + "\n".join(parts) + "\n\n"


# ============================================================
# STREAM RESPONSE
# ============================================================

async def _stream_response(
    query: str,
    mode: str,
    category: str | None,
    history: list[HistoryTurn],
) -> AsyncGenerator[str, None]:
    if mode == "assistant":
        detected = category or classify_problem(query)
        enriched = _build_category_query(query, detected)
        retrieved = await asyncio.to_thread(retrieve, enriched, TOP_K_PER_COLLECTION)
        sys_prompt = f"{ASSISTANT_SYSTEM_PROMPT}\n\nProblem Category: {detected}"
        yield f"data: {json.dumps({'type': 'category', 'category': detected})}\n\n"
    else:
        retrieved = await asyncio.to_thread(retrieve, query, TOP_K_PER_COLLECTION)
        sys_prompt = SYSTEM_PROMPT

    if not retrieved:
        yield f"data: {json.dumps({'type': 'chunk', 'content': 'I could not find relevant information in my legal database.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'confidence': 'low', 'related_questions': []})}\n\n"
        return

    chunk_dicts = [
        {
            "text": r["text"],
            "source": r["source"],
            "metadata": r["metadata"],
            "score": r["score"],
        }
        for r in retrieved
    ]

    sources = list(dict.fromkeys(c["source"] for c in chunk_dicts))
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'chunks': [{'text': c['text'][:300], 'source': c['source'], 'score': c['score']} for c in chunk_dicts[:5]]})}\n\n"

    context = _build_context(chunk_dicts, MAX_CONTEXT_TOKENS - 2000)
    history_ctx = _build_history_context(history)

    user_message = (
        f"{history_ctx}"
        f"LEGAL CONTEXT:\n"
        f"{context}\n\n"
        f"USER QUESTION:\n"
        f"{query}\n\n"
        "Provide a comprehensive answer based on the above legal context. Cite specific sources."
    )

    if not GEMINI_API_KEY:
        error_message = "Gemini API key is missing. Please check your .env file."
        print(f"\nERROR: {error_message}\n")
        yield f"data: {json.dumps({'type': 'chunk', 'content': error_message})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'confidence': 'low', 'related_questions': []})}\n\n"
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    last_error = None

    models_to_try = []
    if GEMINI_MODEL:
        models_to_try.append(GEMINI_MODEL)
    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
        models_to_try.append(GEMINI_FALLBACK_MODEL)

    for model_name in models_to_try:
        try:
            print("\n========================================")
            print(f"Trying Gemini model: {model_name}")
            print("========================================")

            response = client.models.generate_content_stream(
                model=model_name,
                contents=user_message,
                config={"system_instruction": sys_prompt},
            )

            full_answer = ""
            for chunk in response:
                if hasattr(chunk, "text") and chunk.text:
                    full_answer += chunk.text
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk.text})}\n\n"

            if not full_answer.strip():
                raise RuntimeError("Gemini returned an empty response.")

            confidence = _assess_confidence(chunk_dicts, full_answer)
            related = [
                r["metadata"].get("instruction", "")
                for r in retrieved
                if r["metadata"].get("instruction")
            ][:3]

            yield f"data: {json.dumps({'type': 'done', 'confidence': confidence, 'related_questions': related})}\n\n"
            print(f"\nSUCCESS: Gemini response generated using {model_name}\n")
            return

        except Exception as e:
            last_error = e
            print("\n========== GEMINI MODEL ERROR ==========")
            print(f"Model: {model_name}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {str(e)}")
            traceback.print_exc()
            print("========================================\n")
            continue

    print("\n========== ALL GEMINI MODELS FAILED ==========")
    print(f"Final Error Type: {type(last_error).__name__}")
    print(f"Final Error: {last_error}")
    print("==============================================\n")

    yield f"data: {json.dumps({'type': 'chunk', 'content': 'Error generating response. Please try again.'})}\n\n"
    yield f"data: {json.dumps({'type': 'done', 'confidence': 'low', 'related_questions': []})}\n\n"


# ============================================================
# STREAM ENDPOINT
# ============================================================

@app.post("/stream")
async def stream(req: StreamRequest):
    return StreamingResponse(
        _stream_response(req.query, req.mode, req.category, req.history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# SAVE CHAT SESSION
# ============================================================

@app.post("/sessions")
async def save_session(req: SessionSave):
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id text PRIMARY KEY,
                title text NOT NULL,
                mode text NOT NULL,
                messages jsonb NOT NULL,
                created_at timestamptz DEFAULT now(),
                updated_at timestamptz DEFAULT now()
            )
            """
        )

        session_id = req.id or str(uuid.uuid4())

        cur.execute(
            """
            INSERT INTO chat_sessions
            (id, title, mode, messages, updated_at)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (id)
            DO UPDATE SET
                title = EXCLUDED.title,
                messages = EXCLUDED.messages,
                updated_at = now()
            """,
            (session_id, req.title, req.mode, json.dumps(req.messages)),
        )

        conn.commit()
        return {"id": session_id}

    except Exception as e:
        print(f"Session Save Error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if conn:
            conn.close()


# ============================================================
# LIST CHAT SESSIONS
# ============================================================

@app.get("/sessions")
async def list_sessions():
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id text PRIMARY KEY,
                title text NOT NULL,
                mode text NOT NULL,
                messages jsonb NOT NULL,
                created_at timestamptz DEFAULT now(),
                updated_at timestamptz DEFAULT now()
            )
            """
        )
        conn.commit()

        cur.execute(
            """
            SELECT id, title, mode, created_at, updated_at
            FROM chat_sessions
            ORDER BY updated_at DESC
            LIMIT 20
            """
        )

        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "title": r[1],
                "mode": r[2],
                "created_at": str(r[3]),
                "updated_at": str(r[4]),
            }
            for r in rows
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if conn:
            conn.close()


# ============================================================
# GET SINGLE CHAT SESSION
# ============================================================

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, title, mode, messages
            FROM chat_sessions
            WHERE id = %s
            """,
            (session_id,),
        )

        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "id": row[0],
            "title": row[1],
            "mode": row[2],
            "messages": row[3],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if conn:
            conn.close()


# ============================================================
# DELETE CHAT SESSION
# ============================================================

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM chat_sessions
            WHERE id = %s
            """,
            (session_id,),
        )
        conn.commit()
        return {"deleted": True}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if conn:
            conn.close()


# ============================================================
# RUN SERVER DIRECTLY
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )