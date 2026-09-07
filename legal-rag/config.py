import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_FALLBACK_MODEL = "gemini-3.5-flash"

DATABASE_URL = os.getenv("DATABASE_URL", "")

PDF_DIR = str(BASE_DIR / "data" / "raw" / "pdfs")
CSV_PATH = str(BASE_DIR / "data" / "raw" / "indian_law_dataset.csv")
CASE_LAW_DIR = str(BASE_DIR / "data" / "raw" / "case_law")
QA_DATASETS_DIR = str(BASE_DIR / "data" / "raw" / "qa_datasets")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 384

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100
MAX_INPUT_LENGTH = 1000
MAX_CONTEXT_TOKENS = 100000
TOP_K_RESULTS = 10
TOP_K_PER_COLLECTION = 5
TOP_K_BM25 = 10
TOP_K_FINAL = 10
MAX_QUERIES_PER_SESSION = 10
MAX_CONVERSATION_TURNS = 5

SYSTEM_PROMPT = (
    "You are an expert Indian legal assistant. Answer questions based ONLY on the "
    "provided legal documents. Always cite your source (document name and section if "
    "available). If you cannot find the answer in the provided context, say "
    "'I could not find relevant information in my legal database.' "
    "Never hallucinate legal information."
)

ASSISTANT_SYSTEM_PROMPT = (
    "You are a personal Indian legal assistant. Based on the user's legal problem and "
    "the provided legal documents, generate a structured response with:\n"
    "1. USER'S LEGAL RIGHTS: List the specific rights the user has in this situation\n"
    "2. APPLICABLE LAWS/SECTIONS: Cite the exact laws, acts, and sections that apply\n"
    "3. RECOMMENDED ACTIONS: Step-by-step actions the user should take\n"
    "4. DISCLAIMER: Always end with a note to consult a qualified lawyer\n\n"
    "Base your answer ONLY on the provided legal documents. Never hallucinate."
)

QUERY_EXPANSION_PROMPT = (
    "Given the following legal question, generate 2 alternative phrasings that "
    "capture the same intent but use different legal terminology. "
    "Return ONLY the 2 alternatives, one per line, no numbering or extra text.\n\n"
    "Question: {query}"
)

PROBLEM_CATEGORIES = [
    "criminal",
    "civil",
    "property",
    "consumer",
    "labor",
    "family",
]

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://*.vercel.app",
]
