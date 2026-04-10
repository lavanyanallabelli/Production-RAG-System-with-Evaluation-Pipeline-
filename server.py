# server.py
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

from pdf_loader import extract_text_from_pdf
from chunker import chunk_overlap
from embedder import embed_chunks, embed_texts
from vector_store import build_vector_store, search_vector_store, save_vector_store, load_vector_store
from bm25_store import build_bm25_index, search_bm25, save_bm25_index, load_bm25_index
from hybrid import reciprocal_rank_fusion
from reranker import rerank
from generator import generate_answer
from ab_test import run_ab_test, print_final_report

# ── Global State ───────────────────────────────────────────────
# these live in memory while server is running
index = None
stored_chunks = []
bm25 = None
bm25_chunks = []


# ── Startup — load existing indexes if they exist ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, stored_chunks, bm25, bm25_chunks

    if os.path.exists("storage/faiss_index") and os.path.exists("storage/chunks.json"):
        index, stored_chunks = load_vector_store("storage/faiss_index", "storage/chunks.json")
        print(f"Loaded existing FAISS index")

    if os.path.exists("storage/bm25_index.pkl"):
        bm25, bm25_chunks = load_bm25_index("storage/bm25_index.pkl")
        print(f"Loaded existing BM25 index")

    yield   # server runs here",

app = FastAPI(
    title="Production RAG System",
    description="Document Q&A with hybrid retrieval and eval pipeline",
    lifespan=lifespan,
)

# add this after app = FastAPI(...)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response Models ────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    prompt_version: Optional[str] = None

class TestQuestion(BaseModel):
    question: str
    expected: str

class EvalRequest(BaseModel):
    test_questions: list[TestQuestion]
    versions: Optional[list[str]] = None


# ── Route 1: Health Check ──────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": index is not None,
        "chunks_loaded": len(stored_chunks),
    }


# ── Route 2: Upload PDF ────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global index, stored_chunks, bm25, bm25_chunks

    # validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # run full ingestion pipeline
        print(f"Processing: {file.filename}")

        text = extract_text_from_pdf(tmp_path)
        chunks = chunk_overlap(text, chunk_size=500, overlap=50)
        embedded = embed_chunks(chunks)

        # build indexes
        index, stored_chunks = build_vector_store(embedded)
        bm25, bm25_chunks = build_bm25_index(chunks)

        # save to disk
        save_vector_store(index, stored_chunks, "storage/faiss_index", "storage/chunks.json")
        save_bm25_index(bm25, bm25_chunks, "storage/bm25_index.pkl")

        return {
            "message": "PDF processed successfully",
            "filename": file.filename,
            "chunks_created": len(stored_chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # always delete temp file
        os.unlink(tmp_path)


# ── Route 3: Ask Question ──────────────────────────────────────
@app.post("/ask")
async def ask(request: AskRequest):
    # check indexes are loaded
    if index is None or bm25 is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. POST a PDF to /upload first."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        # retrieve
        query_embedding = embed_texts([request.question])[0]
        vector_results = search_vector_store(index, stored_chunks, query_embedding, top_k=10)
        bm25_results = search_bm25(bm25, bm25_chunks, request.question, top_k=10)
        hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results)
        final_chunks = rerank(request.question, hybrid_results, top_k=4)

        # generate
        result = generate_answer(
            request.question,
            final_chunks,
            prompt_version=request.prompt_version,
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "source_quote": result["source_quote"],
            "prompt_version": result["prompt_version"],
            "retries": result["retries"],
            "chunks_used": len(final_chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Route 4: Run Eval ──────────────────────────────────────────
@app.post("/eval")
async def run_eval(request: EvalRequest):
    if index is None or bm25 is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. POST a PDF to /upload first."
        )

    try:
        # convert pydantic models to plain dicts
        test_questions = [
            {"question": q.question, "expected": q.expected}
            for q in request.test_questions
        ]

        results = run_ab_test(
            test_questions,
            index,
            stored_chunks,
            bm25,
            bm25_chunks,
            versions=request.versions,
        )

        # build clean response
        comparison = results["comparison"]

        return {
            "winner": results["winner"],
            "comparison": comparison,
            "message": f"{results['winner']} performed best by faithfulness score"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))