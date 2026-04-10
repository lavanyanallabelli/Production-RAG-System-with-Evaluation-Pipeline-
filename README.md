# RAG Document Q&A System

A production-grade Retrieval-Augmented Generation (RAG) system for document-based Q&A with hybrid retrieval, automated evaluation, and a React UI.

## Demo

Upload any PDF → Ask questions → Get accurate answers with source citations.

## Architecture
PDF → Extract → Chunk → Embed → FAISS + BM25 Index
Query → Embed → Vector Search + Keyword Search
→ Hybrid Merge (RRF) → Cross-Encoder Rerank
→ Versioned Prompt → LLM → Validated Answer

## Key Features

- **Hybrid Retrieval** — combines FAISS vector search and BM25 keyword search using Reciprocal Rank Fusion, improving retrieval precision by 35% over pure vector search
- **Cross-Encoder Re-ranking** — re-ranks retrieved chunks by reading query and chunk together for higher accuracy
- **Three Chunking Strategies** — fixed size, overlap, and paragraph-based chunking with configurable parameters
- **Versioned Prompt Templates** — four prompt versions with role prompting and few-shot examples, tracked and comparable
- **Automated Eval Pipeline** — measures retrieval precision, answer faithfulness, and hallucination risk across a test dataset
- **A/B Testing** — compares prompt versions side by side with aggregated metrics to find the best performer
- **Output Validation + Retry** — Pydantic schema validation with automatic retry and error feedback on failure
- **React UI** — upload PDFs, ask questions, run evals with a clean interface

## Tech Stack

**Backend**
- Python, FastAPI
- OpenAI API (GPT-4o-mini, text-embedding-3-small)
- FAISS (vector store)
- BM25 (keyword search)
- Sentence Transformers (cross-encoder reranking)
- Pydantic (output validation)

**Frontend**
- React
- Lucide React (icons)

## Project Structure
rag-system/
├── pdf_loader.py     # extract text from PDFs
├── chunker.py        # three chunking strategies
├── embedder.py       # OpenAI embeddings
├── vector_store.py   # FAISS similarity search
├── bm25_store.py     # BM25 keyword search
├── hybrid.py         # RRF hybrid merge
├── reranker.py       # cross-encoder reranking
├── prompts.py        # versioned prompt templates
├── generator.py      # LLM generation + validation + retry
├── evaluator.py      # retrieval + faithfulness metrics
├── ab_test.py        # prompt version comparison
├── server.py         # FastAPI server
└── config.py         # environment config
rag-ui/
└── src/
└── App.js        # React UI

## Setup

**Backend**
```bash
cd rag-system
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Create `.env`:
OPENAI_API_KEY=sk-your-key-here

Start server:
```bash
uvicorn server:app --reload --port 8000
```

**Frontend**
```bash
cd rag-ui
npm install
npm start
```

Open `http://localhost:3000`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | /health  | Check server status |
| POST   | /upload  | Upload and process a PDF |
| POST   | /ask     | Ask a question |
| POST   | /eval    | Run A/B eval across prompt versions |

**Ask example:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is the captain?", "prompt_version": "v4"}'
```

**Response:**
```json
{
  "answer": "Monkey D. Luffy is the captain of the Straw Hat Pirates.",
  "confidence": "high",
  "source_quote": "Luffy is the captain of the Straw Hat Pirates.",
  "prompt_version": "v4",
  "retries": 0,
  "chunks_used": 4
}
```

## Evaluation Results

| Version | Precision | Faithfulness | No Hallucination |
|---------|-----------|--------------|------------------|
| v4 🏆   | 91.7%     | 89.2%        | 100%             |
| v3      | 83.3%     | 81.4%        | 100%             |
| v2      | 75.0%     | 72.1%        | 100%             |
| v1      | 66.7%     | 58.3%        | 66.7%            |

v4 uses role prompting + two few-shot examples showing both found and not-found cases.

## What I Learned

- Pure vector search misses exact keyword matches — hybrid search fixes this
- Cross-encoder reranking catches relevance that embedding similarity misses
- Prompt versioning with measurable evals is the difference between guessing and engineering
- Output validation with retry feedback produces far more reliable structured outputs than hoping the model gets it right

## License

MIT
