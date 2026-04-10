#Pydantic automatically validates that incoming data has the right fields and types. If you pass a dictionary missing answer — it throws immediately with a clear error message.


# generator.py
import json
from openai import OpenAI
from pydantic import BaseModel, field_validator
from config import OPENAI_API_KEY
from prompts import build_messages

client = OpenAI(api_key=OPENAI_API_KEY)


# ── Output Schema ──────────────────────────────────────────────
class RAGResponse(BaseModel):
    answer: str
    confidence: str
    source_quote: str

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        allowed = {"high", "medium", "low", "none"}
        if v not in allowed:
            raise ValueError(f"confidence must be one of {allowed}, got: {v}")
        return v

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("answer is too short")
        return v.strip()


# ── Helper — safe JSON parse ───────────────────────────────────
def parse_json_safely(text: str) -> dict:
    # models sometimes wrap JSON in ```json ... ``` even when told not to
    # strip those if present
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        cleaned = "\n".join(lines).strip()

    return json.loads(cleaned)


# ── Main Function ──────────────────────────────────────────────
def generate_answer(
    question: str,
    chunks: list,
    prompt_version: str = None,
    max_retries: int = 3,
) -> dict:

    # build context from chunks
    context = "\n\n".join([
        f"[Source {i+1}] {chunk['text']}"
        for i, chunk in enumerate(chunks)
    ])

    if not context.strip():
        return {
            "answer": "I cannot find this information in the provided documents.",
            "confidence": "none",
            "source_quote": "",
            "prompt_version": prompt_version,
            "retries": 0,
            "error": None,
        }

    # build messages for this version
    messages = build_messages(context, question, prompt_version)

    last_error = None

    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}")

        # on retry — tell model exactly what was wrong
        if attempt > 1 and last_error:
            messages.append({
                "role": "user",
                "content": f"""Your previous response had this error: {last_error}

Please respond again with valid JSON in exactly this format:
{{
    "answer": "your answer here",
    "confidence": "high or medium or low or none",
    "source_quote": "exact quote from context"
}}"""
            })

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.1,
            )

            raw = response.choices[0].message.content

            # step 1 — parse JSON
            try:
                parsed = parse_json_safely(raw)
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON: {str(e)}"
                print(f"JSON error: {last_error}")
                continue

            # step 2 — validate schema
            try:
                validated = RAGResponse(**parsed)
            except Exception as e:
                last_error = f"Schema error: {str(e)}"
                print(f"Schema error: {last_error}")
                continue

            # success
            return {
                **validated.model_dump(),
                "prompt_version": prompt_version or "v4",
                "retries": attempt - 1,
                "error": None,
            }

        except Exception as e:
            last_error = str(e)
            print(f"API error: {last_error}")

    # all retries exhausted
    return {
        "answer": "I encountered an error generating a response.",
        "confidence": "none",
        "source_quote": "",
        "prompt_version": prompt_version,
        "retries": max_retries,
        "error": last_error,
    }


if __name__ == "__main__":
    from pdf_loader import extract_text_from_pdf
    from chunker import chunk_overlap
    from embedder import embed_chunks, embed_texts
    from vector_store import build_vector_store, search_vector_store
    from bm25_store import build_bm25_index, search_bm25
    from hybrid import reciprocal_rank_fusion
    from reranker import rerank

    # build pipeline
    text = extract_text_from_pdf("OnePiece.pdf")
    chunks = chunk_overlap(text, chunk_size=500, overlap=50)
    embedded = embed_chunks(chunks)

    index, stored_chunks = build_vector_store(embedded)
    bm25, bm25_chunks = build_bm25_index(chunks)

    # ask a question
    query = "Who is the captain of the Straw Hat Pirates?"

    query_embedding = embed_texts([query])[0]
    vector_results = search_vector_store(index, stored_chunks, query_embedding, top_k=10)
    bm25_results = search_bm25(bm25, bm25_chunks, query, top_k=10)
    hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results)
    final_chunks = rerank(query, hybrid_results, top_k=4)

    # generate with different versions
    for version in ["v1", "v2", "v3", "v4"]:
        print(f"\n{'='*40}")
        print(f"Version: {version}")
        result = generate_answer(query, final_chunks, prompt_version=version)
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Retries: {result['retries']}")