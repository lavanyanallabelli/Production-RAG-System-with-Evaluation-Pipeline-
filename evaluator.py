# evaluator.py
import json
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


# ── Metric 1 — Retrieval Precision ────────────────────────────
def evaluate_retrieval_precision(
    question: str,
    retrieved_chunks: list,
    expected_answer: str,
) -> dict:
    """
    Asks the LLM: is each retrieved chunk actually relevant
    to answering this question?

    Precision = relevant chunks found / total chunks retrieved
    """

    if not retrieved_chunks:
        return {"precision": 0.0, "relevant": 0, "total": 0}

    relevant_count = 0

    for chunk in retrieved_chunks:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a relevance judge.
Respond only with JSON: {"relevant": true or false, "reason": "one sentence"}"""
                },
                {
                    "role": "user",
                    "content": f"""Question: {question}
Expected answer: {expected_answer}
Chunk: {chunk['text'][:400]}

Is this chunk relevant to answering the question?"""
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=100,
            temperature=0,
        )

        try:
            result = json.loads(response.choices[0].message.content)
            if result.get("relevant"):
                relevant_count += 1
        except:
            pass

    precision = relevant_count / len(retrieved_chunks)

    return {
        "precision": round(precision, 3),
        "relevant": relevant_count,
        "total": len(retrieved_chunks),
    }


# ── Metric 2 — Faithfulness ────────────────────────────────────
def evaluate_faithfulness(
    answer: str,
    retrieved_chunks: list,
) -> dict:
    """
    Asks the LLM: is every claim in this answer
    supported by the retrieved chunks?

    High faithfulness = no hallucination
    Low faithfulness = model invented information
    """

    context = "\n".join([c["text"][:300] for c in retrieved_chunks])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a faithfulness evaluator.
Check if an answer is fully supported by the context.
Respond only with JSON:
{
    "faithfulness_score": float between 0 and 1,
    "verdict": "faithful or unfaithful",
    "unsupported_claims": ["claim1", "claim2"]
}"""
            },
            {
                "role": "user",
                "content": f"""Context:
{context}

Answer to evaluate:
{answer}

Is every claim in the answer supported by the context?"""
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=200,
        temperature=0,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {
            "faithfulness_score": 0.0,
            "verdict": "error",
            "unsupported_claims": []
        }


# ── Metric 3 — Hallucination Detection ────────────────────────
def detect_hallucination(answer: str) -> dict:
    """
    Pattern-based detection of uncertainty signals.
    Fast and free — no API call needed.
    """
    import re

    patterns = [
        (r"i (don't|do not) (know|have)", "explicit_uncertainty"),
        (r"i('m| am) not sure", "uncertainty"),
        (r"i (believe|think|assume)", "hedging"),
        (r"(may|might|could) be", "speculation"),
        (r"as (of|far as) my (knowledge|training)", "knowledge_cutoff"),
    ]

    signals = []
    for pattern, signal_name in patterns:
        if re.search(pattern, answer.lower()):
            signals.append(signal_name)

    if len(signals) >= 2:
        risk = "high"
    elif len(signals) == 1:
        risk = "medium"
    else:
        risk = "low"

    return {
        "hallucination_risk": risk,
        "signals": signals,
    }


# ── Run Full Eval on One Question ─────────────────────────────
def evaluate_one(
    question: str,
    expected_answer: str,
    retrieved_chunks: list,
    generated_answer: str,
) -> dict:

    print(f"  Evaluating: {question[:50]}...")

    precision = evaluate_retrieval_precision(
        question, retrieved_chunks, expected_answer
    )

    faithfulness = evaluate_faithfulness(generated_answer, retrieved_chunks)

    hallucination = detect_hallucination(generated_answer)

    return {
        "question": question,
        "expected": expected_answer,
        "generated": generated_answer,
        "retrieval_precision": precision["precision"],
        "faithfulness_score": faithfulness.get("faithfulness_score", 0),
        "faithfulness_verdict": faithfulness.get("verdict", "error"),
        "hallucination_risk": hallucination["hallucination_risk"],
        "hallucination_signals": hallucination["signals"],
        "unsupported_claims": faithfulness.get("unsupported_claims", []),
    }


if __name__ == "__main__":
    from pdf_loader import extract_text_from_pdf
    from chunker import chunk_overlap
    from embedder import embed_chunks, embed_texts
    from vector_store import build_vector_store, search_vector_store
    from bm25_store import build_bm25_index, search_bm25
    from hybrid import reciprocal_rank_fusion
    from reranker import rerank
    from generator import generate_answer

    # build pipeline
    text = extract_text_from_pdf("OnePiece.pdf")
    chunks = chunk_overlap(text, chunk_size=500, overlap=50)
    embedded = embed_chunks(chunks)

    index, stored_chunks = build_vector_store(embedded)
    bm25, bm25_chunks = build_bm25_index(chunks)

    # test questions with known answers
    test_questions = [
        {
            "question": "Who is the captain of the Straw Hat Pirates?",
            "expected": "Monkey D. Luffy is the captain"
        },
        {
            "question": "What is Luffy's dream?",
            "expected": "Luffy wants to become King of the Pirates"
        },
        {
            "question": "Who is the swordsman of the crew?",
            "expected": "Roronoa Zoro is the swordsman"
        },
    ]

    # run eval on one version first
    version = "v4"
    print(f"\nRunning eval for prompt version: {version}")
    print("="*40)

    all_results = []

    for test in test_questions:
        # retrieve
        query_embedding = embed_texts([test["question"]])[0]
        vector_results = search_vector_store(index, stored_chunks, query_embedding, top_k=10)
        bm25_results = search_bm25(bm25, bm25_chunks, test["question"], top_k=10)
        hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results)
        final_chunks = rerank(test["question"], hybrid_results, top_k=4)

        # generate
        result = generate_answer(test["question"], final_chunks, prompt_version=version)

        # evaluate
        eval_result = evaluate_one(
            test["question"],
            test["expected"],
            final_chunks,
            result["answer"],
        )

        all_results.append(eval_result)

    # aggregate
    avg_precision = sum(r["retrieval_precision"] for r in all_results) / len(all_results)
    avg_faithfulness = sum(r["faithfulness_score"] for r in all_results) / len(all_results)
    low_hallucination = sum(1 for r in all_results if r["hallucination_risk"] == "low") / len(all_results)

    print(f"\n{'='*40}")
    print(f"RESULTS FOR {version}")
    print(f"{'='*40}")
    print(f"Avg retrieval precision: {avg_precision:.1%}")
    print(f"Avg faithfulness score:  {avg_faithfulness:.1%}")
    print(f"Low hallucination rate:  {low_hallucination:.1%}")

    print(f"\nPer question breakdown:")
    for r in all_results:
        print(f"\nQ: {r['question']}")
        print(f"   Generated: {r['generated'][:80]}...")
        print(f"   Precision: {r['retrieval_precision']:.1%} | Faithfulness: {r['faithfulness_score']:.1%} | Hallucination: {r['hallucination_risk']}")