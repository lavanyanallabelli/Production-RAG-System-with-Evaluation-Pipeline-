# ab_test.py
from evaluator import evaluate_one
from generator import generate_answer
from embedder import embed_texts
from vector_store import search_vector_store
from bm25_store import search_bm25
from hybrid import reciprocal_rank_fusion
from reranker import rerank
from prompts import PROMPT_VERSIONS


def run_ab_test(
    test_questions: list,
    index,
    stored_chunks: list,
    bm25,
    bm25_chunks: list,
    versions: list = None,
) -> dict:

    versions = versions or list(PROMPT_VERSIONS.keys())
    all_version_results = {}

    for version in versions:
        print(f"\n{'='*40}")
        print(f"Testing version: {version}")
        print(f"Notes: {PROMPT_VERSIONS[version]['notes']}")
        print(f"{'='*40}")

        version_results = []

        for test in test_questions:
            question = test["question"]
            expected = test["expected"]

            # retrieve — same for all versions
            query_embedding = embed_texts([question])[0]
            vector_results = search_vector_store(index, stored_chunks, query_embedding, top_k=10)
            bm25_results = search_bm25(bm25, bm25_chunks, question, top_k=10)
            hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results)
            final_chunks = rerank(question, hybrid_results, top_k=4)

            # generate — different prompt version each time
            generated = generate_answer(question, final_chunks, prompt_version=version)

            # evaluate
            eval_result = evaluate_one(
                question,
                expected,
                final_chunks,
                generated["answer"],
            )

            eval_result["retries"] = generated["retries"]
            version_results.append(eval_result)

        # aggregate metrics for this version
        avg_precision = sum(
            r["retrieval_precision"] for r in version_results
        ) / len(version_results)

        avg_faithfulness = sum(
            r["faithfulness_score"] for r in version_results
        ) / len(version_results)

        low_hallucination_rate = sum(
            1 for r in version_results
            if r["hallucination_risk"] == "low"
        ) / len(version_results)

        avg_retries = sum(
            r["retries"] for r in version_results
        ) / len(version_results)

        all_version_results[version] = {
            "version": version,
            "notes": PROMPT_VERSIONS[version]["notes"],
            "metrics": {
                "avg_precision": round(avg_precision, 3),
                "avg_faithfulness": round(avg_faithfulness, 3),
                "low_hallucination_rate": round(low_hallucination_rate, 3),
                "avg_retries": round(avg_retries, 3),
            },
            "individual_results": version_results,
        }

        print(f"\nResults:")
        print(f"  Precision:        {avg_precision:.1%}")
        print(f"  Faithfulness:     {avg_faithfulness:.1%}")
        print(f"  Low hallucination:{low_hallucination_rate:.1%}")
        print(f"  Avg retries:      {avg_retries:.2f}")

    # find winner by faithfulness score
    winner = max(
        all_version_results.keys(),
        key=lambda v: all_version_results[v]["metrics"]["avg_faithfulness"]
    )

    return {
        "winner": winner,
        "results": all_version_results,
        "comparison": build_comparison_table(all_version_results),
    }


def build_comparison_table(results: dict) -> list:
    table = []

    for version, data in results.items():
        table.append({
            "version": version,
            "notes": data["notes"],
            **data["metrics"],
        })

    # sort by faithfulness
    return sorted(table, key=lambda x: x["avg_faithfulness"], reverse=True)


def print_final_report(ab_results: dict):
    print(f"\n{'='*50}")
    print(f"FINAL REPORT")
    print(f"{'='*50}")
    print(f"Winner: {ab_results['winner']}")

    print(f"\nRanking by faithfulness:")
    print(f"{'Version':<8} {'Precision':<12} {'Faithful':<12} {'No Halluc':<12} {'Retries'}")
    print(f"{'-'*60}")

    for row in ab_results["comparison"]:
        print(
            f"{row['version']:<8} "
            f"{row['avg_precision']:.1%}      "
            f"{row['avg_faithfulness']:.1%}      "
            f"{row['low_hallucination_rate']:.1%}      "
            f"{row['avg_retries']:.2f}"
        )

    print(f"\nConclusion:")
    winner = ab_results["winner"]
    winner_data = ab_results["results"][winner]
    print(f"  {winner} performed best — {winner_data['notes']}")


if __name__ == "__main__":
    from pdf_loader import extract_text_from_pdf
    from chunker import chunk_overlap
    from embedder import embed_chunks
    from vector_store import build_vector_store
    from bm25_store import build_bm25_index

    # build pipeline
    text = extract_text_from_pdf("OnePiece.pdf")
    chunks = chunk_overlap(text, chunk_size=500, overlap=50)
    embedded = embed_chunks(chunks)

    index, stored_chunks = build_vector_store(embedded)
    bm25, bm25_chunks = build_bm25_index(chunks)

    # your test questions
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

    # run the test
    results = run_ab_test(
        test_questions,
        index,
        stored_chunks,
        bm25,
        bm25_chunks,
    )

    print_final_report(results)