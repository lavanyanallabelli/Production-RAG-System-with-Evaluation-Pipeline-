# sentence-transformer reranker - cross-encoder model

from sentence_transformers import CrossEncoder

# load model once at module level
# loading is expensive - do it once and resue
print("Loading cross-encoder model...")
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-encoder model loaded successfully.")

def rerank(query: str, chunks: list, top_k: int = 3) -> list:
    if not chunks: 
        return []

    # create pairs of (query, chunk_text)
    pairs = [(query, chunk["text"]) for chunk in chunks]

    #score every pair
    scores = model.predict(pairs)

    #add score to each chunk
    for i, score in enumerate(scores):
        chunks[i]["rerank_score"] = float(score)

    #sort by rerank score descending
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_k]

if __name__ == "__main__":
    from pdf_loader import extract_text_from_pdf
    from chunker import chunk_overlap
    from embedder import embed_chunks, embed_texts
    from vector_store import build_vector_store, search_vector_store
    from bm25_store import build_bm25_index, search_bm25
    from hybrid import reciprocal_rank_fusion

    #build everything
    text = extract_text_from_pdf("OnePiece.pdf")
    chunks = chunk_overlap(text, chunk_size=500, overlap=50)
    embedded = embed_chunks(chunks)

    index, stored_chunks = build_vector_store(embedded)
    bm25, bm25_chunks = build_bm25_index(chunks)

    #search
    query = "Who is the captain of the Straw Hat Pirates?"

    query_embedding = embed_texts([query])[0]
    vector_results = search_vector_store(index, stored_chunks, query_embedding, top_k=10)
    bm25_results = search_bm25(bm25, bm25_chunks, query, top_k=10)
    hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results)

    print(f"Before reranking -top 3:")
    for i, r in enumerate(hybrid_results[:3]):
        print(f"\nResult {i+1} (hybrid_score: {r['hybrid_score']:.4f})")
        print(r["text"][:150])

    reranked = rerank(query, hybrid_results, top_k = 3)

    print(f"\nAfter reranking -top 3:")
    for i, r in enumerate(reranked[:3]):
        print(f"\nResult {i+1} (rerank_score: {r['rerank_score']:.4f})")
        print(r["text"][:150])

