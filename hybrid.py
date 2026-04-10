#Algorith - Reciprocal Rank Fusinon(RRF)
# RRF fixes this by throwing away the raw scores and only using rank positions.
from vector_store import search_vector_store
from bm25_store import search_bm25

def reciprocal_rank_fusion(
    vector_results: list, #ranked list from FAISS search
    bm25_results: list, #ranked list from BM25 search
    k: int = 60, #smoothing constant. Standard value used in research papers. Higher k = smaller difference between ranks. 
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,

) -> list:

    fused_scores = {}
    chunk_data = {}

    #process vector results 
    #enumerate() → gives both the index and the value at once.
    for rank, result in enumerate(vector_results):
        idx = result["index"] #get the chunks's position number
        rrf_score = vector_weight * (1 / (k + rank + 1)) #rrf formula

        if idx not in fused_scores: #First time seeing this chunk — create entry with 0.
            fused_scores[idx] = 0
        fused_scores[idx] += rrf_score #Every time — add this round's RRF score on top.
        chunk_data[idx] = result

        #process bm25 results
        for rank, result in enumerate(bm25_results):
            #same process as vector results
            idx = result["index"]
            
            rrf_score = bm25_weight * (1 / (k + rank +1))

            if idx not in fused_scores:
                fused_scores[idx] = 0
            fused_scores[idx] += rrf_score

            #only store chunk data if we haven't seen it before. FAISS version takes priority since it has the vector score attached.
            if idx not in chunk_data:
                chunk_data[idx] = result

        #sort by fused score
        sorted_chunks = sorted(
            fused_scores.items(), #converts the dictionary to a list of pairs:
            key = lambda x: x[1],
            reverse = True #highest score first.
        )

        results = []
        for idx, score in sorted_chunks:
            results.append({
                **chunk_data[idx],
                "hybrid_score" : score,
            })
        return results

if __name__ == "__main__":
    from pdf_loader import extract_text_from_pdf
    from chunker import chunk_overlap
    from embedder import embed_chunks, embed_texts
    from vector_store import build_vector_store
    from bm25_store import build_bm25_index

    #build everything
    text = extract_text_from_pdf("OnePiece.pdf")
    chunks = chunk_overlap(text, chunk_size=500, overlap=50)
    embedded = embed_chunks(chunks)

    index, stored_chunks = build_vector_store(embedded)
    bm25, stored_chunks = build_bm25_index(chunks)

    #search both
    query = "Who is the captain of the Straw Hat Pirates?"

    query_embedding = embed_texts([query])[0]
    vector_results = search_vector_store(index, stored_chunks, query_embedding, top_k=3)
    bm25_results = search_bm25(bm25, stored_chunks, query, top_k=10)

    #merge
    hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results)

    print(f"Query: {query}")
    print(f"\nVector results: {len(vector_results)}")
    print(f"BM25 results: {len(bm25_results)}")
    print(f"Hybrid results: {len(hybrid_results)}")

    print(f"\nTop 3 Hybrid results:")
    for i, result in enumerate(hybrid_results[:3]):
        print(f"\nResult {i+1} (hybrid_score: {result['hybrid_score']:.3f}):")
        print(result["text"][:200])