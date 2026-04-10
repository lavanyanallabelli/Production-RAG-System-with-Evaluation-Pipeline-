import numpy as np
import faiss
import json
import os

def build_vector_store(embedded_chunks: list) -> tuple: # return both index and embedded chunks
    # extract embeddings from chunks
    embeddings = [chunk["embedding"] for chunk in embedded_chunks]

    # convert to numpy array 
    vectors = np.array(embeddings, dtype=np.float32)

    # get dimension of embeddings
    # shape tell you the size of a numpy array in each dieaction.
    dimension = vectors.shape[1]

    # create index flat - checks every single vector, IP - inner product (math used to measure similarity)
    index = faiss.IndexFlatIP(dimension)

    # Normalization scales every vector so its length equals exactly 1. This is required for inner product to equal cosine similarity.
    # Without normalization — inner product gives wrong results.
    # With normalization — inner product equals cosine similarity — which is what we want.
    # Think of it like this. Two people walking in the same direction — one walks 10 steps, one walks 2 steps. They're going the same direction (similar meaning) but different distances (different text lengths). Normalization makes both distances equal to 1 so you only compare direction, not distance.
    faiss.normalize_L2(vectors)

    # add vectors to index, This loads all 500 normalized vectors into the FAISS index. Now FAISS knows about all your chunks and can search them.
    index.add(vectors)

    print(f"Built FAISS index with {index.ntotal} vectors")
    return index, embedded_chunks

def search_vector_store(index, chunks: list, query_embedding: list, top_k:int = 3) -> list:

    #we wrap [query_embedding] in an extra list. this because FAISS expects a 2D array even for a single query.  It's designed to handle multiple queries at once.
    query_vector = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_vector)

    # search, FAISS compares your question vector against every chunk vector and returns the top_k closest ones.
    distances, indices = index.search(query_vector, top_k)

    #distances = [[0.95, 0.87, 0.76, 0.71]]   # similarity scores
    #indices   = [[42,   18,   91,   7   ]]    # positions in original array
    #Chunk at position 42 has score 0.95 — most similar.
    #Chunk at position 18 has score 0.87 — second most similar.
    #Both are 2D arrays — [0] gets results for our single query.

    results = []

    for score, idx in zip(distances[0], indices[0]): # pairs scores with position
        if idx == -1: # FAISS returns -1 for empty slots when there aren't enough results. We skip those.
            continue
        results.append({
            **chunks[idx], # get the chunk dictionary at that position., spread all its keys into the new dictionary.
            "score": float(score), #dd the similarity score.
        })
    return results


def save_vector_store(index, chunks: list, index_path: str, chunks_path: str):

    #creates the storage folder. exist_ok=True means don't throw an error if the folder already exists.
    os.makedirs("storage", exist_ok=True)

    #saves the FAISS index to disk. FAISS handles its own binary format.
    faiss.write_index(index, index_path)

    #Why remove embeddings? Because FAISS already stores them in the index file.
    chunks_to_save = [{k: v for k, v in chunk.items() if k!= "embedding"} for chunk in chunks]

    # opens file for writing. The with block automatically closes the file when done even if an error occurs.
    with open(chunks_path, "w") as f:
        json.dump(chunks_to_save, f)

    print(f"Saved index and {len(chunks)} chunks")


def load_vector_store(index_path: str, chunks_path: str) -> tuple:
    index = faiss.read_index(index_path)

    with open(chunks_path, "r") as f:
        chunks = json.load(f)

    #Load the FAISS index from its binary file. Load the chunks from JSON. Return both.
    #json.load(f) — reads JSON from a file and converts to Python list/dict. Like JSON.parse but reads from a file.
    #index.ntotal — property on the FAISS index telling you how many vectors are stored in it.

    print(f"Load index with {index.ntotal} vectors")
    return index, chunks




if __name__ == "__main__":
    from pdf_loader import extract_text_from_pdf
    from chunker import chunk_overlap
    from embedder import embed_chunks

    #full pipeline
    text = extract_text_from_pdf("OnePiece.pdf")
    chunks = chunk_overlap(text, chunk_size=500, overlap=50)
    embedded = embed_chunks(chunks)

    # build
    index, stored_chunks = build_vector_store(embedded)

    #save
    save_vector_store(index, stored_chunks, "storage/faiss_index", "storage/chunks.json")

    # search test
    from embedder import embed_texts
    query = "Who is the captain of the straw hat pirates?"
    query_embedding = embed_texts([query])[0]

    results = search_vector_store(index, stored_chunks, query_embedding, top_k=3)

    print(f"\nQuery: {query}")
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (score: {result['score']:.3f}):")
        print(result["text"][:200])