import json  #for daving/loading simple data
import pickle  #for saving complex pythong objects like bm25 model itself. json can't handle it
import os #for creating folders.
from rank_bm25 import BM25Okapi #the specific BM25 algorithm from the rank_bm25 library. Okapi is the name of the research system that invented this version of BM25.

def build_bm25_index(chunks: list) -> tuple:

    #tokend each chunk - split into individual words
    tokenized = [chunk["text"].lower().split() for chunk in chunks]

# BM25 doesn't understand raw strings. It needs each chunk as a list of individual words. This is called tokenization — splitting text into tokens.
#You hand all the tokenized chunks to BM25Okapi. It analyzes them and builds an internal model that knows:

#How many times each word appears in each chunk
#How rare each word is across all chunks
#Rare words score higher than common words

#For example "the" appears in every chunk — low value. "nakama" appears in 2 chunks — high value when found.
    bm25 = BM25Okapi(tokenized)

    print(f"Built BM25 index with {len(chunks)} chunks")
    return bm25, chunks #returned trained bm25 model and the original chunks.

def search_bm25(bm25, chunks: list, query: str, top_k: int = 3) -> list:

    #tokenize query same way as chunks
    query_tokens = query.lower().split()

    #get score for every chunk
    scores = bm25.get_scores(query_tokens)

    #get top_k indecies sorted by score
    # argsort() → indices that would sort ascending
    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    #oop through the top indices. Skip any with score 0 — those had no matching words at all, not worth returning.
#chunks[idx] gets the chunk at that position.
#**chunks[idx] spreads all its keys into the new dict.
#"bm25_score" adds the score.
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                **chunks[idx],
                "bm25_score" : float(scores[idx]),
            })
    return results


def save_bm25_index(bm25, chunks: list, path: str):
    os.makedirs("storage", exist_ok=True)

    #wb - write binary
    with open(path, "wb") as f:
        #We save both the model and chunks together in one dictionary. That way loading is simple — one file gives you everything.
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"saved BM25 index to {path}")


def load_bm25_index(path: str) -> tuple:

    #rb - read binary
    with open(path, "rb") as f:
        data = pickle.load(f) #deserializes the binary back into the Python dictionary we saved. Then we pull out the two pieces and return them.

    print(f"Loaded BM25 index with {len(data['chunks'])} chunks")
    return data["bm25"], data["chunks"]

#Notice — BM25 does not need embeddings. No OpenAI API call happens here. BM25 works purely on word matching — completely free to run.

if __name__ == "__main__":
    from pdf_loader import extract_text_from_pdf
    from chunker import chunk_overlap

    text = extract_text_from_pdf("OnePiece.pdf")
    chunks = chunk_overlap(text, chunk_size=500, overlap=50)

    #build
    bm25, stored_chunks = build_bm25_index(chunks)

    #save
    save_bm25_index(bm25, stored_chunks, "storage/bm25_index.pkl")

    #search test
    query = "Who is the captain of the Straw Hat Pirates?"
    results = search_bm25(bm25, stored_chunks, query, top_k=3)

    print(f"\nQuery: {query}")
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (bm25_score: {result['bm25_score']:.3f}):")
        print(result["text"][:200])