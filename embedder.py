from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: list) -> list:
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = texts,
    )

    embeddings = [item.embedding for item in response.data]
    return embeddings

def embed_chunks(chunks: list) -> list:
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append({
            **chunk,
            "embedding": embedding
        })

    return embedded_chunks

if __name__=="__main__":
    from pdf_loader import extract_text_from_pdf
    from chunker import chunk_overlap

    text = extract_text_from_pdf("OnePiece.pdf")
    chunks = chunk_overlap(text, chunk_size=500, overlap=50)

    # just embed first 3 chunks to save cost
    simple_chunks = chunks[:3]
    embedded = embed_chunks(simple_chunks)

    print(f"Total chunks embedded: {len(embedded)}")
    print(f"\nFirst chunk text:\n{embedded[0]['text'][:100]}...")
    print(f"\nEmbedding length: {len(embedded[0]['embedding'])}")
    print(f"\nFirst 5 numbers of embedding:\n{embedded[0]['embedding'][:5]}")