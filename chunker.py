from pdf_loader import extract_text_from_pdf

def chunk_fixed_size(text:str, chunk_size:int = 500) -> list:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "text": chunk,
                "index": len(chunks),
                "start_char": start,
                "end_char": end,
            })
        start = end
    return chunks

def chunk_overlap(text:str, chunk_size = 500, overlap:int = 50) -> list:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "text": chunk,
                "index": len(chunks),
                "start_char": start,
                "end_char": end,
                "strategy": "overlap",
            })

        start = start + chunk_size - overlap

    return chunks

def chunk_paragraph(text:str, max_chunk_size: int = 500) -> list:
    #split on double newlines - natural paragraph boundary
    paragraphs = text.split("\n\n")

    # clean each paragraph - remove extra whitespaces
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []

    for para in paragraphs:
        if len(para) <= max_chunk_size:
            chunks.append({
                "text": para,
                "index": len(chunks),
                "strategy": "paragraph",
            })
        else:
            # paragraph is too long - split it with overlap
            sub_chunks = chunk_overlap(para, max_chunk_size, overlap=50)
            for sub in sub_chunks:
                chunks.append({
                    "text": sub["text"],
                    "index": len(chunks),
                    "strategy": "paragraph+overflow",
                })
    return chunks

if __name__ == "__main__":
    text = extract_text_from_pdf("OnePiece.pdf")

    fixed_chunks = chunk_fixed_size(text, chunk_size=500)
    overlap_chunks = chunk_overlap(text, chunk_size=500, overlap=50)
    para_chunks = chunk_paragraph(text, max_chunk_size=500)

    print("=== COMPARISON ===")
    print(f"Fixed size chunks:  {len(fixed_chunks)}")
    print(f"Overlap chunks:     {len(overlap_chunks)}")
    print(f"Paragraph chunks:   {len(para_chunks)}")

    print("\n=== PARAGRAPH CHUNK EXAMPLE ===")
    print(f"Chunk 0:\n{para_chunks[0]['text']}")