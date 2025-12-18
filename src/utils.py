def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = max(0, end - overlap)
    return chunks
