from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import lancedb
from sentence_transformers import SentenceTransformer

from src.config import settings


def chunk_text(text: str, chunk_words: int = 180, overlap_words: int = 40):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = max(end - overlap_words, start + 1)
    return chunks



def main() -> None:
    data_dir = settings.DATA_PATH / "txt"
    db_dir = settings.DATA_PATH / "lancedb"
    db_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(data_dir.glob("*.txt"))
    print(f"Reading from: {data_dir}")
    print(f"Found {len(txt_files)} .txt files")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    rows: List[Dict] = []
    for fp in txt_files:
        text = fp.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            rows.append(
                {
                    "id": f"{fp.stem}::chunk{i:04d}",
                    "source": fp.name,
                    "chunk_index": i,
                    "text": chunk,
                }
            )

    print(f"Total chunks: {len(rows)}")
    texts = [r["text"] for r in rows]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    for r, emb in zip(rows, embeddings):
        r["vector"] = emb

    db = lancedb.connect(str(db_dir))
    table = db.create_table("transcripts", data=rows, mode="overwrite")
    print("Saved to LanceDB table:", table.name)


if __name__ == "__main__":
    main()
