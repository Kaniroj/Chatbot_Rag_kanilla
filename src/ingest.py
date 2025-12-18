from pathlib import Path
import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer

from .config import settings
from .utils import chunk_text

def ingest(transcript_dir: str = "data/transcripts"):
    db = lancedb.connect(settings.lancedb_dir)
    embedder = SentenceTransformer(settings.embed_model)

    rows = []
    for fp in Path(transcript_dir).glob("*.txt"):
        video_id = fp.stem
        text = fp.read_text(encoding="utf-8", errors="ignore")

        chunks = chunk_text(text, chunk_size=800, overlap=120)
        vectors = embedder.encode(chunks, normalize_embeddings=True).tolist()

        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            rows.append(
                {
                    "id": f"{video_id}_{i}",
                    "video_id": video_id,
                    "chunk_index": i,
                    "text": chunk,
                    "vector": vec,
                }
            )

    df = pd.DataFrame(rows)

    if settings.table_name in db.table_names():
        tbl = db.open_table(settings.table_name)
        tbl.add(df)
    else:
        db.create_table(settings.table_name, data=df)

    return len(df)

if __name__ == "__main__":
    n = ingest()
    print(f"Ingested {n} chunks into LanceDB.")
