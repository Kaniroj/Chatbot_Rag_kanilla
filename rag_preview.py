from sentence_transformers import SentenceTransformer
import lancedb
from src.config import settings


def retrieve(query: str, k: int = 5):
    db = lancedb.connect(str(settings.DATA_PATH / "lancedb"))
    tbl = db.open_table("transcripts")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([query], normalize_embeddings=True).tolist()[0]

    rows = tbl.search(q_emb).limit(k).to_list()
    return rows


if __name__ == "__main__":
    q = "Vad är PydanticAI?"
    results = retrieve(q)

    print("\nTOP RESULTS (no LLM call):\n")
    for r in results:
        print(f"- {r['source']} (chunk {r['chunk_index']})")
        print(r["text"][:300])
        print()
