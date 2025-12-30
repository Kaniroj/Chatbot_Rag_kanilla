import lancedb
from sentence_transformers import SentenceTransformer
from src.config import settings

db = lancedb.connect(str(settings.DATA_PATH / "lancedb"))
tbl = db.open_table("transcripts")

model = SentenceTransformer("all-MiniLM-L6-v2")
q = "Vad är data engineering?"
q_emb = model.encode([q], normalize_embeddings=True).tolist()[0]

res = tbl.search(q_emb).limit(5).to_pandas()
print(res[["source", "chunk_index", "text"]].head())
