from pydantic import BaseModel
import os

class Settings(BaseModel):
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Vector DB
    lancedb_dir: str = os.getenv("LANCEDB_DIR", "lancedb")
    table_name: str = os.getenv("LANCEDB_TABLE", "transcripts")

    # Embeddings
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

settings = Settings()
