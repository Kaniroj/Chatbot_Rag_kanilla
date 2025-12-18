from pathlib import Path

# Root of the project (rag_youtuber_kanilla/)
ROOT_DIR = Path(__file__).resolve().parents[1]

# Folder with raw source documents (.md, .pdf, etc.)
DATA_PATH = ROOT_DIR / "data"

# Folder for vector DB / processed knowledge
VECTOR_DATABASE_PATH = ROOT_DIR / "knowledge_base"

# Optional safety checks (can be removed later)
if not DATA_PATH.exists():
    raise FileNotFoundError(f"DATA_PATH does not exist: {DATA_PATH}")

if not VECTOR_DATABASE_PATH.exists():
    VECTOR_DATABASE_PATH.mkdir(parents=True, exist_ok=True)
