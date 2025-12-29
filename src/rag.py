from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import lancedb
from google import genai
from pydantic_ai import Agent

from .config import settings
from .constants import VECTOR_DATABASE_PATH
from .data_models import RagResponse, EMBEDDING_DIM


# -----------------------------
# Init: DB + GenAI client
# -----------------------------

_db = lancedb.connect(uri=VECTOR_DATABASE_PATH)

# Initiera client en gång (snabbare + stabilare)
_genai_client = genai.Client(api_key=settings.api_key)

# VIKTIGT: måste matcha ingestion (samma embedding-modell)
_EMBED_MODEL = settings.embed_model  # t.ex. "models/text-embedding-004"


# -----------------------------
# Agent (LLM) – svarar ENBART från hämtad kontext
# -----------------------------

rag_agent = Agent(
    model="google-gla:gemini-2.5-flash",
    retries=2,
    system_prompt=(
        "You are Kokchun, a teacher in data engineering with deep expertise in the subject. "
        "Always answer strictly based on the retrieved course material. "
        "You may use your teaching experience to make the explanation clearer, but never invent information. "
        "If the retrieved sources are not sufficient to answer the question, say so explicitly. "
        "Keep the answer clear, concise, and straight to the point, with a maximum of 6 sentences. "
        "Always mention which file or material was used as the source."
    ),
    output_type=RagResponse,
)


# -----------------------------
# Embeddings
# -----------------------------

def embed_text(text: str) -> List[float]:
    """Skapar embedding med google.genai. Returnerar list[float] med EMBEDDING_DIM."""
    res = _genai_client.models.embed_content(
        model=_EMBED_MODEL,
        contents=text,
    )
    vec = res.embeddings[0].values

    if len(vec) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding-dimension mismatch: got {len(vec)} but expected {EMBEDDING_DIM}. "
            f"Check EMBEDDING_DIM in data_models.py and embed_model in config.py."
        )
    return vec


# -----------------------------
# Retrieval helpers
# -----------------------------

@dataclass(frozen=True)
class RetrievedChunk:
    filename: str
    filepath: str
    content: str


def _open_articles_table():
    """
    Öppnar tabellen 'articles'. Antag: ingestion har skapat tabellen.
    Om du använder annat namn, ändra här.
    """
    return _db.open_table("articles")


def retrieve_chunks(query: str, k: int = 3) -> List[RetrievedChunk]:
    """Vector search: embed query -> sök mot kolumnen 'embedding' -> topp-k chunks."""
    query_vec = embed_text(query)
    tbl = _open_articles_table()

    results = (
        tbl.search(query_vec, vector_column_name="embedding")
        .limit(k)
        .to_list()
    )

    chunks: List[RetrievedChunk] = []
    for r in results:
        chunks.append(
            RetrievedChunk(
                filename=str(r.get("filename", "")),
                filepath=str(r.get("filepath", "")),
                content=str(r.get("content", "")),
            )
        )
    return chunks


def _build_context(chunks: List[RetrievedChunk]) -> Tuple[str, List[str]]:
    """
    Bygger kompakt kontext + källor.
    Källor: ["filename (filepath)", ...] (unika, i ordning)
    """
    sources: List[str] = []
    blocks: List[str] = []

    for c in chunks:
        src = f"{c.filename} ({c.filepath})".strip()
        if src and src not in sources:
            sources.append(src)

        blocks.append(
            "SOURCE:\n"
            f"Filename: {c.filename}\n"
            f"Filepath: {c.filepath}\n\n"
            "CONTENT:\n"
            f"{c.content}\n"
        )

    context = "\n---\n".join(blocks) if blocks else ""
    return context, sources


# -----------------------------
# Tool (valfritt, men bra för spårbarhet)
# -----------------------------

@rag_agent.tool_plain
def retrieve_top_documents(query: str, k: int = 3) -> str:
    """
    Gör retrieval och returnerar en sammanfogad context-string.
    Agenten kan anropa detta verktyg, men vi kan också anropa retrieval direkt i RAGBot.
    """
    chunks = retrieve_chunks(query, k=k)
    if not chunks:
        return "No relevant documents found in the vector database."

    context, _ = _build_context(chunks)
    return context


# -----------------------------
# RAGBot (det du importerar i api.py)
# -----------------------------

class RAGBot:
    """
    Enkel wrapper som:
      1) hämtar topp-k chunks från LanceDB
      2) bygger prompt med kontext
      3) kör agenten och returnerar RagResponse (inkl källor)
    """

    def __init__(self, k_default: int = 3):
        self.k_default = k_default

    async def answer(self, question: str, k: int | None = None) -> RagResponse:
        k = k or self.k_default

        chunks = retrieve_chunks(question, k=k)
        if not chunks:
            # Här förutsätter vi att RagResponse kan initieras så här.
            # Om din RagResponse har andra fält, justera.
            return RagResponse(
                answer="Jag hittar inga relevanta dokument i vektordatabasen för att kunna svara säkert.",
                sources=[],
            )

        context, sources = _build_context(chunks)

        prompt = (
            "KONTEXT (endast detta får användas):\n"
            f"{context}\n\n"
            "FRÅGA:\n"
            f"{question}\n\n"
            "Instruktion: Svara på svenska."
        )

        result = await rag_agent.run(prompt)
        out: RagResponse = result.data

        # Säkerställ att sources alltid finns (även om modellen glömmer)
        if getattr(out, "sources", None) in (None, [], ()):
            try:
                out.sources = sources  # type: ignore[attr-defined]
            except Exception:
                pass

        return out
