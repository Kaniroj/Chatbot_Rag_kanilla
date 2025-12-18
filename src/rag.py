from __future__ import annotations

from typing import List

import lancedb
from pydantic_ai import Agent

from backend.config import settings
from backend.constants import VECTOR_DATABASE_PATH
from backend.data_models import RagResponse, EMBEDDING_DIM

# Google GenAI (nya SDK:t)
from google import genai


vector_db = lancedb.connect(uri=VECTOR_DATABASE_PATH)

# Initiera client en gång
genai_client = genai.Client(api_key=settings.api_key)

# VIKTIGT: välj embedding-modell här (ska matcha din ingestion)
EMBED_MODEL = settings.embed_model  # t.ex. "models/text-embedding-004"


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


def embed_text(text: str) -> List[float]:
    """
    Skapar embedding med google.genai.
    Returnerar en vektor (list[float]) som ska matcha EMBEDDING_DIM.
    """
    res = genai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    vec = res.embeddings[0].values

    # Safety check: dimension-mismatch ska faila tidigt, inte ge konstiga sökresultat.
    if len(vec) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding-dimension mismatch: got {len(vec)} but expected {EMBEDDING_DIM}. "
            f"Check EMBEDDING_DIM in data_models.py and embed_model in config.py."
        )
    return vec


@rag_agent.tool_plain
def retrieve_top_documents(query: str, k: int = 3) -> str:
    """
    Gör vector search: embedda query -> sök mot 'embedding' kolumnen -> returnera topp-k.
    """
    query_vec = embed_text(query)

    results = (
        vector_db["articles"]
        .search(query_vec, vector_column_name="embedding")
        .limit(k)
        .to_list()
    )

    if not results:
        return "No relevant documents found in the vector database."

    # Bygg en kompakt “context blob” som LLM:en kan använda
    chunks = []
    for r in results:
        chunks.append(
            f"Filename: {r.get('filename')}\n"
            f"Filepath: {r.get('filepath')}\n"
            f"Content:\n{r.get('content')}\n"
        )

    return "\n---\n".join(chunks)
import lancedb
from pydantic import BaseModel
from pydantic_ai import Agent
from sentence_transformers import SentenceTransformer

from .config import settings
from .prompts import SYSTEM_PROMPT

class RAGAnswer(BaseModel):
    answer: str
    sources: list[str]  # t.ex. ["video1_3", "video2_7"]

class RAGBot:
    def __init__(self):
        self.db = lancedb.connect(settings.lancedb_dir)
        self.tbl = self.db.open_table(settings.table_name)
        self.embedder = SentenceTransformer(settings.embed_model)

        # PydanticAI agent (LLM)
        self.agent = Agent(
            model=settings.openai_model,
            system_prompt=SYSTEM_PROMPT,
            result_type=RAGAnswer,
        )

    def retrieve(self, question: str, k: int = 5):
        qvec = self.embedder.encode([question], normalize_embeddings=True).tolist()[0]
        res = (
            self.tbl.search(qvec)
            .limit(k)
            .select(["id", "video_id", "chunk_index", "text"])
            .to_list()
        )
        return res

    async def answer(self, question: str, k: int = 5) -> RAGAnswer:
        hits = self.retrieve(question, k=k)

        context_blocks = []
        sources = []
        for h in hits:
            sources.append(h["id"])
            context_blocks.append(
                f"[source={h['id']} video_id={h['video_id']} chunk_index={h['chunk_index']}]\n{h['text']}"
            )

        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""
KONTEXT:
{context}

FRÅGA:
{question}

Svara på svenska.
"""
        result = await self.agent.run(prompt)
        # säkerställ sources även om modellen glömmer
        out = result.data
        if not out.sources:
            out.sources = sources
        return out
