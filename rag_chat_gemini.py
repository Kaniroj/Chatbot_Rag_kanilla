import os
from typing import List

import lancedb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

from src.config import settings

# ladda .env
load_dotenv()

# init Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
Du är Kokchun Giang, en data engineering-youtuber.

Regler:
- Svara ENDAST med information som finns i CONTEXT.
- Om CONTEXT inte räcker: skriv exakt "Jag hittar inte detta i videomaterialet." och inget mer.
- Lägg alltid till en rad "Källor:" med 1-3 SOURCE-filer du använde.
- Hitta aldrig på definitioner (inga generella fakta utanför CONTEXT).
"""


embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query: str, k: int = 10) -> List[str]:
    db = lancedb.connect(str(settings.DATA_PATH / "lancedb"))
    tbl = db.open_table("transcripts")

    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()[0]
    rows = tbl.search(q_emb).limit(k).to_list()

    ctx = []
    for r in rows:
        ctx.append(
            f"SOURCE: {r['source']} (chunk {r['chunk_index']})\n{r['text']}"
        )
    return ctx


def ask(query: str) -> str:
    context_blocks = retrieve(query, k=10)
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{query}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


if __name__ == "__main__":
    while True:
        q = input("\nFråga (enter för att avsluta): ").strip()
        if not q:
            break
        print("\nSvar:\n", ask(q))
