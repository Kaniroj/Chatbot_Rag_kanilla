from dotenv import load_dotenv
load_dotenv(".env", override=True)

import os
from typing import List

import lancedb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from pydantic_ai import Agent
from src.config import settings

load_dotenv()

SYSTEM_PROMPT = """
Du är Kokchun Giang (data engineering-youtuber).
Svara kort, pedagogiskt och lite nördigt.
Använd ENDAST information från CONTEXT. Om det inte finns i CONTEXT: säg att du inte vet baserat på materialet.
"""

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt=SYSTEM_PROMPT,
)

model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query: str, k: int = 5) -> List[str]:
    db = lancedb.connect(str(settings.DATA_PATH / "lancedb"))
    tbl = db.open_table("transcripts")

    q_emb = model.encode([query], normalize_embeddings=True).tolist()[0]
    rows = tbl.search(q_emb).limit(k).to_list()

    ctx = []
    for r in rows:
        ctx.append(f"SOURCE: {r['source']} (chunk {r['chunk_index']})\n{r['text']}")
    return ctx


def ask(query: str) -> str:
    context_blocks = retrieve(query, k=5)
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
    result = agent.run_sync(prompt)
    return result.data


if __name__ == "__main__":
    while True:
        q = input("\nFråga (enter för att avsluta): ").strip()
        if not q:
            break
        print("\nSvar:\n", ask(q))
