from __future__ import annotations


from typing import List

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

from backend.config import settings
from knowledge_base.retriever import retrieve, format_context, RetrievedChunk


SYSTEM_PROMPT = """
You are a nerdy, playful Swedish data-engineering YouTuber helping students learn.
You answer questions based ONLY on the provided transcript context.
If the context is insufficient, say you don't know based on the transcripts.
Be clear, practical, and slightly humorous (but not cringe).
Always cite sources in the format: (source_file, chunk_index).
""".strip()


model = GeminiModel(settings.chat_model)

agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
)


async def answer_question(question: str, k: int = 5) -> str:
    chunks: List[RetrievedChunk] = retrieve(question, k=k)
    context = format_context(chunks)

    sources_text = "\n".join(
        f"- ({c.source_file}, chunk {c.chunk_index})"
        for c in chunks
    )

    prompt = f"""
TRANSCRIPT CONTEXT:
{context}

USER QUESTION:
{question}

Answer the question using the context above.
At the end of your answer, include a section called "Sources" and list:
{sources_text}
""".strip()

    result = await agent.run(prompt)
    return result.data
