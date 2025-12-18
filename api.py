from fastapi import FastAPI, HTTPException

from backend.rag import rag_agent
from backend.data_models import Prompt

app = FastAPI(
    title="RAG API",
    description="API för att köra RAG-frågor mot dokumentation",
    version="1.0.0",
)


@app.get("/health", tags=["system"])
async def health_check():
    """
    Enkel hälsokontroll för att verifiera att API:t kör.
    """
    return {"status": "ok"}


@app.post("/rag/query", tags=["rag"])
async def query_rag(prompt: Prompt):
    """
    Kör en RAG-fråga baserat på användarens prompt.
    """
    try:
        result = await rag_agent.run(prompt.prompt)
        return {"output": result.output}
    except Exception as exc:
        # Fail fast, men med ett begripligt fel
        raise HTTPException(
            status_code=500,
            detail=f"RAG-agenten misslyckades: {str(exc)}",
        )
