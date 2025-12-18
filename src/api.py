from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from .rag import RAGBot

load_dotenv()

app = FastAPI(title="RAG Youtuber API")
bot = RAGBot()

class AskRequest(BaseModel):
    question: str
    k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
async def ask(req: AskRequest):
    ans = await bot.answer(req.question, k=req.k)
    return ans.model_dump()
