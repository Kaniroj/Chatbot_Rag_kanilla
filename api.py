from fastapi import FastAPI
from pydantic import BaseModel

from rag_chat_gemini import ask  # återanvänd din RAG-funktion

app = FastAPI(title="Kokchun RAG API")

class QuestionIn(BaseModel):
    question: str

class AnswerOut(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerOut)
def ask_endpoint(payload: QuestionIn):
    return {"answer": ask(payload.question)}

@app.get("/health")
def health():
    return {"status": "ok"}
