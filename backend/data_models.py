from pydantic import BaseModel, Field
from lancedb.pydantic import LanceModel, Vector

EMBEDDING_DIM = 768  # OBS: måste matcha embedding-modellen du använder


class Article(LanceModel):
    doc_id: str
    filepath: str
    filename: str = Field(description="the stem of the file i.e. without the suffix")
    content: str
    embedding: Vector(EMBEDDING_DIM)


class Prompt(BaseModel):
    prompt: str = Field(description="prompt from user, if empty consider prompt as missing")


class RagResponse(BaseModel):
    filename: str = Field(description="filename of the retrieved file without suffix")
    filepath: str = Field(description="absolute path to the retrieved file")
    answer: str = Field(description="answer based on the retrieved file")
