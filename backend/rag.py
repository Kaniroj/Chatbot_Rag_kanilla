from pydantic_ai import Agent
from backend.data_models import RagResponse
from backend.constants import VECTOR_DATABASE_PATH
import lancedb


vector_db = lancedb.connect(uri=VECTOR_DATABASE_PATH)


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



@rag_agent.tool_plain
def retrieve_top_documents(query: str, k=3) -> str:
    """
    Uses vector search to find the closest k matching documents to the query
    """
    results = vector_db["articles"].search(query=query).limit(k).to_list()

    return f"""
    
    Filename: {results[0]["filename"]},
    
    Filepath: {results[0]["filepath"]},

    Content: {results[0]["content"]}
    
    """