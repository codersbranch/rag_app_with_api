from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from rag_engine import (
    load_vector_store,
    retrieve_context,
    generate_answer,
    get_available_tags
)

app = FastAPI(title="Software company RAG Chatbot API")


# -----------------------------
# Request Model
# -----------------------------
class QueryRequest(BaseModel):
    tag: str
    question: str


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {"status": "API is running"}


# -----------------------------
# Get Available Tags
# -----------------------------
@app.get("/tags")
def list_tags():
    return {"tags": get_available_tags()}


# -----------------------------
# Ask Question Endpoint
# -----------------------------
@app.post("/ask")
def ask_question(request: QueryRequest):
    index_path = os.path.join("vector_data", f"{request.tag}_index/vectors.index")
    docs_path = os.path.join("vector_data", f"{request.tag}_index/metadata.pkl")

    if not os.path.exists(index_path) or not os.path.exists(docs_path):
        raise HTTPException(status_code=404, detail="Tag data not found")

    index, documents = load_vector_store(index_path, docs_path)
    context = retrieve_context(request.question, index, documents)
    answer = generate_answer(request.question, context)

    return {
        "question": request.question,
        "tag": request.tag,
        "answer": answer   
    }
