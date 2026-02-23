import faiss
import pickle
import ollama
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "deepseek-r1:1.5b"

# Cache embedder so it doesn't reload every request
embedder = SentenceTransformer(MODEL_NAME)


def load_vector_store(index_path, docs_path):
    index = faiss.read_index(index_path)
    with open(docs_path, "rb") as f:
        documents = pickle.load(f)
    return index, documents


def retrieve_context(query, index, documents, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    return [documents[i] for i in indices[0]]


def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def generate_answer(query, context):
    formatted_context = "\n".join(context)

    prompt = f"""
You are an expert software company assistant.
Answer ONLY from the given context.

Context:
{formatted_context}

Question: {query}
"""

    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"temperature": 0.3, "max_tokens": 2000}
    )

    return remove_think_tags(response["response"])


def get_available_tags():
    parent_folder = "uploaded_pdfs"
    return [
        name for name in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, name))
    ]
