import os
import requests
from dotenv import load_dotenv
from embedder import embed_chunks
from indexer import search_index

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "nex-agi/deepseek-v3.1-nex-n1:free"

def generate_answer(query, k=3):
    query_embedding = embed_chunks([query])[0]
    top_chunks = search_index(query_embedding, k)

    context = "\n".join(top_chunks)
    
    prompt = f"""You are a helpful assistant answering questions based only on the provided context.

Context:
{context}

Question: {query}

Answer the question with clear, step-by-step reasoning.
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You answer questions based only on the context using logical reasoning. Keep your answers short and concise."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(OPENROUTER_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError:
        return f"Error details: {response.json()}"