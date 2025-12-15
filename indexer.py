import faiss
import numpy as np
import pickle
import os

INDEX_PATH = "vector_store/index.faiss"
META_PATH = "vector_store/chunks.pkl"

def save_index(embeddings, chunks):
    os.makedirs("vector_store", exist_ok=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved {len(chunks)} chunks to index.")

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def search_index(query_embedding, k=3):
    index, chunks = load_index()
    D, I = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in I[0]]
