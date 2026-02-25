import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import pickle
from typing import List, Dict

class EmbeddingStore:
    """
    Handles embeddings for rows of tables and allows retrieval using FAISS.
    """

    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", store_dir="hybrid-table-rag/embeddings/faiss_store"):
        self.model = SentenceTransformer(embedding_model_name)
        self.store_dir = store_dir
        self.index_file = os.path.join(self.store_dir, "faiss_index")
        self.meta_file = os.path.join(self.store_dir, "faiss_meta.pkl")
        self.index = None
        self.metadata = []

        # Ensure folder exists
        os.makedirs(self.store_dir, exist_ok=True)

        # Load store if exists
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self.load_store()

    def build_index(self, rows: List[str], metadatas: List[Dict], rebuild=True):
        embeddings = self.model.encode(rows, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]

        if rebuild or self.index is None:
            self.index = faiss.IndexFlatIP(dim)  # Cosine similarity via inner product
            self.index.add(embeddings)
            self.metadata = metadatas
        else:
            self.index.add(embeddings)
            self.metadata.extend(metadatas)

        self.save_store()

    def search(self, query: str, top_k: int = 5):
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, idxs = self.index.search(q_emb, top_k)
        results = []
        for dist, idx in zip(distances[0], idxs[0]):
            meta = self.metadata[idx]
            results.append({**meta, "score": float(dist)})
        return results

    def save_store(self):
        # Make sure the folder exists before writing
        os.makedirs(self.store_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.metadata, f)

    def load_store(self):
        self.index = faiss.read_index(self.index_file)
        with open(self.meta_file, "rb") as f:
            self.metadata = pickle.load(f)