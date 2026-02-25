# hybrid-table-rag/retrieval/semantic_retriever.py
from typing import List, Dict
from hybridtablerag.embeddings.embedding_store import EmbeddingStore

class SemanticRetriever:
    """
    Retrieves top-k relevant rows using embeddings.
    """

    def __init__(self, store_path="embeddings/faiss_store", embedding_model_name="all-MiniLM-L6-v2"):
        self.store = EmbeddingStore(embedding_model_name=embedding_model_name, store_path=store_path)

    def query(self, query_str: str, top_k: int = 5) -> List[Dict]:
        return self.store.search(query_str, top_k=top_k)