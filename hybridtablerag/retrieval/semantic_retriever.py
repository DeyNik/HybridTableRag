# hybrid-table-rag/retrieval/semantic_retriever.py

from typing import Optional
from hybridtablerag.embeddings.embedding_store import EmbeddingStore
from hybridtablerag.llm.factory import get_llm


class SemanticRetriever:
    """
    Retrieves top-k relevant rows using embeddings and generates answers via LLM.
    """

    def __init__(
        self,
        llm_name: str = "ollama",
        llm_model: str = "llama2",
        embedding_store: Optional[EmbeddingStore] = None,
        store_path: str = "embeddings/faiss_store",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        # Use injected store OR create new one
        self.store = embedding_store or EmbeddingStore(
            embedding_model_name=embedding_model_name,
            store_path=store_path
        )

        self.llm = get_llm(llm_name, llm_model)

    def answer_query(self, query: str, top_k: int = 5) -> str:
        top_rows = self.store.search(query, top_k=top_k)

        if not top_rows:
            return "No relevant information found in embeddings."

        context_text = "\n".join([
            f"{r.get('table')} | {r.get('row_idx')} | {r.get('text', '')}"
            for r in top_rows
        ])

        prompt = f"""
        You are a data analyst.

        Answer ONLY using the context below.

        Context:
        {context_text}

        Question:
        {query}

        Do not invent data. Be concise.
        """

        response = self.llm.generate(prompt)

        # Handle response structure safely
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "").strip()

        return str(response).strip()