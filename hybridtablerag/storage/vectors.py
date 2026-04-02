"""
storage/vectors.py
==================
DuckDB vector search using vss extension.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Embedding interface
# ─────────────────────────────────────────────────────────────

class EmbeddingProvider(ABC):

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


# ─────────────────────────────────────────────────────────────
# SentenceTransformer (LOCAL)
# ─────────────────────────────────────────────────────────────

class SentenceTransformerProvider(EmbeddingProvider):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dim


# ─────────────────────────────────────────────────────────────
# Azure OpenAI embeddings
# ─────────────────────────────────────────────────────────────

class AzureOpenAIEmbeddingProvider(EmbeddingProvider):

    def __init__(self, deployment: str, api_key: str, endpoint: str, api_version: str = "2024-02-01"):
        from openai import AzureOpenAI

        if not all([deployment, api_key, endpoint]):
            raise ValueError("Azure embedding config missing")

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        self.deployment = deployment
        self._dim = 3072 if "large" in deployment else 1536

    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        batch_size = 2048

        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]

            response = self.client.embeddings.create(
                input=chunk,
                model=self.deployment
            )

            results.extend([item.embedding for item in response.data])

        return results

    @property
    def dimension(self) -> int:
        return self._dim


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def get_embedding_provider() -> EmbeddingProvider:
    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformer")

    if provider == "azure_openai":
        return AzureOpenAIEmbeddingProvider(
            deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
        )

    model = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformerProvider(model)


# ─────────────────────────────────────────────────────────────
# VectorStore
# ─────────────────────────────────────────────────────────────

class VectorStore:

    EMBEDDING_COL = "_embedding"

    def __init__(self, conn, provider: EmbeddingProvider):
        self.conn = conn
        self.provider = provider

    def setup(self):
        try:
            self.conn.execute("INSTALL vss")
            self.conn.execute("LOAD vss")
            print("[VectorStore] vss extension loaded")
        except Exception as e:
            print(f"[VectorStore] Failed to load vss: {e}")

    def embed_table(
        self,
        table_name: str,
        text_columns: List[str],
        pk_column: str,
        bts_log: list,
    ):
        if not text_columns:
            bts_log.append("No text columns provided for embedding")
            return

        df = self.conn.execute(
            f"SELECT {pk_column}, {', '.join(text_columns)} FROM {table_name}"
        ).fetchdf()

        if df.empty:
            bts_log.append(f"No rows to embed in {table_name}")
            return

        texts = []
        pks = []

        for _, row in df.iterrows():
            parts = [
                f"{col}: {row[col]}"
                for col in text_columns
                if pd.notna(row[col])
            ]

            if not parts:
                continue  # skip empty rows

            texts.append(" | ".join(parts))
            pks.append(row[pk_column])

        if not texts:
            bts_log.append("No valid text rows for embedding")
            return

        vectors = self.provider.embed(texts)
        dim = self.provider.dimension

        self.conn.execute(f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS {self.EMBEDDING_COL} FLOAT[{dim}]
        """)

        update_data = list(zip(vectors, pks))

        self.conn.executemany(
            f"""
            UPDATE {table_name}
            SET {self.EMBEDDING_COL} = ?
            WHERE {pk_column} = ?
            """,
            update_data
        )

        try:
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding
                ON {table_name}
                USING HNSW ({self.EMBEDDING_COL})
                WITH (metric = 'cosine')
            """)
        except Exception:
            pass

        bts_log.append(
            f"Embedded {len(update_data)} rows from {table_name}"
        )

    def search(
        self,
        query: str,
        table_name: str,
        top_k: int = 10,
        sql_filter: Optional[str] = None,
    ) -> pd.DataFrame:

        query_vec = self.provider.embed([query])[0]
        dim = self.provider.dimension

        where_clause = f"WHERE {sql_filter}" if sql_filter else ""

        df = self.conn.execute(f"""
            SELECT *,
                   array_distance({self.EMBEDDING_COL}, ?::FLOAT[{dim}]) AS similarity_score
            FROM {table_name}
            {where_clause}
            ORDER BY similarity_score ASC
            LIMIT {top_k}
        """, [query_vec]).fetchdf()

        if self.EMBEDDING_COL in df.columns:
            df = df.drop(columns=[self.EMBEDDING_COL])

        return df

    def get_embedded_tables(self) -> List[str]:
        result = self.conn.execute("""
            SELECT DISTINCT table_name
            FROM information_schema.columns
            WHERE column_name = '_embedding'
        """).fetchall()

        return [row[0] for row in result]