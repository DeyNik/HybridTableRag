"""
reasoning/column_resolver.py
============================
Hybrid column resolver:
1. Heuristic matching
2. Optional vector similarity
3. LLM fallback (ONLY if needed)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


class ColumnResolver:

    def __init__(self, llm=None, vector_store=None):
        self.llm = llm
        self.vector_store = vector_store

    # -------- MAIN ENTRY --------
    def resolve(
        self,
        user_query: str,
        schema_ctx,
        plan,
        bts_log: List[str],
    ) -> Dict[str, str]:

        columns = self._extract_columns(schema_ctx)

        if not columns:
            return {}

        # Normalize helper
        def norm(x):
            return re.sub(r'[^a-z0-9]', '', x.lower())

        norm_columns = {col: norm(col) for col in columns}

        # Extract candidate tokens from query
        tokens = re.findall(r'\b[a-zA-Z_]+\b', user_query.lower())

        # Remove useless words
        stopwords = {
            "show", "give", "get", "count", "by", "of",
            "the", "a", "an", "and", "or", "to"
        }
        tokens = [t for t in tokens if t not in stopwords]

        mapping = {}

        # Match tokens → columns
        for token in tokens:
            nt = norm(token)

            best_col = None
            best_score = 0

            for col, ncol in norm_columns.items():

                # Exact or substring match
                if nt == ncol:
                    best_col = col
                    best_score = 1.0
                    break

                if nt in ncol or ncol in nt:
                    score = len(nt) / (len(ncol) + 1e-5)
                    if score > best_score:
                        best_score = score
                        best_col = col

            if best_col:
                mapping[token] = best_col

        if mapping:
            bts_log.append(f"ColumnResolver: heuristic match → {mapping}")
            return mapping

        # ---- fallback: vector ----
        if self.vector_store:
            try:
                col_embeddings = self.vector_store.provider.embed(columns)
                query_emb = self.vector_store.provider.embed([user_query])[0]

                best = None
                best_score = float("inf")

                for col, emb in zip(columns, col_embeddings):
                    score = sum((a - b) ** 2 for a, b in zip(query_emb, emb))
                    if score < best_score:
                        best_score = score
                        best = col

                if best:
                    mapping = {"primary": best}
                    bts_log.append(f"ColumnResolver: vector fallback → {mapping}")
                    return mapping

            except Exception as e:
                bts_log.append(f"ColumnResolver vector failed: {e}")

        # ---- fallback: LLM ----
        if self.llm:
            try:
                prompt = f"""
    Map query terms to columns.

    Query: {user_query}

    Columns: {columns}

    Return JSON like:
    {{"term": "column_name"}}
    """
                raw = self.llm.generate(prompt)

                import json
                mapping = json.loads(raw)

                if isinstance(mapping, dict):
                    bts_log.append(f"ColumnResolver: llm fallback → {mapping}")
                    return mapping

            except Exception as e:
                bts_log.append(f"ColumnResolver LLM failed: {e}")

        return {}

    # -------- HELPERS --------

    def _extract_columns(self, schema_ctx) -> List[str]:
        cols = []
        tables = schema_ctx.get("tables", [schema_ctx])

        for t in tables:
            for c in t.get("columns", []):
                cols.append(c["name"])

        return cols


    def _vector_match(self, query: str, columns: List[str]) -> Dict[str, str]:
        try:
            col_embeddings = self.vector_store.provider.embed(columns)
            query_emb = self.vector_store.provider.embed([query])[0]

            best = None
            best_score = float("inf")

            for col, emb in zip(columns, col_embeddings):
                score = sum((a - b) ** 2 for a, b in zip(query_emb, emb))
                if score < best_score:
                    best_score = score
                    best = col

            return {"best_match": best}

        except Exception:
            return {}

    def _llm_match(self, query: str, schema_ctx) -> Dict[str, str]:
        prompt = f"""
Map the user query to the most relevant column.

Query: {query}

Schema:
{schema_ctx}

Return JSON:
{{"column": "column_name"}}
"""

        raw = self.llm.generate(prompt)

        import json
        try:
            return json.loads(raw)
        except Exception:
            return {}