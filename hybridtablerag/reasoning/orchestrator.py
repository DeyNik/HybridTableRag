"""
reasoning/orchestrator.py
==========================

Upgraded orchestrator:
- Multi-table schema support
- Context injection (conversation memory)
- Vector + conversational paths
- Backward compatible with current API layer
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from hybridtablerag.reasoning.intent import IntentClassifier
from hybridtablerag.reasoning.python_exec import PythonExecutor
from hybridtablerag.storage.schema import build_multi_table_schema_context


# ──────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    intent: str
    user_query: str
    session_id: str

    sql: Optional[str] = None
    dataframe: Optional[pd.DataFrame] = None
    reasoning: Optional[str] = None

    python_code: Optional[str] = None
    python_dataframe: Optional[pd.DataFrame] = None
    chart: Optional[Any] = None

    vector_results: Optional[pd.DataFrame] = None
    vector_query: Optional[str] = None

    llm_answer: Optional[str] = None
    context_used: Optional[str] = None

    error: Optional[str] = None
    python_error: Optional[str] = None
    vector_error: Optional[str] = None

    debug_info: Dict[str, Any] = field(default_factory=dict)
    bts_log: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.error is None


# ──────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────

class QueryOrchestrator:

    SQL_MAX_RETRIES = 3

    def __init__(
        self,
        llm,
        store,
        context_store,
        sql_generator,
        table_names: List[str],
        relationships: List[dict],
        vector_store=None,
        default_table: str = None,
    ):
        self.llm = llm
        self.store = store
        self.conn = store.conn
        self.context_store = context_store
        self.sql_generator = sql_generator
        self.vector_store = vector_store

        self.table_names = table_names
        self.relationships = relationships
        self.default_table = default_table or table_names[0]

        self.intent_classifier = IntentClassifier(llm)
        self.python_exec = PythonExecutor(llm)

        self._schema_ctx = None  # cached

    # ──────────────────────────────────────────────────────────
    def _build_schema_ctx(self, bts_log):
        if self._schema_ctx is None:
            bts_log.append("📦 Building schema context")
            self._schema_ctx = build_multi_table_schema_context(
                self.conn,
                self.table_names,
                self.relationships,
                bts_log=bts_log,
            )
        return self._schema_ctx

    # ──────────────────────────────────────────────────────────
    def _inject_context(self, user_query, session_id, bts_log):
        if not self.context_store:
            return user_query

        try:
            context = self.context_store.build_context_summary(session_id)
            if context:
                bts_log.append("🧠 Injected conversation context")
                return f"""
Previous conversation:
{context}

Current question:
{user_query}
"""
            return user_query
        except Exception as e:
            bts_log.append(f"Context error: {e}")
            return user_query

    # ──────────────────────────────────────────────────────────
    def _run_sql(self, query, schema_ctx, reasoning, bts_log):
        from hybridtablerag.reasoning.sql import SQLValidator, clean_sql

        sql = ""
        last_error = ""

        for attempt in range(1, self.SQL_MAX_RETRIES + 1):
            bts_log.append(f"⚙️ SQL attempt {attempt}")

            try:
                gen = self.sql_generator.generate_sql(
                    query,
                    schema_ctx,
                    self.relationships,
                    reasoning,
                )

                if isinstance(gen, dict):
                    sql = gen["sql_query"]
                else:
                    sql = gen

                sql = clean_sql(sql)
                SQLValidator.validate(sql)

                df = self.conn.execute(sql).fetchdf()
                return sql, df, None

            except Exception as e:
                last_error = str(e)
                bts_log.append(f"SQL error: {e}")

        raise RuntimeError(f"SQL failed: {last_error}")

    # ──────────────────────────────────────────────────────────
    def _run_python(self, user_query, df, table, bts_log):
        python_mode = self.intent_classifier.classify_python_mode(user_query)

        return self.python_exec.execute(
            user_query,
            df,
            table,
            bts_log,
            python_mode,
        )

    # ──────────────────────────────────────────────────────────
    def _run_vector(self, user_query, bts_log):
        if not self.vector_store:
            return None

        try:
            df = self.vector_store.search(
                user_query,
                self.default_table,
            )
            return df
        except Exception as e:
            bts_log.append(f"Vector error: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    def _run_conversational(self, user_query, session_id, bts_log):
        try:
            history = self.context_store.get_history(session_id, last_n=10)

            history_text = "\n".join(
                f"Q: {h['user_query']} → {h.get('result_summary', '')}"
                for h in history
            )

            prompt = f"""
You are a helpful assistant.

Conversation so far:
{history_text}

User question:
{user_query}

Answer ONLY using conversation context.
"""

            return self.llm.generate(prompt)

        except Exception as e:
            bts_log.append(f"Conversational error: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    def run(
        self,
        user_query: str,
        session_id: str,
        reasoning: bool = False,
        debug_mode: bool = False,
        force_intent: Optional[str] = None,
    ) -> QueryResult:

        bts_log = []

        intent = force_intent or self.intent_classifier.classify(user_query)

        result = QueryResult(
            intent=intent,
            user_query=user_query,
            session_id=session_id,
            bts_log=bts_log,
        )

        schema_ctx = self._build_schema_ctx(bts_log)
        augmented_query = self._inject_context(user_query, session_id, bts_log)

        # ───────── VECTOR ─────────
        if intent == "vector":
            result.vector_results = self._run_vector(user_query, bts_log)
            result.vector_query = user_query

        # ───────── CONVERSATIONAL ─────────
        elif intent == "conversational":
            result.llm_answer = self._run_conversational(
                user_query, session_id, bts_log
            )

        # ───────── SQL + PYTHON ─────────
        else:
            try:
                result.sql, result.dataframe, _ = self._run_sql(
                    augmented_query,
                    schema_ctx,
                    reasoning,
                    bts_log,
                )
            except Exception as e:
                result.error = str(e)

            try:
                df = result.dataframe
                if df is None or df.empty:
                    df = self.conn.execute(
                        f"SELECT * FROM {self.default_table}"
                    ).fetchdf()

                (
                    result.python_dataframe,
                    result.chart,
                    result.python_code,
                ) = self._run_python(user_query, df, self.default_table, bts_log)

            except Exception as e:
                result.python_error = str(e)

        # ───────── SAVE CONTEXT ─────────
        try:
            if self.context_store:
                summary = ""
                if result.dataframe is not None:
                    summary = f"{len(result.dataframe)} rows"
                elif result.llm_answer:
                    summary = "text answer"

                self.context_store.save_turn(
                    session_id=session_id,
                    user_query=user_query,
                    intent=intent,
                    result_summary=summary,
                    sql_generated=result.sql,
                    error=result.error,
                )
        except Exception as e:
            bts_log.append(f"Context save error: {e}")

        # ───────── DEBUG ─────────
        if debug_mode:
            result.debug_info = {
                "schema_ctx": schema_ctx,
                "relationships": self.relationships,
            }

        return result