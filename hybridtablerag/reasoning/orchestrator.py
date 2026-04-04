"""
reasoning/orchestrator.py
==========================
✅ FIXED: Schema cache invalidation, validated fallback query, retry backoff, vector params
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from hybridtablerag.reasoning.intent import IntentClassifier
from hybridtablerag.reasoning.python_exec import PythonExecutor
from hybridtablerag.storage.schema import build_multi_table_schema_context
from hybridtablerag.storage.store import _escape_identifier  # Reuse escaping helper


# Result container

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


# Orchestrator

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
        if default_table:
            if default_table not in table_names:
                raise ValueError(f"default_table '{default_table}' not in table_names: {table_names}")
            self.default_table = default_table
        else:
            self.default_table = table_names[0] if table_names else None

        self.intent_classifier = IntentClassifier(llm)
        self.python_exec = PythonExecutor(llm)

        self._schema_ctx = None  # cached
        self._schema_cached_at = None  

    def invalidate_schema_cache(self):
        """
        Call this after ingesting new data to force schema rebuild.
        """
        self._schema_ctx = None
        self._schema_cached_at = None

    def _build_schema_ctx(self, bts_log, max_age_seconds: int = 300):
        """
        Build or return cached schema context.
        """
        now = time.time()
        # Rebuild if cache is empty or stale
        if self._schema_ctx is None or (self._schema_cached_at and now - self._schema_cached_at > max_age_seconds):
            bts_log.append("Building schema context")
            self._schema_ctx = build_multi_table_schema_context(
                self.conn,
                self.table_names,
                self.relationships,
                bts_log=bts_log,
            )
            self._schema_cached_at = now
        return self._schema_ctx


    def _inject_context(self, user_query, session_id, bts_log):
        if not self.context_store:
            return user_query

        try:
            context = self.context_store.build_context_summary(session_id)
            if context:
                bts_log.append("Injected conversation context")
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
                bts_log.append(f"SQL error (attempt {attempt}): {e}")
                
                if attempt < self.SQL_MAX_RETRIES:
                    backoff = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s, 2s...
                    bts_log.append(f"Retrying in {backoff}s...")
                    time.sleep(backoff)

        raise RuntimeError(f"SQL failed after {self.SQL_MAX_RETRIES} attempts: {last_error}")

    def _run_python(self, user_query, df, table, bts_log):
        python_mode = self.intent_classifier.classify_python_mode(user_query)

        return self.python_exec.execute(
            user_query,
            df,
            table,
            bts_log,
            python_mode,
        )

    def _run_vector(self, user_query, bts_log, top_k: int = 10, sql_filter: Optional[str] = None):
   
        if not self.vector_store:
            bts_log.append("Vector store not available")
            return None

        try:
            return self.vector_store.search(
                user_query,
                self.default_table,
                top_k=top_k,
                sql_filter=sql_filter,
            )
        except Exception as e:
            bts_log.append(f"Vector error: {e}")
            return None

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

    def run(
        self,
        user_query: str,
        session_id: str,
        reasoning: bool = False,
        debug_mode: bool = False,
        force_intent: Optional[str] = None,
        vector_top_k: int = 10,
        vector_filter: Optional[str] = None,
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

        # VECTOR
        if intent == "vector":
            result.vector_results = self._run_vector(
                user_query, bts_log, 
                top_k=vector_top_k, 
                sql_filter=vector_filter
            )
            result.vector_query = user_query

        # CONVERSATIONAL
        elif intent == "conversational":
            result.llm_answer = self._run_conversational(
                user_query, session_id, bts_log
            )

        # SQL + PYTHON
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
                bts_log.append(f"SQL path failed: {e}")

            # Only run Python path if SQL succeeded or we have a fallback table
            if result.dataframe is None or result.dataframe.empty:
                if self.default_table and self.default_table in self.table_names:
                    try:
                        df = self.conn.execute(
                            f"SELECT * FROM {_escape_identifier(self.default_table)} LIMIT 100"
                        ).fetchdf()
                    except Exception as e:
                        bts_log.append(f"Fallback query failed: {e}")
                        df = None
                else:
                    df = None
            else:
                df = result.dataframe

            if df is not None and not df.empty:
                try:
                    (
                        result.python_dataframe,
                        result.chart,
                        result.python_code,
                    ) = self._run_python(user_query, df, self.default_table, bts_log)
                except Exception as e:
                    result.python_error = str(e)
                    bts_log.append(f"Python path failed: {e}")

        # ───────── SAVE CONTEXT ─────────
        try:
            if self.context_store:
                summary = ""
                if result.dataframe is not None and not result.dataframe.empty:
                    summary = f"{len(result.dataframe)} rows"
                elif result.llm_answer:
                    summary = "text answer"
                elif result.vector_results is not None:
                    summary = f"{len(result.vector_results)} vector results"

                self.context_store.save_turn(
                    session_id=session_id,
                    user_query=user_query,
                    intent=intent,
                    result_summary=summary,
                    sql_generated=result.sql,
                    error=result.error or result.python_error or result.vector_error,
                )
        except Exception as e:
            bts_log.append(f"Context save error: {e}")

        #DEBUG 
        if debug_mode:
            result.debug_info = {
                "schema_ctx": schema_ctx,
                "relationships": self.relationships,
                "table_names": self.table_names,
            }

        return result