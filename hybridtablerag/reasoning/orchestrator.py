"""
reasoning/orchestrator.py
==========================
QueryPlan-driven orchestrator.

Design
------
1. IntentClassifier produces ONE QueryPlan (one LLM call).
   The plan answers: which tables? aggregation? visualization? semantic? conversational?

2. Orchestrator reads the plan and decides:
   - Which SQL to run (table selection, join awareness, vector distance if semantic)
   - Whether to run Python (visualization_reason from plan drives mode)
   - Whether to do hybrid SQL+vector (both in one DuckDB query when possible)
   - Whether to answer conversationally (no DB needed)

3. Hybrid SQL+vector:
   When plan.needs_semantic=True AND vector embeddings exist on the table:
   - The SQL generator injects an array_distance() expression
   - We pass the embedded query vector as a parameter to DuckDB
   - Result is SQL-filtered AND vector-ranked in ONE query
   - Falls back to pure SQL if vector store not available

4. Table selection:
   plan.relevant_tables tells us which tables to include in the schema context.
   If the LLM identified no relevant tables, all tables are used.

Fixes vs previous version
--------------------------
- schema_ctx passed as dict to generate_sql (was list → mismatch fixed in sql.py)
- all_values format mismatch fixed in sql.py
- Table selection from QueryPlan (was always default_table)
- Hybrid vector+SQL path (was separate branch)
- Python mode and visualization_reason from QueryPlan (was separate LLM call)
- schema_ctx cache invalidated on table_names change
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from hybridtablerag.reasoning.intent import IntentClassifier, QueryPlan
from hybridtablerag.reasoning.python_exec import PythonExecutor
from hybridtablerag.reasoning.sql import SQLValidator, clean_sql
from hybridtablerag.storage.schema import (
    build_multi_table_schema_context,
    format_schema_for_prompt,
)
from hybridtablerag.reasoning.column_resolver import ColumnResolver


@dataclass
class QueryResult:
    intent:           str
    user_query:       str
    session_id:       str

    # SQL path
    sql:              Optional[str]            = None
    dataframe:        Optional[pd.DataFrame]   = None
    reasoning:        Optional[str]            = None

    # Python path
    python_code:      Optional[str]            = None
    python_dataframe: Optional[pd.DataFrame]   = None
    chart:            Optional[Any]            = None

    # Vector path (when hybrid not possible)
    vector_results:   Optional[pd.DataFrame]   = None
    vector_query:     Optional[str]            = None

    # Conversational path
    llm_answer:       Optional[str]            = None

    # Meta
    context_used:     Optional[str]            = None
    error:            Optional[str]            = None
    python_error:     Optional[str]            = None
    vector_error:     Optional[str]            = None
    debug_info:       Dict[str, Any]           = field(default_factory=dict)
    bts_log:          List[str]                = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.error is None


COLUMN_RESOLVE_PROMPT = """
You are an expert at mapping user questions to database columns.

You are given:
1. A user query
2. A full database schema (tables, columns, sample values)
3. Relationships between tables

Your job:
Identify which columns best match the *meaning* of words in the query.

Focus on:
- semantic meaning (not just name similarity)
- column values (VERY important)
- relationships if meaning spans multiple tables

USER QUERY:
{query}

SCHEMA:
{schema}

RELATIONSHIPS:
{relationships}

INSTRUCTIONS:
- Only return mappings when confident
- Use EXACT column names
- Prefer columns whose VALUES match the query meaning
- If multiple columns match, choose the BEST one
- If nothing matches, return empty mappings

OUTPUT FORMAT (strict JSON only):
{{
  "mappings": {{
    "user_term": "table.column"
  }}
}}
"""

class QueryOrchestrator:

    SQL_MAX_RETRIES = 3

    def __init__(
        self,
        llm,
        store,
        context_store,
        sql_generator,
        table_names:    List[str],
        relationships:  List[dict],
        vector_store=None,
        default_table:  str = None,
    ):
        self.llm           = llm
        self.store         = store
        self.conn          = store.conn
        self.context_store = context_store
        self.sql_generator = sql_generator
        self.vector_store  = vector_store
        self.table_names   = table_names
        self.relationships = relationships
        self.default_table = default_table or (table_names[0] if table_names else None)

        self.column_resolver = ColumnResolver(
            llm=self.llm,
            vector_store=self.vector_store
        )

        self.intent_classifier = IntentClassifier(llm)
        self.python_exec       = PythonExecutor(llm)

        # Schema cache
        self._schema_ctx:         Optional[Dict]  = None
        self._schema_cached_at:   Optional[float] = None
        self._schema_table_sig:   Optional[str]   = None   # detect table changes


    # Cache management 

    def invalidate_schema_cache(self):
        self._schema_ctx = None
        self._schema_cached_at = None
        self._schema_table_sig = None

    def _build_schema_ctx(self, bts_log: List[str], max_age: int = 300) -> Dict:
        now     = time.time()
        table_sig = ",".join(sorted(self.table_names))

        stale = (
            self._schema_ctx is None
            or self._schema_table_sig != table_sig
            or (self._schema_cached_at and now - self._schema_cached_at > max_age)
        )

        if stale:
            bts_log.append(" Building schema context…")
            self._schema_ctx = build_multi_table_schema_context(
                self.conn, self.table_names, self.relationships, bts_log=bts_log,
            )
            self._schema_cached_at = now
            self._schema_table_sig = table_sig
        return self._schema_ctx

    # Context injection

    def _inject_context(self, user_query: str, session_id: str, bts_log: List[str]) -> str:
        if not self.context_store:
            return user_query
        try:
            ctx = self.context_store.build_context_summary(session_id)
            if ctx:
                bts_log.append("Injected conversation context")
                return f"Previous conversation:\n{ctx}\n\nCurrent question:\n{user_query}"
        except Exception as e:
            bts_log.append(f"Context error: {e}")
        return user_query

    #  Table selection

    def _resolve_columns(self, user_query, schema_ctx, plan, bts_log):
        try:
            return self.column_resolver.resolve(
                user_query=user_query,
                schema_ctx=schema_ctx,
                plan=plan,
                bts_log=bts_log,
            )
        except Exception as e:
            bts_log.append(f"Column resolution failed: {e}")
            return {}
        
    def _select_schema_for_plan(self, plan: QueryPlan, full_schema: Dict) -> Dict:
        """
        Filter the full schema context to only the tables the plan needs.
        If plan didn't identify relevant tables, return the full schema.
        """
        if not plan.relevant_tables:
            return full_schema

        all_tables = full_schema.get("tables", [full_schema])
        selected = [t for t in all_tables if t.get("table_name") in plan.relevant_tables]

        if not selected:
            return full_schema   # fallback: use all

        return {
            "tables": selected,
            "relationships": [
                r for r in full_schema.get("relationships", [])
                if r.get("from_table") in plan.relevant_tables
                and r.get("to_table") in plan.relevant_tables
            ],
        }
    
    def _resolve_vector_table(self, plan: QueryPlan, schema_ctx: Dict) -> Optional[str]:
        """
        Dynamically select table that supports embeddings.
        """
        tables = schema_ctx.get("tables", [schema_ctx])

        for t in tables:
            cols = [c["name"] for c in t.get("columns", [])]
            if "_embedding" in cols:
                if not plan.relevant_tables or t["table_name"] in plan.relevant_tables:
                    return t["table_name"]

        return None

    def _resolve_columns(
        self,
        user_query: str,
        schema_ctx,
        relationships,
        bts_log
    ):
        import json
        import re

        try:
            # Build schema text (reuse your existing formatter)
            from hybridtablerag.reasoning.sql import format_schema_for_prompt

            schema_text = format_schema_for_prompt(schema_ctx)

            # Format relationships
            rel_lines = []
            for r in relationships or []:
                rel_lines.append(
                    f"{r['from_table']}.{r['from_column']} -> "
                    f"{r['to_table']}.{r['to_column']}"
                )
            rel_text = "\n".join(rel_lines) if rel_lines else "None"

            prompt = COLUMN_RESOLVE_PROMPT.format(
                query=user_query,
                schema=schema_text,
                relationships=rel_text,
            )

            raw = self.llm.generate(prompt).strip()

            # Clean output
            raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
            raw = re.sub(r"```$", "", raw).strip()

            parsed = json.loads(raw)
            mapping = parsed.get("mappings", {})

            # Validate mapping (VERY IMPORTANT)
            valid_mapping = {}
            all_columns = set()

            for table in schema_ctx.get("tables", [schema_ctx]):
                tname = table.get("table_name")
                for col in table.get("columns", []):
                    all_columns.add(f"{tname}.{col['name']}")

            for k, v in mapping.items():
                if v in all_columns:
                    valid_mapping[k] = v

            bts_log.append(f"Column mapping (LLM full-context): {valid_mapping}")

            return valid_mapping

        except Exception as e:
            bts_log.append(f"Column resolver failed: {e}")
            return {}


    # SQL execution (with retry + empty-result check) 

    def _run_sql(
        self,
        query:       str,
        schema_ctx:  Dict,
        plan:        QueryPlan,
        reasoning:   bool,
        bts_log:     List[str],
        query_vector: Optional[List[float]] = None,
        column_map=None
    ) -> tuple[str, pd.DataFrame, Optional[str]]:

        last_error = ""

        for attempt in range(1, self.SQL_MAX_RETRIES + 1):
            bts_log.append(f"SQL attempt {attempt}/{self.SQL_MAX_RETRIES}")

            try:
                # Determine if we should inject vector distance into SQL
                vector_table = None
                embed_dim = 384

                if plan.needs_semantic and self.vector_store and query_vector is not None:
                    vector_table = self._resolve_vector_table(plan, schema_ctx)
                    if vector_table:
                        embed_dim = self.vector_store.provider.dimension if self.vector_store.provider else 384

                gen = self.sql_generator.generate_sql(
                    user_query=query,
                    schema_metadata=schema_ctx,
                    relationships=self.relationships,
                    reasoning=reasoning,
                    plan=plan,
                    vector_table=vector_table,
                    embed_dim=embed_dim,
                )

                if isinstance(gen, dict):
                    sql          = gen["sql_query"]
                    reasoning_tx = gen.get("reasoning")
                else:
                    sql          = gen
                    reasoning_tx = None

                sql = clean_sql(sql)
                SQLValidator.validate(sql)
                bts_log.append(f"Generated SQL:\n{sql}")

                # Execute — pass query vector as parameter if SQL uses it
                if plan.needs_semantic and query_vector is not None and "?" in sql:
                    df = self.conn.execute(sql, [query_vector]).fetchdf()
                else:
                    df = self.conn.execute(sql).fetchdf()

                bts_log.append(f"SQL returned {len(df)} rows × {len(df.columns)} cols")

                # Empty result check — only retry if we have attempts left
                if len(df) == 0 and attempt < self.SQL_MAX_RETRIES:
                    bts_log.append("0 rows — checking for value mismatch…")
                    dist_block = _build_dist_hint(schema_ctx)
                    fix_prompt = _EMPTY_RESULT_PROMPT.format(
                        query=query,
                        sql=sql,
                        distributions=dist_block
                    ) + f"\nPlan hints: aggregation={plan.needs_aggregation}, join={plan.needs_join}"
                    raw_fix = self.llm.generate(fix_prompt).strip()
                    if raw_fix.upper().startswith("NO_RESULTS"):
                        bts_log.append(f"LLM: no data exists — {raw_fix[10:].strip()[:80]}")
                        return sql, df, reasoning_tx

                    sql = clean_sql(raw_fix)
                    SQLValidator.validate(sql)
                    bts_log.append(f"Revised SQL:\n{sql}")
                    df = self.conn.execute(sql).fetchdf()
                    bts_log.append(f" Revised SQL returned {len(df)} rows")

                return sql, df, reasoning_tx

            except Exception as e:
                last_error = str(e)
                bts_log.append(f"SQL error (attempt {attempt}): {last_error[:120]}")
                if attempt < self.SQL_MAX_RETRIES:
                    time.sleep(0.5 * (2 ** (attempt - 1)))

        raise RuntimeError(f"SQL failed after {self.SQL_MAX_RETRIES} attempts: {last_error}")

    #  Python execution 

    def _run_python(
        self,
        user_query: str,
        df:         pd.DataFrame,
        plan:       QueryPlan,
        bts_log:    List[str],
    ):
        return self.python_exec.execute(
            user_query=user_query,
            df=df,
            table_name=self.default_table or "",
            bts_log=bts_log,
            mode=plan.python_mode,
            visualization_reason=plan.visualization_reason,
        )

    #  Conversational

    def _run_conversational(self, user_query: str, session_id: str, bts_log: List[str]) -> str:
        try:
            history = self.context_store.get_history(session_id, last_n=10)
            history_text = "\n".join(
                f"Q: {h['user_query']} → {h.get('result_summary', '')}"
                for h in history
            )
            prompt = (
                f"You are a helpful assistant.\n\nConversation so far:\n{history_text}\n\n"
                f"User question:\n{user_query}\n\nAnswer ONLY using the conversation above."
            )
            return self.llm.generate(prompt)
        except Exception as e:
            bts_log.append(f"Conversational error: {e}")
            return "I couldn't retrieve conversation history."

    # Main entry point

    def run(
        self,
        user_query:     str,
        session_id:     str,
        reasoning:      bool = False,
        debug_mode:     bool = False,
        force_intent:   Optional[str] = None,
    ) -> QueryResult:

        bts_log: List[str] = []
        result = QueryResult(
            intent=force_intent or "sql",
            user_query=user_query,
            session_id=session_id,
            bts_log=bts_log,
        )

        try:
            # 1. Build schema context 
            full_schema = self._build_schema_ctx(bts_log)

            #2. Build query plan (one LLM call) 
            if force_intent:
                # Minimal plan from forced intent
                plan = _intent_to_plan(force_intent, self.table_names)
                bts_log.append(f"Force intent: {force_intent}")
            else:
                schema_summary = format_schema_for_prompt(full_schema)
                plan = self.intent_classifier.classify(
                    user_query,
                    schema_summary=schema_summary,
                    available_tables=self.table_names,
                )
                bts_log.append(
                    f"Plan: sql={plan.needs_sql} python={plan.needs_python} "
                    f"semantic={plan.needs_semantic} conv={plan.is_conversational} "
                    f"mode={plan.python_mode}"
                )
                if plan.reasoning:
                    bts_log.append(f"   Reasoning: {plan.reasoning}")

            result.intent = _plan_to_intent_label(plan)


            # 3. Inject conversation context
            augmented_query = self._inject_context(user_query, session_id, bts_log)
            result.context_used = augmented_query if augmented_query != user_query else None

            # 4. Select relevant schema
            schema_ctx = self._select_schema_for_plan(plan, full_schema)
            selected_tables = [t.get("table_name") for t in schema_ctx.get("tables", [schema_ctx])]
            bts_log.append(f"Using tables: {selected_tables}")

            # # 4.5 Resolve semantic column mappings (NEW LAYER)
            # column_map = self._resolve_columns(user_query, schema_ctx, plan, bts_log)

            # 5. Conversational path
            if plan.is_conversational:
                result.llm_answer = self._run_conversational(user_query, session_id, bts_log)
                _save_context(self.context_store, session_id, user_query, "conversational",
                              "text answer", bts_log=bts_log)
                return result

            # 6. Embed query if semantic search needed
            query_vector: Optional[List[float]] = None
            if plan.needs_semantic and self.vector_store:
                try:
                    search_q = plan.semantic_query or user_query
                    query_vector = self.vector_store.provider.embed([search_q])[0]
                    bts_log.append(f"Query embedded for semantic search ({len(query_vector)} dims)")
                except Exception as e:
                    bts_log.append(f"Embedding failed, falling back to SQL-only: {e}")

            # 7. SQL path
            if plan.needs_sql or not plan.needs_python:
                try:
                    result.sql, result.dataframe, result.reasoning = self._run_sql(
                        augmented_query,
                        schema_ctx,
                        plan,
                        reasoning,
                        bts_log,
                        query_vector=query_vector
                    )
                except Exception as e:
                    result.error = str(e)
                    bts_log.append(f"SQL path failed: {e}")

            # 8. Pure vector fallback (if SQL failed or no SQL needed) 
            if (result.dataframe is None or result.dataframe.empty) and plan.needs_semantic:
                if self.vector_store and query_vector is not None:
                    try:
                        bts_log.append("Falling back to pure vector search…")
                        result.vector_results = self.vector_store.search(
                            user_query, self.default_table, top_k=10
                        )
                        result.vector_query = user_query
                        bts_log.append(f"Vector search: {len(result.vector_results)} results")
                    except Exception as e:
                        result.vector_error = str(e)
                        bts_log.append(f"Vector error: {e}")

            # 9. Python / visualization path
            if plan.needs_python:
                # Use SQL result if available and non-empty, else load focused data
                py_df = None
                if result.dataframe is not None and not result.dataframe.empty:
                    py_df = result.dataframe.copy()
                    bts_log.append("Python using SQL result as input")
                else:
                    # Try a pre-filter SQL to get relevant data for Python
                    if self.sql_generator and self.default_table:
                        try:
                            pre_sql = self.sql_generator.generate_sql(
                                f"Return only the columns and rows relevant to: {user_query}",
                                schema_ctx, self.relationships, reasoning=False, plan=plan,
                            )
                            py_df = self.conn.execute(clean_sql(
                                pre_sql["sql_query"] if isinstance(pre_sql, dict) else pre_sql
                            )).fetchdf()
                            bts_log.append(f"Python pre-filter: {len(py_df)} rows")
                        except Exception as e:
                            bts_log.append(f"Pre-filter failed: {e}")

                    if py_df is None or py_df.empty:
                        # Last resort: full table (capped)
                        try:
                            table = plan.relevant_tables[0] if plan.relevant_tables else self.default_table
                            py_df = self.conn.execute(
                                f'SELECT * FROM "{table}" LIMIT 5000'
                            ).fetchdf()
                            bts_log.append(f"Loaded full table for Python: {len(py_df)} rows (capped at 5000)")
                        except Exception as e:
                            bts_log.append(f"Could not load data for Python: {e}")

                if py_df is not None and not py_df.empty:
                    try:
                        result.python_dataframe, result.chart, result.python_code = (
                            self._run_python(user_query, py_df, plan, bts_log)
                        )
                    except Exception as e:
                        result.python_error = str(e)
                        bts_log.append(f"Python path failed: {e}")
                        bts_log.append(traceback.format_exc())

        except Exception as exc:
            result.error = str(exc)
            bts_log.append(f"Orchestrator error: {exc}")
            bts_log.append(traceback.format_exc())

        # 10. Save context
        summary = (
            f"{len(result.dataframe)} rows" if result.dataframe is not None and not result.dataframe.empty
            else "text answer" if result.llm_answer
            else f"{len(result.vector_results)} vector results" if result.vector_results is not None
            else "no result"
        )
        _save_context(
            self.context_store, session_id, user_query,
            result.intent, summary,
            sql=result.sql, error=result.error or result.python_error, bts_log=bts_log,
        )

        # 11. Debug info
        if debug_mode:
            result.debug_info = {
                "schema_tables": [t.get("table_name") for t in schema_ctx.get("tables", [schema_ctx])
                                  if isinstance(t, dict)],
                "plan": {
                    "needs_sql": plan.needs_sql if 'plan' in dir() else None,
                    "needs_python": plan.needs_python if 'plan' in dir() else None,
                    "needs_semantic": plan.needs_semantic if 'plan' in dir() else None,
                    "python_mode": plan.python_mode if 'plan' in dir() else None,
                    "relevant_tables": plan.relevant_tables if 'plan' in dir() else [],
                    "reasoning": plan.reasoning if 'plan' in dir() else "",
                },
                "relationships": self.relationships,
            }

        return result


# Helpers

_EMPTY_RESULT_PROMPT = """\
The SQL returned 0 rows.

User question: {query}
SQL: {sql}

Column distributions (use EXACT values in filters):
{distributions}

Diagnose and fix. Common causes:
- Wrong filter value (check distributions above)
- Case mismatch
- Over-restrictive AND conditions

If genuinely no data exists, respond: NO_RESULTS: <one sentence why>
Otherwise return ONLY the corrected SQL.
"""


def _build_dist_hint(schema_ctx: Dict) -> str:
    lines = []
    tables = schema_ctx.get("tables", [schema_ctx])
    for table in tables:
        for col in table.get("columns", []):
            if col.get("all_values"):
                vals = list(col["all_values"].keys()) if isinstance(col["all_values"], dict) \
                       else [v.get("value", "") for v in col["all_values"]]
                lines.append(f"  {col['name']}: {vals}")
            elif col.get("range"):
                lines.append(f"  {col['name']}: {col['range']['min']} → {col['range']['max']}")
    return "\n".join(lines) or "  (no distribution data)"


def _intent_to_plan(force_intent: str, table_names: List[str]) -> "QueryPlan":
    """Create a minimal QueryPlan from a force_intent string."""
    from hybridtablerag.reasoning.intent import QueryPlan
    p = QueryPlan(relevant_tables=table_names)
    if force_intent == "python":
        p.needs_sql    = True
        p.needs_python = True
        p.python_mode  = "both"
    elif force_intent == "vector":
        p.needs_semantic = True
    elif force_intent == "conversational":
        p.is_conversational = True
        p.needs_sql         = False
    # sql: defaults are fine
    return p


def _plan_to_intent_label(plan: "QueryPlan") -> str:
    if plan.is_conversational: return "conversational"
    if plan.needs_semantic and not plan.needs_sql: return "vector"
    if plan.needs_python and plan.needs_sql: return "sql+python"
    if plan.needs_python: return "python"
    return "sql"


def _save_context(context_store, session_id, user_query, intent, summary,
                  sql=None, error=None, bts_log=None):
    if not context_store:
        return
    try:
        context_store.save_turn(
            session_id=session_id,
            user_query=user_query,
            intent=intent,
            result_summary=summary,
            sql_generated=sql,
            error=error,
        )
    except Exception as e:
        if bts_log is not None:
            bts_log.append(f"Context save error: {e}")