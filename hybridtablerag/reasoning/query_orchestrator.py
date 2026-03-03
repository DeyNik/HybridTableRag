"""
reasoning/query_orchestrator.py
================================
Intent router + SQL + Python/viz execution paths.

Flow:
    user query
        │
        ▼
    IntentRouter  ── "sql"    ──►  LLMSQLGenerator → DuckDB → DataFrame
        │
        └─────────── "python" ──►  LLMPythonGenerator → exec() → DataFrame + Chart
                                                              │
                                                        QueryResult
                                            (df, chart, sql, reasoning, bts_log)
"""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Everything Streamlit needs to render a query response."""
    intent: str                                    # "sql" | "python"
    user_query: str
    sql: Optional[str]                   = None    # executed SQL (if any)
    dataframe: Optional[pd.DataFrame]   = None
    chart: Optional[Any]                = None    # plotly Figure or None
    reasoning: Optional[str]            = None    # LLM chain-of-thought
    error: Optional[str]                = None
    bts_log: List[str]                  = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.error is None


# ──────────────────────────────────────────────────────────────────────────────
# Intent router  (replaces LangGraph for this stage)
# ──────────────────────────────────────────────────────────────────────────────

_INTENT_PROMPT = """\
You are a query router for a data analytics system.

Given a user question, decide the best execution strategy:

- "sql"    → question asks for filtered rows, counts, aggregates, rankings,
             or anything a SQL SELECT can answer directly.
- "python" → question asks for a chart, plot, graph, visualization, trend line,
             or requires pandas transformations SQL cannot express cleanly.

Reply with ONLY one word: sql  OR  python
No explanation. No punctuation. Just the word.

User question: {query}
"""


class IntentRouter:
    """
    Single-call LLM classifier: sql vs python/viz.
    Falls back to "sql" on any error — pipeline never hard-fails here.
    """

    def __init__(self, llm):
        self.llm = llm

    def route(self, user_query: str) -> str:
        try:
            prompt = _INTENT_PROMPT.format(query=user_query)
            response = self.llm.generate(prompt).strip().lower()
            if any(k in response for k in ("python", "chart", "plot", "visual")):
                return "python"
            return "sql"
        except Exception:
            return "sql"


# ──────────────────────────────────────────────────────────────────────────────
# Python / visualisation executor
# ──────────────────────────────────────────────────────────────────────────────

_PYTHON_PROMPT = """\
You are a senior data analyst. You have access to a pandas DataFrame called `df`
loaded from the DuckDB table: {table_name}

DataFrame columns and types:
{columns}

Sample rows (first 3):
{sample}

User request:
{query}

Write Python code that:
1. Uses the existing `df` variable (do NOT reload data).
2. Stores the final result as `result_df` (a pandas DataFrame).
3. If a chart is requested, creates a plotly figure stored as `fig`
   (use `import plotly.express as px`). Otherwise do NOT define `fig`.
4. Never calls plt.show() or any st.* functions.
5. Never prints anything.

Return ONLY the Python code. No markdown fences. No explanations.
"""


class PythonExecutor:
    """
    Asks LLM for pandas/plotly code, executes it in a restricted namespace
    containing only `df` and approved imports.
    """

    def __init__(self, llm):
        self.llm = llm

    def _build_prompt(self, user_query: str, df: pd.DataFrame, table_name: str) -> str:
        columns = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
        sample = df.head(3).to_string(index=False)
        return _PYTHON_PROMPT.format(
            table_name=table_name,
            columns=columns,
            sample=sample,
            query=user_query,
        )

    def execute(
        self,
        user_query: str,
        df: pd.DataFrame,
        table_name: str,
        bts_log: List[str],
    ) -> tuple[Optional[pd.DataFrame], Optional[Any], str]:
        """Returns (result_df, fig, generated_code)."""
        import plotly.express as px  # noqa: F401

        prompt = self._build_prompt(user_query, df, table_name)
        bts_log.append("🐍  Generating Python/pandas code via LLM…")

        code = self.llm.generate(prompt).strip()
        code = re.sub(r"^```(?:python)?\n?", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\n?```$", "", code).strip()
        bts_log.append(f"Generated code:\n{code}")

        # Guard: if the LLM returned an error message instead of code,
        # exec() would raise a cryptic SyntaxError. Catch it here with
        # a clear message so the user sees what actually went wrong.
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            raise RuntimeError(
                f"LLM did not return valid Python code.\n"
                f"Likely cause: LLM call failed or returned an error message.\n"
                f"LLM output was:\n{code}\n\nSyntaxError: {e}"
            ) from e

        namespace: Dict[str, Any] = {"df": df.copy(), "pd": pd, "px": px}
        exec(code, namespace)  # noqa: S102

        return namespace.get("result_df", df), namespace.get("fig"), code


# ──────────────────────────────────────────────────────────────────────────────
# SQL cleanup  (single-pass, replaces the fragile multi-step cleanup
#               that was inline in sql_generator.py)
# ──────────────────────────────────────────────────────────────────────────────

def clean_sql(raw: str) -> str:
    sql = raw.strip()
    sql = re.sub(r"^```(?:sql)?\s*\n?", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\n?```\s*$", "", sql).strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    return sql


# ──────────────────────────────────────────────────────────────────────────────
# Schema context builder  (batched — 3 queries per table regardless of width)
# ──────────────────────────────────────────────────────────────────────────────

def build_schema_context(conn, table_name: str, bts_log: List[str]) -> Dict[str, Any]:
    """
    Compact schema dict for LLM prompt injection.
    Replaces build_structured_schema_metadata() which ran N×3 queries.
    """
    bts_log.append(f"🔍  Building schema context for '{table_name}'…")

    row_count  = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    schema_rows = conn.execute(f"DESCRIBE {table_name}").fetchall()

    null_exprs = ", ".join(
        f"SUM(CASE WHEN \"{r[0]}\" IS NULL THEN 1 ELSE 0 END) AS \"{r[0]}\""
        for r in schema_rows
    )
    null_row = conn.execute(f"SELECT {null_exprs} FROM {table_name}").fetchone()
    null_map  = {r[0]: cnt for r, cnt in zip(schema_rows, null_row)}

    sample_df = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf()

    columns = []
    for row in schema_rows:
        col_name, col_type = row[0], row[1]
        samples = sample_df[col_name].dropna().tolist()[:3]
        columns.append({
            "name": col_name,
            "type": col_type,
            "null_count": null_map.get(col_name, 0),
            "sample_values": [str(s) for s in samples],
        })

    ctx = {"table_name": table_name, "row_count": row_count, "columns": columns}
    bts_log.append(f"   → {len(columns)} columns, {row_count} rows indexed")
    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# Register cleaned DataFrame into DuckDB
# (replaces DuckDBManager.register_csv for cleaned data)
# ──────────────────────────────────────────────────────────────────────────────

def register_cleaned_df(
    conn,
    df: pd.DataFrame,
    table_name: str,
    bts_log: List[str],
) -> None:
    """
    Persists a cleaned pandas DataFrame as a DuckDB table.
    Must be used instead of register_csv() so the cleaning pipeline is respected.
    """
    conn.register("_tmp_cleaned", df)
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _tmp_cleaned")
    conn.unregister("_tmp_cleaned")
    bts_log.append(
        f"📥  Registered '{table_name}' in DuckDB "
        f"({len(df)} rows × {len(df.columns)} cols)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class QueryOrchestrator:
    """
    Wires IntentRouter → LLMSQLGenerator | PythonExecutor → DuckDB.

    Usage:
        orch = QueryOrchestrator(llm=llm, conn=db.conn, sql_generator=sql_gen)
        result = orch.run("Show tickets resolved in under 3 days")
        # result.dataframe, result.chart, result.sql, result.bts_log
    """

    def __init__(
        self,
        llm,
        conn,
        sql_generator=None,
        default_table: Optional[str] = None,
    ):
        self.llm            = llm
        self.conn           = conn
        self.router         = IntentRouter(llm)
        self.python_exec    = PythonExecutor(llm)
        self.sql_generator  = sql_generator
        self.default_table  = default_table

    # ------------------------------------------------------------------
    def _resolve_table(self, bts_log: List[str]) -> str:
        tables = [t[0] for t in self.conn.execute("SHOW TABLES").fetchall()]
        if not tables:
            raise RuntimeError("No tables registered in DuckDB.")
        if self.default_table and self.default_table in tables:
            return self.default_table
        if len(tables) == 1:
            return tables[0]
        bts_log.append(f"⚠️  Multiple tables: {tables}. Using first.")
        return tables[0]

    # ------------------------------------------------------------------
    def run(
        self,
        user_query: str,
        reasoning: bool = False,
        force_intent: Optional[str] = None,
    ) -> QueryResult:

        bts_log: List[str] = []
        result = QueryResult(intent="sql", user_query=user_query, bts_log=bts_log)

        try:
            table_name = self._resolve_table(bts_log)
            schema_ctx = build_schema_context(self.conn, table_name, bts_log)

            intent = force_intent or self.router.route(user_query)
            result.intent = intent
            bts_log.append(f"🧭  Intent routed → {intent.upper()}")

            # ── SQL path ───────────────────────────────────────────────
            if intent == "sql":
                if self.sql_generator is None:
                    raise RuntimeError("sql_generator not set on QueryOrchestrator.")

                bts_log.append("⚙️  Generating SQL via LLM…")
                gen = self.sql_generator.generate_sql(
                    user_query=user_query,
                    schema_metadata=[schema_ctx],
                    relationships=[],
                    reasoning=reasoning,
                )

                if isinstance(gen, dict):
                    sql = clean_sql(gen["sql_query"])
                    result.reasoning = gen.get("reasoning")
                else:
                    sql = clean_sql(gen)

                bts_log.append(f"Generated SQL:\n{sql}")
                result.sql = sql

                # Re-use your existing SQLValidator
                from hybridtablerag.reasoning.sql_generator import SQLValidator
                SQLValidator.validate(sql)
                bts_log.append("✅  SQL passed safety validation")

                result.dataframe = self.conn.execute(sql).fetchdf()
                bts_log.append(
                    f"✅  Returned {len(result.dataframe)} rows "
                    f"× {len(result.dataframe.columns)} cols"
                )

            # ── Python / viz path ──────────────────────────────────────
            elif intent == "python":
                bts_log.append("📊  Python path — loading full table…")
                full_df = self.conn.execute(
                    f"SELECT * FROM {table_name}"
                ).fetchdf()

                result.dataframe, result.chart, code = self.python_exec.execute(
                    user_query=user_query,
                    df=full_df,
                    table_name=table_name,
                    bts_log=bts_log,
                )
                result.sql = f"-- Python path\n{code}"
                bts_log.append("✅  Python execution complete")

        except Exception as exc:
            result.error = str(exc)
            bts_log.append(f"❌  {exc}")
            bts_log.append(traceback.format_exc())

        return result