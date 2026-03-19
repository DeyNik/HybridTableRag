"""
reasoning/query_orchestrator.py
================================
Full file — paste this replacing your existing query_orchestrator.py

Fixes vs previous version:
  - SQLValidator + clean_sql imported once at top level (not re-imported inside run())
  - SQL path: retry loop on DuckDB execution errors (feeds error back to LLM, max 3 attempts)
  - SQL path: DuckDB error detail preserved in bts_log and result.error
  - Python path: SQL pre-filter + 3-attempt retry loop
  - bts_log prefixes consistent (checkmark, gear, warning, cross) so UI colour-coding works
  - _resolve_table warning is clearer
"""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from hybridtablerag.reasoning.sql_generator import SQLValidator, clean_sql


# Result container

@dataclass
class QueryResult:
    intent: str                                   # "both" | "sql" | "python"
    user_query: str

    # SQL path results
    sql: Optional[str]                      = None
    dataframe: Optional[pd.DataFrame]       = None
    reasoning: Optional[str]                = None

    # Python path results
    python_code: Optional[str]              = None
    python_dataframe: Optional[pd.DataFrame] = None
    chart: Optional[Any]                    = None
    python_narrative: Optional[str]         = None

    # Shared
    error: Optional[str]                    = None
    python_error: Optional[str]             = None
    bts_log: List[str]                      = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def has_sql(self) -> bool:
        return self.dataframe is not None

    @property
    def has_python(self) -> bool:
        return self.python_dataframe is not None or self.chart is not None


# Intent router
# Always runs BOTH paths. The router classifies what the Python path should
# produce (chart / stats / both / table) so PythonExecutor knows its goal.

_INTENT_PROMPT = """\
You are a query router for a data analytics system that ALWAYS runs two paths:
  1. SQL path  -- executes a DuckDB SQL query for structured results
  2. Python path -- performs data science analysis + optional visualisation

Your job is to classify what the Python path should do for this question.

Respond with ONLY one of these four words:
  chart    -- user wants a graph / plot / visualisation
  stats    -- user wants statistical analysis (distributions, correlations,
              outliers, trends, regression, percentiles, cohort analysis)
  both     -- user wants analysis AND a chart makes the answer clearer
  table    -- question is purely structural (SQL result is enough,
              Python should return a clean summary table)

No explanation. No punctuation. Just one word.

User question: {query}
"""


class IntentRouter:
    """
    Classifies what the Python path should produce.
    SQL path ALWAYS runs — this only controls the Python executor mode.
    """
    def __init__(self, llm):
        self.llm = llm

    def route(self, user_query: str) -> str:
        """Returns one of: chart | stats | both | table"""
        try:
            response = self.llm.generate(
                _INTENT_PROMPT.format(query=user_query)
            ).strip().lower()
            if "both" in response:
                return "both"
            if "stats" in response:
                return "stats"
            if any(k in response for k in ("chart", "plot", "visual", "graph")):
                return "chart"
            return "table"
        except Exception:
            return "table"


# Python executor — full data science + analytical + visualisation path

_PYTHON_PROMPT = """\
You are an expert data scientist with deep knowledge of pandas, numpy, scipy,
and plotly. You have access to a pandas DataFrame called `df`.

Table: {table_name}
Columns and types:
{columns}

Column value distributions (use EXACT values for any filters):
{distributions}

Sample rows (first 3):
{sample}

Available in scope — use freely:
  df         — the DataFrame
  pd         — pandas
  np         — numpy (statistics, array ops, percentile, corrcoef, histogram, etc.)
  scipy      — scipy.stats (pearsonr, spearmanr, ttest_ind, chi2_contingency, etc.)
  px         — plotly.express  (bar, line, scatter, box, histogram, heatmap, pie, sunburst, etc.)
  go         — plotly.graph_objects  (multi-trace, subplots, custom layouts)
  make_subplots — plotly.subplots.make_subplots
  dt         — datetime module
  Counter    — collections.Counter

User request:
{query}

INSTRUCTIONS:
1. Think about what analytical approach best answers this question:
   - Simple aggregation → use pandas groupby / value_counts
   - Distribution / spread → use np.percentile, np.std, scipy.stats.describe
   - Correlation / relationship → use np.corrcoef or scipy.stats.pearsonr / spearmanr
   - Trend over time → resample by date period, plot as line chart
   - Comparison across groups → groupby + bar chart or box plot
   - Outlier detection → IQR method with np.percentile
   - Statistical test → scipy.stats

2. Always store the tabular result as `result_df` (a pandas DataFrame).
   Even for pure chart questions, produce a summary DataFrame as result_df.

3. If a chart would make the answer clearer (even if not explicitly asked),
   create a plotly figure stored as `fig`. Good candidates:
   - Any grouped count or sum → bar chart
   - Any time series → line chart
   - Any distribution → histogram or box plot
   - Any correlation matrix → heatmap
   Always add a descriptive title, axis labels, and use a clean template="plotly_white".

4. Never call plt.show(), st.*, print(), or display().
5. Handle NaN values before any aggregation — use .dropna() or .fillna() as appropriate.
6. Use the exact column names shown above — do not guess or rename.

Return ONLY executable Python code. No markdown fences. No explanations.
"""

_PYTHON_RETRY_PROMPT = """\
The Python code you generated failed at runtime.

Original user request:
{query}

DataFrame columns and types:
{columns}

Code that failed:
{code}

Error:
{error}

Fix ONLY what caused the error. Common causes:
- Wrong column name — use the exact names listed above
- Method not available on this dtype (e.g. .str on numeric column)
- NaN not handled before aggregation — add .dropna() first
- scipy not imported — use `from scipy import stats` inside the code
- Plotly API mismatch — check px vs go method signatures
- make_subplots rows/cols mismatch with number of traces added

Return ONLY the fixed Python code. No markdown fences. No explanations.
"""


class PythonExecutor:

    MAX_RETRIES = 3

    def __init__(self, llm):
        self.llm = llm

    def _build_namespace(self, df: pd.DataFrame) -> Dict[str, Any]:
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly.subplots as _subplots
        import numpy as np
        import datetime
        from collections import Counter
        try:
            import scipy
        except ImportError:
            scipy = None
        return {
            "df":            df.copy(),
            "pd":            pd,
            "np":            np,
            "px":            px,
            "go":            go,
            "make_subplots": _subplots.make_subplots,
            "dt":            datetime,
            "Counter":       Counter,
            "scipy":         scipy,
        }

    def _col_summary(self, df: pd.DataFrame) -> str:
        return "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())

    def _distribution_summary(self, df: pd.DataFrame) -> str:
        """
        Build a compact distribution block from the actual DataFrame so
        the LLM knows exact categorical values, numeric ranges, and date
        ranges — same information that build_schema_context provides, but
        derived directly from the focused df passed to the executor.
        """
        lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].dropna()
            if len(non_null) == 0:
                lines.append(f"  {col}: all null")
                continue
            n_unique = non_null.nunique()
            if n_unique <= 25 and dtype in ("object", "string", "category", "bool"):
                vc = non_null.value_counts()
                val_str = ", ".join(f"{v} ({c})" for v, c in vc.items())
                lines.append(f"  {col}: {val_str}")
            elif "int" in dtype or "float" in dtype:
                lines.append(
                    f"  {col}: min={non_null.min()}, max={non_null.max()}, "
                    f"mean={non_null.mean():.2f}"
                )
            elif "datetime" in dtype or "date" in dtype:
                lines.append(f"  {col}: {non_null.min()} → {non_null.max()}")
            else:
                samples = non_null.head(3).tolist()
                lines.append(f"  {col}: samples — {', '.join(str(s) for s in samples)}")
        return "\n".join(lines)

    def _strip_fences(self, code: str) -> str:
        code = code.strip()
        code = re.sub(r"^```(?:python)?\s*\n?", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\n?```\s*$", "", code).strip()
        return code

    def execute(
        self,
        user_query: str,
        df: pd.DataFrame,
        table_name: str,
        bts_log: List[str],
        python_mode: str = "table",   # chart | stats | both | table
    ) -> tuple[Optional[pd.DataFrame], Optional[Any], str]:
        """
        python_mode controls the analytical goal:
          chart  — prioritise a good visualisation
          stats  — prioritise statistical depth (numpy/scipy)
          both   — full analysis + chart
          table  — clean summary DataFrame is enough
        """
        col_summary   = self._col_summary(df)
        distributions = self._distribution_summary(df)

        # Append mode hint to the user query so the LLM understands its goal
        mode_hints = {
            "chart": "Produce a high-quality plotly chart as the primary output.",
            "stats": (
                "Perform in-depth statistical analysis. Use numpy and scipy freely: "
                "descriptive stats, percentiles, correlations, distributions, outlier "
                "detection, or statistical tests as appropriate. "
                "The result_df should contain the statistical findings."
            ),
            "both": (
                "Perform statistical analysis AND produce a plotly chart that best "
                "visualises the findings. Use numpy/scipy for the analysis, "
                "plotly for the chart."
            ),
            "table": "Return a clean, well-structured summary DataFrame as result_df.",
        }
        augmented_query = f"{user_query}\n\nGoal: {mode_hints.get(python_mode, '')}"

        prompt = _PYTHON_PROMPT.format(
            table_name=table_name,
            columns=col_summary,
            distributions=distributions,
            sample=df.head(3).to_string(index=False),
            query=augmented_query,
        )
        code = ""

        for attempt in range(1, self.MAX_RETRIES + 1):
            bts_log.append(f"Python attempt {attempt}/{self.MAX_RETRIES}")

            raw  = self.llm.generate(prompt).strip()
            code = self._strip_fences(raw)
            bts_log.append(f"Generated code:\n{code}")

            try:
                compile(code, "<string>", "exec")
            except SyntaxError as e:
                err = f"SyntaxError: {e}"
                bts_log.append(f"Syntax error on attempt {attempt}: {e}")
                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(f"Python failed after {self.MAX_RETRIES} attempts. Last: {err}")
                prompt = _PYTHON_RETRY_PROMPT.format(
                    query=user_query, columns=col_summary, code=code, error=err
                )
                continue

            namespace = self._build_namespace(df)
            try:
                exec(code, namespace)  # noqa: S102
            except Exception as e:
                err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                bts_log.append(f"Runtime error on attempt {attempt}: {type(e).__name__}: {e}")
                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(f"Python failed after {self.MAX_RETRIES} attempts. Last: {err}")
                prompt = _PYTHON_RETRY_PROMPT.format(
                    query=user_query, columns=col_summary, code=code, error=err
                )
                continue

            result_df = namespace.get("result_df")
            fig       = namespace.get("fig")

            if result_df is None:
                bts_log.append("result_df not assigned - falling back to input df.")
                result_df = df
            elif not isinstance(result_df, pd.DataFrame):
                bts_log.append(f"result_df is {type(result_df).__name__} - converting.")
                try:
                    result_df = pd.DataFrame(result_df)
                except Exception:
                    result_df = df

            bts_log.append(
                f"Python succeeded (attempt {attempt}) - "
                f"{len(result_df)} rows, chart={'yes' if fig else 'no'}"
            )
            return result_df, fig, code

        raise RuntimeError("PythonExecutor exhausted all retry attempts.")


# SQL retry prompt

_SQL_RETRY_PROMPT = """\
The SQL query you generated failed when executed against DuckDB.

User question:
{query}

Available schema:
{schema_json}

SQL that failed:
{sql}

DuckDB error:
{error}

Common DuckDB-specific fixes:
- DATE - DATE returns INTEGER (days), not INTERVAL
- Use double-quotes for column names with spaces
- STRFTIME: use strftime('%Y-%m', col) not TO_CHAR
- No ILIKE - use LOWER(col) LIKE LOWER('%pattern%')
- Window functions require ORDER BY inside OVER()

Return ONLY the corrected SQL. No markdown. No explanation.
"""

_EMPTY_RESULT_PROMPT = """\
The SQL query executed successfully but returned 0 rows.

User question:
{query}

SQL that returned 0 rows:
{sql}

Actual column values in the data (use EXACTLY these values in filters):
{col_distributions}

Diagnose why 0 rows were returned and write a corrected SQL query.

Most common causes:
- Wrong value in WHERE clause — check the exact values listed above
- Case mismatch (e.g. 'high' vs 'High' vs 'HIGH')
- Date range that excludes all data — check the range listed above
- AND condition that is too restrictive — try relaxing one condition

If this question genuinely cannot be answered from this data, respond with:
NO_RESULTS: <one sentence explaining why>

Otherwise return ONLY the corrected SQL query. No markdown. No explanation.
"""


# ── CHANGE 1: Enriched schema context ────────────────────────────────────────
# For categorical columns  (distinct_count <= CATEGORICAL_THRESHOLD):
#   fetch every distinct value + its count so the LLM knows exactly what to
#   filter on.  Eliminates "WHERE priority = 'Urgent'" when only 'High'/'Low'
#   exist — the single biggest source of 0-row results.
# For numeric/date columns: fetch min, max, avg so the LLM can write
#   sensible range filters without guessing.
# Cost: one extra batched query per table at schema-build time.
# ─────────────────────────────────────────────────────────────────────────────

CATEGORICAL_THRESHOLD = 25   # columns with <= this many distinct values get full distribution

def _safe_val(v: Any) -> Any:
    """Make a value JSON-safe (handles date, Decimal, numpy types)."""
    import datetime
    from decimal import Decimal
    if v is None:
        return None
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    try:
        import numpy as np
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
    except ImportError:
        pass
    return v


def build_schema_context(conn, table_name: str, bts_log: List[str]) -> Dict[str, Any]:
    bts_log.append(f"Building schema context for '{table_name}'")

    row_count   = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    schema_rows = conn.execute(f"DESCRIBE {table_name}").fetchall()

    # Batched null counts — one query regardless of column count
    null_exprs = ", ".join(
        f'SUM(CASE WHEN "{r[0]}" IS NULL THEN 1 ELSE 0 END) AS "{r[0]}"'
        for r in schema_rows
    )
    null_row = conn.execute(f"SELECT {null_exprs} FROM {table_name}").fetchone()
    null_map  = {r[0]: cnt for r, cnt in zip(schema_rows, null_row)}

    # Distinct counts — one batched query
    distinct_exprs = ", ".join(
        f'COUNT(DISTINCT "{r[0]}") AS "{r[0]}"' for r in schema_rows
    )
    distinct_row  = conn.execute(f"SELECT {distinct_exprs} FROM {table_name}").fetchone()
    distinct_map  = {r[0]: cnt for r, cnt in zip(schema_rows, distinct_row)}

    # Sample rows for high-cardinality / text columns
    sample_df = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf()

    columns = []
    categorical_count = 0
    enriched_count    = 0

    for row in schema_rows:
        col_name  = row[0]
        col_type  = row[1].upper()
        null_cnt  = null_map.get(col_name, 0)
        dist_cnt  = distinct_map.get(col_name, 0)

        col_meta: Dict[str, Any] = {
            "name":       col_name,
            "type":       col_type,
            "null_count": null_cnt,
            "distinct":   dist_cnt,
        }

        is_numeric = any(t in col_type for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "BIGINT", "HUGEINT"))
        is_date    = any(t in col_type for t in ("DATE", "TIMESTAMP"))
        is_text    = any(t in col_type for t in ("VARCHAR", "TEXT", "STRING"))

        # ── Categorical: full value distribution ──────────────────────────
        if dist_cnt <= CATEGORICAL_THRESHOLD and dist_cnt > 0:
            try:
                vc_rows = conn.execute(
                    f'SELECT "{col_name}", COUNT(*) as cnt '
                    f'FROM {table_name} '
                    f'WHERE "{col_name}" IS NOT NULL '
                    f'GROUP BY "{col_name}" ORDER BY cnt DESC'
                ).fetchall()
                col_meta["all_values"] = {
                    str(_safe_val(r[0])): int(r[1]) for r in vc_rows
                }
                categorical_count += 1
                enriched_count    += 1
            except Exception:
                # Never crash schema build — fall back to samples
                samples = sample_df[col_name].dropna().tolist()[:3]
                col_meta["sample_values"] = [str(s) for s in samples]

        # ── Numeric: min / max / avg ──────────────────────────────────────
        elif is_numeric:
            try:
                stats = conn.execute(
                    f'SELECT MIN("{col_name}"), MAX("{col_name}"), '
                    f'AVG("{col_name}") FROM {table_name}'
                ).fetchone()
                col_meta["range"] = {
                    "min": _safe_val(stats[0]),
                    "max": _safe_val(stats[1]),
                    "avg": round(float(stats[2]), 2) if stats[2] is not None else None,
                }
                enriched_count += 1
            except Exception:
                samples = sample_df[col_name].dropna().tolist()[:3]
                col_meta["sample_values"] = [str(s) for s in samples]

        # ── Date: min / max ───────────────────────────────────────────────
        elif is_date:
            try:
                stats = conn.execute(
                    f'SELECT MIN("{col_name}"), MAX("{col_name}") FROM {table_name}'
                ).fetchone()
                col_meta["range"] = {
                    "min": _safe_val(stats[0]),
                    "max": _safe_val(stats[1]),
                }
                enriched_count += 1
            except Exception:
                samples = sample_df[col_name].dropna().tolist()[:3]
                col_meta["sample_values"] = [str(s) for s in samples]

        # ── High-cardinality text: 3 samples ─────────────────────────────
        else:
            samples = sample_df[col_name].dropna().tolist()[:3]
            col_meta["sample_values"] = [str(s) for s in samples]

        columns.append(col_meta)

    ctx = {"table_name": table_name, "row_count": row_count, "columns": columns}
    bts_log.append(
        f"   {len(columns)} columns, {row_count} rows | "
        f"{categorical_count} categorical (full distribution), "
        f"{enriched_count} enriched total"
    )
    return ctx


def register_cleaned_df(conn, df: pd.DataFrame, table_name: str, bts_log: List[str]) -> None:
    conn.register("_tmp_cleaned", df)
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _tmp_cleaned")
    conn.unregister("_tmp_cleaned")
    bts_log.append(f"Registered '{table_name}' in DuckDB ({len(df)} rows x {len(df.columns)} cols)")


# Orchestrator

class QueryOrchestrator:

    SQL_MAX_RETRIES = 3

    def __init__(self, llm, conn, sql_generator=None, default_table: Optional[str] = None):
        self.llm           = llm
        self.conn          = conn
        self.router        = IntentRouter(llm)
        self.python_exec   = PythonExecutor(llm)
        self.sql_generator = sql_generator
        self.default_table = default_table

    def _resolve_table(self, bts_log: List[str]) -> str:
        tables = [t[0] for t in self.conn.execute("SHOW TABLES").fetchall()]
        if not tables:
            raise RuntimeError("No tables registered in DuckDB.")
        if self.default_table and self.default_table in tables:
            return self.default_table
        if len(tables) == 1:
            return tables[0]
        bts_log.append(
            f"Multiple tables {tables} and no default set - "
            f"using '{tables[0]}'. Set orchestrator.default_table to be explicit."
        )
        return tables[0]

    def _run_sql_with_retry(
        self,
        user_query: str,
        schema_ctx: Dict[str, Any],
        reasoning: bool,
        bts_log: List[str],
    ) -> tuple[str, pd.DataFrame, Optional[str]]:
        import json

        if self.sql_generator is None:
            raise RuntimeError("sql_generator not set on QueryOrchestrator.")

        sql           = ""
        reasoning_out = None
        last_error    = ""

        for attempt in range(1, self.SQL_MAX_RETRIES + 1):
            bts_log.append(f"SQL attempt {attempt}/{self.SQL_MAX_RETRIES}")

            if attempt == 1:
                gen = self.sql_generator.generate_sql(
                    user_query=user_query,
                    schema_metadata=[schema_ctx],
                    relationships=[],
                    reasoning=reasoning,
                )
            else:
                retry_prompt = _SQL_RETRY_PROMPT.format(
                    query=user_query,
                    schema_json=json.dumps(schema_ctx, indent=2),
                    sql=sql,
                    error=last_error,
                )
                raw_retry = self.sql_generator.llm.generate(retry_prompt).strip()
                gen = clean_sql(raw_retry)

            if isinstance(gen, dict):
                sql           = clean_sql(gen["sql_query"])
                reasoning_out = gen.get("reasoning")
            else:
                sql = clean_sql(gen)

            bts_log.append(f"Generated SQL:\n{sql}")

            SQLValidator.validate(sql)
            bts_log.append("SQL passed safety validation")

            try:
                df = self.conn.execute(sql).fetchdf()
            except Exception as e:
                last_error = str(e)
                bts_log.append(f"DuckDB error on attempt {attempt}: {last_error}")
                if attempt == self.SQL_MAX_RETRIES:
                    raise RuntimeError(
                        f"SQL execution failed after {self.SQL_MAX_RETRIES} attempts.\n"
                        f"Last SQL:\n{sql}\nLast error: {last_error}"
                    )
                continue   # retry with _SQL_RETRY_PROMPT on next iteration

            bts_log.append(f"Query returned {len(df)} rows x {len(df.columns)} cols")

            # ── CHANGE 3: empty-result check ──────────────────────────────
            # A 0-row result is not a DuckDB error — it's a silent wrong answer.
            # Most common cause: LLM used a value that doesn't exist in the data
            # (e.g. WHERE priority = 'Urgent' when only 'High'/'Medium' exist).
            # Diagnose by feeding the actual column distributions back to the LLM.
            if len(df) == 0 and attempt < self.SQL_MAX_RETRIES:
                bts_log.append(
                    "Query returned 0 rows — checking for wrong values or logic..."
                )
                # Build a compact distribution hint from schema_ctx
                dist_lines = []
                for col in schema_ctx.get("columns", []):
                    if col.get("all_values"):
                        vals = list(col["all_values"].keys())
                        dist_lines.append(f"  {col['name']}: {vals}")
                    elif col.get("range"):
                        dist_lines.append(
                            f"  {col['name']}: range {col['range']['min']} → {col['range']['max']}"
                        )
                dist_block = "\n".join(dist_lines) if dist_lines else "  (no distribution data)"

                empty_prompt = _EMPTY_RESULT_PROMPT.format(
                    query=user_query,
                    sql=sql,
                    col_distributions=dist_block,
                )
                raw_fix = self.sql_generator.llm.generate(empty_prompt).strip()
                raw_fix = clean_sql(raw_fix)

                # LLM may respond with NO_RESULTS if the question genuinely
                # has no answer — surface that as a clean message, not an error
                if raw_fix.upper().startswith("NO_RESULTS"):
                    explanation = raw_fix[10:].strip().lstrip(":").strip()
                    bts_log.append(f"LLM confirms no data: {explanation}")
                    # Return empty df with the explanation preserved in bts_log
                    return sql, df, reasoning_out

                bts_log.append(f"Revised SQL after empty-result check:\n{raw_fix}")
                sql = raw_fix
                SQLValidator.validate(sql)

                try:
                    df = self.conn.execute(sql).fetchdf()
                    bts_log.append(
                        f"Revised query returned {len(df)} rows x {len(df.columns)} cols"
                    )
                except Exception as e:
                    bts_log.append(f"Revised SQL also failed: {e}")
                    # Don't raise — return the empty df from the original query
                    # so the UI shows something rather than a hard error

                return sql, df, reasoning_out
            # ── end Change 3 ──────────────────────────────────────────────

            return sql, df, reasoning_out

        raise RuntimeError("SQL retry loop exhausted.")

    def run(
        self,
        user_query: str,
        reasoning: bool = False,
        force_intent: Optional[str] = None,
    ) -> QueryResult:
        """
        Always runs BOTH SQL and Python paths.
        force_intent now controls the Python executor MODE (chart/stats/both/table)
        rather than choosing between sql vs python.

        SQL path:    structured result table via DuckDB
        Python path: data science analysis + optional chart via numpy/scipy/plotly
        Both results are stored in QueryResult and rendered side by side in the UI.
        """
        bts_log: List[str] = []
        result = QueryResult(intent="both", user_query=user_query, bts_log=bts_log)

        try:
            table_name = self._resolve_table(bts_log)
            schema_ctx = build_schema_context(self.conn, table_name, bts_log)

            # Router classifies Python mode — SQL always runs regardless
            python_mode = force_intent or self.router.route(user_query)
            result.intent = python_mode
            bts_log.append(f"Python mode: {python_mode.upper()} | SQL: always on")

            # ── SQL path (always) ──────────────────────────────────────────
            bts_log.append("--- SQL PATH ---")
            try:
                result.sql, result.dataframe, result.reasoning = self._run_sql_with_retry(
                    user_query=user_query,
                    schema_ctx=schema_ctx,
                    reasoning=reasoning,
                    bts_log=bts_log,
                )
            except Exception as sql_exc:
                result.error = str(sql_exc)
                bts_log.append(f"SQL path failed: {sql_exc}")
                # Python path still runs even if SQL fails

            # ── Python path (always) ───────────────────────────────────────
            bts_log.append("--- PYTHON PATH ---")
            try:
                # SQL pre-filter: get focused df for Python analysis
                # Uses the SQL result if available, otherwise generates its own filter
                focused_df = None

                if result.dataframe is not None and not result.dataframe.empty:
                    # Reuse SQL result as focused input — avoids duplicate DuckDB call
                    focused_df = result.dataframe.copy()
                    bts_log.append(
                        f"Python using SQL result as input: "
                        f"{len(focused_df)} rows x {len(focused_df.columns)} cols"
                    )
                elif self.sql_generator:
                    try:
                        bts_log.append("Python: generating SQL pre-filter...")
                        pre_sql_raw = self.sql_generator.generate_sql(
                            user_query=(
                                f"Return only the columns and rows relevant to: {user_query}"
                            ),
                            schema_metadata=[schema_ctx],
                            relationships=[],
                            reasoning=False,
                        )
                        pre_sql = clean_sql(
                            pre_sql_raw["sql_query"]
                            if isinstance(pre_sql_raw, dict)
                            else pre_sql_raw
                        )
                        SQLValidator.validate(pre_sql)
                        focused_df = self.conn.execute(pre_sql).fetchdf()
                        bts_log.append(
                            f"Pre-filter: {len(focused_df)} rows x {len(focused_df.columns)} cols"
                        )
                    except Exception as e:
                        bts_log.append(f"Pre-filter failed ({e}), using full table.")

                if focused_df is None or focused_df.empty:
                    focused_df = self.conn.execute(
                        f"SELECT * FROM {table_name}"
                    ).fetchdf()
                    bts_log.append(f"Full table loaded: {len(focused_df)} rows")

                result.python_dataframe, result.chart, result.python_code = (
                    self.python_exec.execute(
                        user_query=user_query,
                        df=focused_df,
                        table_name=table_name,
                        bts_log=bts_log,
                        python_mode=python_mode,
                    )
                )
                bts_log.append("Python execution complete")

            except Exception as py_exc:
                result.python_error = str(py_exc)
                bts_log.append(f"Python path failed: {py_exc}")
                bts_log.append(traceback.format_exc())
                # SQL result is still valid — don't overwrite result.error

        except Exception as exc:
            result.error = str(exc)
            bts_log.append(f"Orchestrator error: {exc}")
            bts_log.append(traceback.format_exc())

        return result