"""
reasoning/python_exec.py
========================
LLM-generated Python execution for visualization and statistical analysis.

Fixes vs previous version
--------------------------
- PYTHON_PROMPT: schema_summary was built but never passed to .format() → KeyError fixed
- _restrict_builtins: removed — broke numpy/pandas internals. Using safe namespace instead.
- Retry loop: 3 attempts with error context fed back to LLM
- Mode comes from QueryPlan (passed in), not a separate LLM call
- Distribution summary injected into prompt so LLM knows exact column values
"""

from __future__ import annotations

import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# Prompts

_PYTHON_PROMPT = """\
You are an expert data scientist writing Python for data analysis and visualization.

DataFrame: `df`
Table: {table_name}

Column types:
{col_types}

Column distributions (use EXACT values shown here for any filters or labels):
{distributions}

Sample rows (first 3):
{sample}

Available in scope — use freely:
  df, pd, np, px (plotly.express), go (plotly.graph_objects),
  make_subplots, scipy, dt (datetime), Counter

Goal: {mode_hint}

User question:
{query}

Instructions:
1. Store the result as `result_df` (a pandas DataFrame, even for chart-only answers).
2. If a chart would make the answer clearer, create `fig` using px or go.
   - Add descriptive title and axis labels.
   - Use template="plotly_white".
   - Do NOT define `fig` if no chart is needed.
3. Handle NaN values before aggregation (.dropna() or .fillna()).
4. Use the EXACT column names shown above — do not rename or guess.
5. Never call plt.show(), st.*, print(), or display().

Return ONLY executable Python code. No markdown fences. No explanations.
"""

_PYTHON_RETRY_PROMPT = """\
The code you generated failed.

User question: {query}
Mode: {mode}

Column types:
{col_types}

Code that failed:
{code}

Error:
{error}

Fix the error. Common causes:
- Wrong column name — use exact names from column types above
- NaN not handled before groupby/agg — add .dropna()
- Wrong dtype for operation — check column types
- Plotly API mismatch — use px.bar(df, x=..., y=...) pattern

Return ONLY the corrected Python code. No fences. No explanation.
"""

_MODE_HINTS = {
    "chart":  "Produce a high-quality Plotly chart as the primary output. Result_df should be the data behind the chart.",
    "stats":  "Perform in-depth statistical analysis using numpy/scipy. Result_df should contain the statistical findings.",
    "both":   "Perform statistical analysis AND produce a Plotly chart that best visualises the findings.",
    "table":  "Return a clean, well-structured summary DataFrame as result_df. A chart is optional but welcome if helpful.",
}


# Executor 

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
            "__builtins__": {
                "len": len,
                "range": range,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "float": float,
                "int": int,
                "str": str,
                "list": list,
                "dict": dict,
                "set": set,
                "enumerate": enumerate,
                "zip": zip,
            },
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
            "make_subplots": _subplots.make_subplots,
            "scipy": scipy,
            "dt": datetime,
            "Counter": Counter,
            "result_df": None,
            "fig": None,
        }

    def _col_types(self, df: pd.DataFrame) -> str:
        return "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())

    def _distribution_summary(self, df: pd.DataFrame) -> str:
        """Compact distribution for each column — exact values for LLM."""
        lines = []
        for col in df.columns:
            non_null = df[col].dropna()
            if non_null.empty:
                lines.append(f"  {col}: all null")
                continue
            dtype = str(df[col].dtype)
            n_unique = non_null.nunique()
            if n_unique <= 25 and dtype in ("object", "string", "category", "bool"):
                vc = non_null.value_counts()
                val_str = ", ".join(f"{v} ({c})" for v, c in vc.items())
                lines.append(f"  {col}: {val_str}")
            elif "int" in dtype or "float" in dtype:
                lines.append(f"  {col}: min={non_null.min()}, max={non_null.max()}, mean={non_null.mean():.2f}")
            elif "datetime" in dtype or "date" in dtype:
                lines.append(f"  {col}: {non_null.min()} → {non_null.max()}")
            else:
                lines.append(f"  {col}: samples — {', '.join(str(s) for s in non_null.head(3).tolist())}")
        return "\n".join(lines)

    def _strip_fences(self, code: str) -> str:
        code = code.strip()
        code = re.sub(r"^```(?:python)?\s*\n?", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\n?```\s*$", "", code).strip()
        return code

    def execute(
        self,
        user_query:  str,
        df:          pd.DataFrame,
        table_name:  str,
        bts_log:     List[str],
        mode:        str = "table",
        visualization_reason: str = "",   # from QueryPlan used to enrich mode hint
    ) -> Tuple[Optional[pd.DataFrame], Any, str]:
        """
        Execute LLM-generated Python with up to MAX_RETRIES attempts.
        Returns (result_df, fig, code).
        """
        col_types     = self._col_types(df)
        distributions = self._distribution_summary(df)
        sample        = df.head(3).to_string(index=False)

        # Enrich mode hint with visualization_reason from QueryPlan
        mode_hint = _MODE_HINTS.get(mode, _MODE_HINTS["table"])
        if visualization_reason:
            mode_hint += f" Context: {visualization_reason}"

        prompt = _PYTHON_PROMPT.format(
            table_name=table_name,
            col_types=col_types,
            distributions=distributions,
            sample=sample,
            mode_hint=mode_hint,
            query=user_query,
        )

        code = ""
        last_error = ""

        for attempt in range(1, self.MAX_RETRIES + 1):
            bts_log.append(f"Python attempt {attempt}/{self.MAX_RETRIES}")

            try:
                raw = self.llm.generate(prompt).strip()
                code = self._strip_fences(raw)
            except Exception as e:
                bts_log.append(f"LLM generation failed: {e}")
                return df, None, ""

            if not code:
                bts_log.append("Empty code returned")
                return df, None, ""

            # Syntax check before exec
            try:
                compile(code, "<string>", "exec")
            except SyntaxError as e:
                last_error = f"SyntaxError: {e}"
                bts_log.append(f"Syntax error on attempt {attempt}: {e}")
                if attempt == self.MAX_RETRIES:
                    return df, None, code
                prompt = _PYTHON_RETRY_PROMPT.format(
                    query=user_query, mode=mode,
                    col_types=col_types, code=code, error=last_error,
                )
                continue

            # Execute
            namespace = self._build_namespace(df)
            try:
                exec(code, namespace)   # noqa: S102
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                bts_log.append(f"Runtime error on attempt {attempt}: {type(e).__name__}: {e}")
                if attempt == self.MAX_RETRIES:
                    return df, None, code
                prompt = _PYTHON_RETRY_PROMPT.format(
                    query=user_query, mode=mode,
                    col_types=col_types, code=code, error=last_error,
                )
                continue

            # Extract outputs
            result_df = namespace.get("result_df")
            fig       = namespace.get("fig")

            # Validate result_df
            if result_df is None:
                bts_log.append(" result_df not assigned - falling back to input df")
                result_df = df
            elif not isinstance(result_df, pd.DataFrame):
                bts_log.append(f"result_df is {type(result_df).__name__} — converting")
                try:
                    result_df = pd.DataFrame(result_df)
                except Exception:
                    result_df = df

            # Chart requested but not produced -- log, don't retry
            if mode in ("chart", "both") and fig is None:
                bts_log.append(" Chart mode requested but `fig` not created by LLM")

            bts_log.append(
                f"Python succeeded (attempt {attempt}) — "
                f"{len(result_df)} rows, chart={'yes' if fig else 'no'}"
            )
            return result_df, fig, code

        return df, None, code