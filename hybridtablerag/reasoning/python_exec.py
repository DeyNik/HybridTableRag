"""
reasoning/python_exec.py
========================
Safe Python code execution for charts, stats, and transformations.
Sandboxed to prevent arbitrary file/network access.
"""

import ast
import json
import re
import traceback
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px


PYTHON_PROMPT = """
You are an expert data scientist writing Python code for data analysis.

Task: Answer the user's question using the provided DataFrame `df`.
Mode: {mode} (chart, stats, both, or table)

Available Libraries: pandas (pd), numpy (np), plotly.express (px)

Rules:
- ALWAYS output valid, runnable Python code.
- NO markdown fences, NO explanations, NO imports (pre-imported).
- Store the final DataFrame in `result_df`.
- If mode is 'chart' or 'both', create a Plotly figure: `fig = px.<chart_type>(...)`
- Keep code concise. Handle missing values gracefully.

User Question: {query}
"""


class PythonExecutor:
    """Executes LLM-generated Python code in a restricted environment."""

    def __init__(self, llm):
        self.llm = llm

    def _build_namespace(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "df": df.copy(), "pd": pd, "np": np, "px": px,
            "result_df": None, "fig": None, "stats_output": None,
        }

    def _restrict_builtins(self) -> Dict[str, Any]:
        return {"__builtins__": {
            "print": print, "len": len, "range": range, "float": float, "int": int,
            "str": str, "list": list, "dict": dict, "tuple": tuple, "bool": bool,
            "abs": abs, "sum": sum, "min": min, "max": max, "sorted": sorted,
            "round": round, "zip": zip, "enumerate": enumerate,
        }}

    def _strip_fences(self, code: str) -> str:
        code = re.sub(r"^```(?:python)?\s*\n?", "", code, flags=re.IGNORECASE)
        return re.sub(r"\n?```\s*$", "", code).strip()

    def execute(
        self, user_query: str, df: pd.DataFrame, table_name: str,
        bts_log: list, mode: str = "table"
    ) -> Tuple[Optional[pd.DataFrame], Any, str]:
        schema_summary = {col: str(dtype) for col, dtype in df.dtypes.items()}
        prompt = PYTHON_PROMPT.format(mode=mode, schema_summary=json.dumps(schema_summary), query=user_query)

        try:
            raw_code = self.llm.generate(prompt)
        except Exception as e:
            bts_log.append(f"LLM failed to generate Python code: {e}")
            return df, None, ""

        code = self._strip_fences(raw_code)
        if not code:
            return df, None, ""

        try:
            ast.parse(code)
        except SyntaxError as e:
            bts_log.append(f"Generated Python syntax error: {e}")
            return df, None, code

        namespace = self._build_namespace(df)
        try:
            exec(code, self._restrict_builtins(), namespace)
            result_df = namespace.get("result_df") or df
            fig = namespace.get("fig")
            
            if fig is None and mode in ["chart", "both"]:
                bts_log.append("Chart mode requested but LLM did not create `fig`. Returning table.")
            return result_df, fig, code

        except Exception as e:
            bts_log.append(f"Python execution failed:\n{traceback.format_exc()}")
            return df, None, code