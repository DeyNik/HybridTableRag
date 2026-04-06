"""
reasoning/sql.py
================
LLM → DuckDB SQL generation.

Fixes vs previous version
--------------------------
- _format_schema_for_prompt: all_values accepted as BOTH dict {str:int} AND list [{value,count}]
- _basic_sql_validation: schema_metadata accepted as dict (multi-table) OR list (single-table)
- generate_sql: schema_metadata accepted as dict or list — normalised internally
- Hybrid SQL+vector: when plan.needs_semantic=True, injects HNSW distance expression into SQL
- QueryPlan-aware prompt: uses plan hints for better SQL generation
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

from hybridtablerag.llm.base import BaseLLM


# Safety 

FORBIDDEN_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE", "REPLACE"]


class SQLValidator:
    @staticmethod
    def validate(sql: str) -> None:
        # Strip string literals before keyword check to avoid false positives
        sql_no_strings = re.sub(r"'[^']*'", "''", sql)
        sql_no_strings = re.sub(r'"[^"]*"', '""', sql_no_strings)
        sql_upper = sql_no_strings.upper().strip()

        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            raise ValueError("Only SELECT statements (with optional CTEs) are allowed.")

        for keyword in FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{keyword}\b", sql_upper):
                before = sql[:re.search(rf"\b{keyword}\b", sql_upper).start()]
                if before.count("'") % 2 == 0:
                    raise ValueError(f"Forbidden SQL keyword: {keyword}")


# SQL cleanup

def clean_sql(raw: str) -> str:
    sql = raw.strip()
    sql = re.sub(r"^```(?:sql)?\s*\n?", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\n?```\s*$", "", sql).strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    return sql


# DuckDB date fix 

def _fix_duckdb_date_arithmetic(sql: str) -> tuple[str, list[str]]:
    fixes: list[str] = []
    interval_cmp = re.compile(
        r'(CAST\s*\([^)]+\)|\b\w+(?:\.\w+)?)\s*-\s*'
        r'(CAST\s*\([^)]+\)|\b\w+(?:\.\w+)?)\s*([<>=!]+)\s*'
        r"INTERVAL\s*['\"](\d+)\s*(\w+)['\"]",
        re.IGNORECASE,
    )
    def _rewrite(m):
        lhs, rhs, op, n, unit = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5).lower().rstrip('s')
        if unit == 'day':
            fixes.append("Fixed DATE subtraction vs INTERVAL → integer")
            return f"{lhs} - {rhs} {op} {n}"
        fixes.append(f"Fixed INTERVAL → datediff('{unit}')")
        return f"datediff('{unit}', {rhs}, {lhs}) {op} {n}"
    return interval_cmp.sub(_rewrite, sql), fixes


# Schema formatter

def _normalise_schema_input(schema_metadata) -> List[Dict]:
    """
    Accept schema_metadata as:
      A) dict with "tables" key  (multi-table from build_multi_table_schema_context)
      B) list of table dicts     (legacy single/multi table)
      C) single table dict       (legacy)
    Always returns a list of table dicts.
    """
    if isinstance(schema_metadata, dict):
        if "tables" in schema_metadata:
            return schema_metadata["tables"]       # multi-table dict → extract list
        return [schema_metadata]                   # single table dict
    if isinstance(schema_metadata, list):
        return schema_metadata                     # already a list
    return []


def _format_all_values(all_values) -> str:
    """
    Accept all_values as:
      A) dict {str: int}            → {"High": 420, "Medium": 310}
      B) list of {value, count}     → [{"value": "High", "count": 420}]
    Returns formatted string for prompt.
    """
    if isinstance(all_values, dict):
        return ", ".join(f"{v} ({c})" for v, c in all_values.items())
    if isinstance(all_values, list):
        return ", ".join(f"{item.get('value', item)} ({item.get('count', '')})" for item in all_values)
    return str(all_values)


def format_schema_for_prompt(schema_metadata) -> str:
    """
    Format schema context as clean readable text for LLM prompts.
    Accepts dict or list (see _normalise_schema_input).
    """
    tables = _normalise_schema_input(schema_metadata)
    lines = []

    for table in tables:
        lines.append(f"Table: {table['table_name']} ({table.get('row_count', '?')} rows)")
        for col in table.get("columns", []):
            name     = col["name"]
            col_type = col["type"]
            null_cnt = col.get("null_count", 0)
            null_note = f"nulls: {null_cnt}" if null_cnt > 0 else "no nulls"
            lines.append(f"  {name} ({col_type}) | {null_note}")

            if col.get("all_values"):
                lines.append(f"    values: {_format_all_values(col['all_values'])}")
            elif col.get("range"):
                r = col["range"]
                avg = f", avg {r['avg']}" if r.get("avg") is not None else ""
                lines.append(f"    range: {r['min']} → {r['max']}{avg}")
            elif col.get("sample_values"):
                lines.append(f"    samples: {', '.join(str(v) for v in col['sample_values'])}")

    # Include relationships if present
    if isinstance(schema_metadata, dict) and schema_metadata.get("relationships"):
        lines.append("\nRelationships (use for JOINs):")
        for r in schema_metadata["relationships"]:
            lines.append(
                f"  {r['from_table']}.{r['from_column']} → "
                f"{r['to_table']}.{r['to_column']} ({r.get('type', 'many_to_one')})"
            )

    return "\n".join(lines)


# LLM SQL Generator

class LLMSQLGenerator:

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def _build_prompt(
        self,
        user_query: str,
        schema_metadata,
        relationships: List[dict],
        reasoning: bool,
        plan=None,           # QueryPlan from intent.py (optional)
        vector_table: Optional[str] = None,
        vector_col:   Optional[str] = "_embedding",
        embed_dim:    Optional[int] = 384,
        column_map: Optional[Dict[str, str]] = None
    ) -> str:
        schema_block = format_schema_for_prompt(schema_metadata)

        # Build plan-aware hints
        plan_hints = []
        if plan:
            if plan.needs_aggregation:
                plan_hints.append("- This question requires aggregation (COUNT/SUM/AVG/GROUP BY). Always alias computed columns.")
            if plan.needs_ranking:
                plan_hints.append("- This question requires ranking. Use ORDER BY ... LIMIT or ROW_NUMBER() OVER(...).")
            if plan.needs_join:
                plan_hints.append("- This question requires joining multiple tables. Use the relationships listed in the schema.")
            if plan.needs_semantic and vector_table:
                plan_hints.append(
                    f"- This question requires semantic similarity search. "
                    f"Include: array_distance({vector_table}.{vector_col}, ?::FLOAT[{embed_dim}]) AS similarity_score "
                    f"in your SELECT, and ORDER BY similarity_score ASC."
                )
                plan_hints.append("  The query vector will be substituted at execution time via parameter binding.")
            if not plan_hints:
                plan_hints.append("- Answer the question directly. Use aggregation only if the question implies it.")

        hints_block = "\n".join(plan_hints) if plan_hints else ""

        print(plan)

        reasoning_block = ""
        if plan and plan.reasoning:
            reasoning_block = f"""
        IMPORTANT:
        Use this interpretation derived earlier:
        {plan.reasoning}
        """
        # column_hint_block = ""
        # if column_map:
        #     column_hint_block = f"""
        # COLUMN MAPPINGS (STRICT):
        # You MUST use ONLY these mappings when referring to concepts in the query.

        # {column_map}

        # RULES:
        # - Do NOT invent new columns.
        # - Do NOT reinterpret terms differently.
        # - If a term exists in this mapping, you MUST use the mapped column.
        # - If a term is not mapped, use only schema columns exactly as defined.
        # """
            
        if column_map:
            grounded_query = user_query + "\n\nResolved terms:\n" + str(column_map)
        else:
            grounded_query = user_query

        output_fmt = (
            'Return valid JSON: {"reasoning": "...", "sql_query": "..."}'
            if reasoning else
            "Return ONLY the SQL query. No markdown, no explanation."
        )

        return f"""
You are a senior data engineer writing DuckDB SQL.

{reasoning_block}


STRICT RULES:
- Use ONLY the tables and columns listed in the schema below.
- Do NOT invent tables, columns, or values.
- Do NOT create derived categories (like CASE WHEN, 'open', 'closed', etc.) unless explicitly required by the query.
- Do NOT infer business logic from column names.- Values in WHERE clauses MUST match EXACTLY the values shown in the schema.
- Use GROUP BY whenever you use an aggregate function.
- Always alias computed columns.

DuckDB-SPECIFIC SYNTAX:
- DATE subtraction: date_a - date_b → INTEGER days (not INTERVAL)
- Monthly grouping: strftime('%Y-%m', date_col)
- No ILIKE: use LOWER(col) LIKE LOWER('%value%')

QUERY GUIDANCE:
{hints_block}

SCHEMA:
{schema_block}

USER QUESTION:
{grounded_query}

{output_fmt}
""".strip()

    
    def _extract_known_tables(self, schema_metadata) -> set:
        tables = _normalise_schema_input(schema_metadata)
        return {t["table_name"] for t in tables if "table_name" in t}

    def _extract_known_columns(self, schema_metadata) -> set:
            tables = _normalise_schema_input(schema_metadata)
            cols = set()
            for t in tables:
                for c in t.get("columns", []):
                    cols.add(c["name"].lower())
            return cols
    
    def _basic_sql_validation(self, sql: str, schema_metadata) -> bool:
        known_tables = self._extract_known_tables(schema_metadata)
        known_columns = self._extract_known_columns(schema_metadata)

        # --- 1. Remove string literals completely ---
        sql_clean = re.sub(r"'[^']*'", "", sql)
        sql_clean = re.sub(r'"[^"]*"', "", sql_clean)

        sql_norm = sql_clean.lower()

        # # --- 2. Extract tokens ---
        # tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", sql_norm)

        # # --- 3. Full SQL keyword set (generic, not minimal) ---
        # SQL_KEYWORDS = {
        #     "select","from","where","group","by","order","limit","and","or","as","on",
        #     "join","left","right","inner","outer","full","cross",
        #     "count","sum","avg","min","max","distinct",
        #     "case","when","then","else","end",
        #     "is","null","not","in","like","between","exists",
        #     "having","union","all","with","over","partition","row_number",
        #     "asc","desc"
        # }

        # # --- 4. Validate only potential identifiers ---
        # unknown = []

        # for token in tokens:
        #     if (
        #         token not in SQL_KEYWORDS
        #         and token not in known_columns
        #         and token not in known_tables
        #     ):
        #         unknown.append(token)

        # --- 5. Allow small noise, block real hallucinations ---
        # if len(unknown) > 5:
        #     raise ValueError(f"Too many unknown identifiers: {unknown[:5]}")

        # --- 6. Ensure at least one known table is used ---
        if not any(t.lower() in sql_norm for t in known_tables):
            raise ValueError(f"SQL does not reference any known table. Known: {known_tables}")

        return True

    def generate_sql(
        self,
        user_query: str,
        schema_metadata,
        relationships: List[dict],
        reasoning: bool = False,
        plan=None,
        vector_table: Optional[str] = None,
        vector_col: str = "_embedding",
        embed_dim: int = 384,
        column_map=None
    ):
        """
        Intelligent SQL generation with self-correcting retries.
        """

        MAX_RETRIES = 3
        last_error = None
        last_sql = None

        for attempt in range(MAX_RETRIES):

            # Build prompt
            if attempt == 0:
                prompt = self._build_prompt(
                    user_query, schema_metadata, relationships, reasoning,
                    plan=plan,
                    vector_table=vector_table,
                    vector_col=vector_col,
                    embed_dim=embed_dim
                )
            else:
                # Correction prompt
                prompt = f"""
    The previous SQL query is incorrect.

    ERROR:
    {last_error}

    PREVIOUS SQL:
    {last_sql}

    Fix the SQL using the schema. Do NOT repeat the same mistake.

    STRICT RULES:
    - Use only valid columns
    - Do not invent fields
    - Follow column mappings strictly: {column_map}

    Return corrected SQL only.
    """

            raw = self.llm.generate(prompt).strip()

            # Cleanup
            raw = re.sub(r"^```(?:json|sql)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw).strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

            reasoning_text = ""
            if reasoning:
                try:
                    parsed = json.loads(raw)
                    sql = clean_sql(parsed["sql_query"])
                    reasoning_text = parsed.get("reasoning", "")
                except Exception:
                    m = re.search(r'"sql_query"\s*:\s*"([^"]+)"', raw)
                    sql = clean_sql(m.group(1)) if m else clean_sql(raw)
            else:
                sql = clean_sql(raw)

            last_sql = sql

            try:
                sql, _ = _fix_duckdb_date_arithmetic(sql)

                # Strong validation
                self._basic_sql_validation(sql, schema_metadata)
                SQLValidator.validate(sql)

                # SUCCESS
                if reasoning:
                    return {"sql_query": sql, "reasoning": reasoning_text}
                return sql

            except Exception as e:
                last_error = str(e)

                # Hard stop if final attempt
                if attempt == MAX_RETRIES - 1:
                    raise ValueError(f"SQL generation failed after retries: {last_error}")

                # else retry with correction

        raise ValueError("Unexpected SQL generation failure")