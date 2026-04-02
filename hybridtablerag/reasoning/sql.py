"""
reasoning/sql.py
===========================

Improved version:
- Fixes regex bug in date arithmetic
- Adds multi-table JOIN awareness
- Keeps backward compatibility with single-table schema
- Cleaner validation + safer LLM output handling
"""

import json
import re
from typing import Any, Dict, List

from hybridtablerag.llm.base import BaseLLM


# ──────────────────────────────────────────────────────────────
# Safety allow-list
# ──────────────────────────────────────────────────────────────

FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT",
    "ALTER", "TRUNCATE", "CREATE",
]


class SQLValidator:
    @staticmethod
    def validate(sql: str) -> None:
        sql_upper = sql.upper()
        if not sql_upper.strip().startswith("SELECT"):
            raise ValueError("Only SELECT statements are allowed.")
        for keyword in FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{keyword}\b", sql_upper):
                raise ValueError(f"Forbidden SQL keyword detected: {keyword}")


# ──────────────────────────────────────────────────────────────
# SQL cleanup
# ──────────────────────────────────────────────────────────────

def clean_sql(raw: str) -> str:
    sql = raw.strip()
    sql = re.sub(r"^```(?:sql)?\s*\n?", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\n?```\s*$", "", sql).strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    return sql


# ──────────────────────────────────────────────────────────────
# DuckDB fixes
# ──────────────────────────────────────────────────────────────

def _fix_duckdb_date_arithmetic(sql: str) -> tuple[str, list[str]]:
    fixes: list[str] = []

    interval_cmp = re.compile(
        r'(CAST\s*\([^)]+\)|\b\w+(?:\.\w+)?)\s*'
        r'-\s*'
        r'(CAST\s*\([^)]+\)|\b\w+(?:\.\w+)?)\s*'
        r'([<>=!]+)\s*'
        r"INTERVAL\s*['\"](\d+)\s*(\w+)['\"]",   # ✅ FIXED (removed space before \d+)
        re.IGNORECASE,
    )

    def _rewrite(m):
        lhs  = m.group(1)
        rhs  = m.group(2)
        op   = m.group(3)
        n    = m.group(4)
        unit = m.group(5).lower().rstrip('s')

        if unit == 'day':
            fixes.append("Converted DATE subtraction from INTERVAL to integer")
            return f"{lhs} - {rhs} {op} {n}"

        fixes.append(f"Converted INTERVAL to datediff({unit})")
        return f"datediff('{unit}', {rhs}, {lhs}) {op} {n}"

    return interval_cmp.sub(_rewrite, sql), fixes


# ──────────────────────────────────────────────────────────────
# Schema formatter (supports multi-table)
# ──────────────────────────────────────────────────────────────

def _format_schema_for_prompt(schema_metadata: list) -> str:
    lines = []

    for table in schema_metadata:
        lines.append(
            f"Table: {table['table_name']} ({table.get('row_count', '?')} rows)"
        )

        for col in table.get("columns", []):
            name      = col["name"]
            col_type  = col["type"]
            null_cnt  = col.get("null_count", 0)
            null_note = f"nulls: {null_cnt}" if null_cnt > 0 else "no nulls"

            lines.append(f"  {name} ({col_type}) | {null_note}")

            if col.get("all_values"):
                val_str = ", ".join(
                    f"{v['value']} ({v['count']})"
                    for v in col["all_values"]
                )
                lines.append(f"    values: {val_str}")

            elif col.get("range"):
                r = col["range"]
                lines.append(f"    range: {r['min']} → {r['max']}")

            elif col.get("sample_values"):
                lines.append(f"    samples: {', '.join(col['sample_values'])}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# LLM SQL Generator
# ──────────────────────────────────────────────────────────────

class LLMSQLGenerator:

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        user_query: str,
        schema_metadata,
        relationships,
        reasoning: bool = False,
    ) -> str:

        if reasoning:
            output_instruction = """
Return valid JSON:
{
  "reasoning": "...",
  "sql_query": "..."
}
"""
        else:
            output_instruction = "Return ONLY SQL."

        schema_block = _format_schema_for_prompt(schema_metadata)

        prompt = f"""
You are a senior data engineer writing DuckDB SQL.

STRICT RULES:
- Use ONLY provided tables/columns
- Do NOT hallucinate schema
- Use correct joins if multiple tables exist
- Always alias aggregations

User Question:
{user_query}

Schema:
{schema_block}
"""

        if relationships:
            rel_lines = []
            for r in relationships:
                rel_lines.append(
                    f"{r['from_table']}.{r['from_column']} = "
                    f"{r['to_table']}.{r['to_column']}"
                )

            prompt += "\nAvailable JOIN paths:\n" + "\n".join(rel_lines)
            prompt += "\nUse these joins when needed.\n"

        prompt += f"\n{output_instruction}\n"

        return prompt

    # ------------------------------------------------------------------
    def _basic_sql_validation(
        self,
        sql: str,
        schema_metadata: List[Dict[str, Any]],
    ) -> bool:

        known_tables = {
            t["table_name"] for t in schema_metadata if "table_name" in t
        }

        sql_norm = re.sub(r"[`\"]", "", sql.lower())

        if not any(t.lower() in sql_norm for t in known_tables):
            raise ValueError(
                f"SQL does not reference known tables: {known_tables}"
            )

        return True

    # ------------------------------------------------------------------
    def generate_sql(
        self,
        user_query: str,
        schema_metadata,
        relationships,
        reasoning: bool = False,
    ):

        prompt = self._build_prompt(
            user_query,
            schema_metadata,
            relationships,
            reasoning,
        )

        raw = self.llm.generate(prompt).strip()
        raw = raw.replace("```", "").strip()

        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

        if reasoning:
            parsed = json.loads(raw)
            sql = clean_sql(parsed["sql_query"])
            sql, _ = _fix_duckdb_date_arithmetic(sql)
            self._basic_sql_validation(sql, schema_metadata)
            return {"sql_query": sql, "reasoning": parsed["reasoning"]}

        sql = clean_sql(raw)
        sql, _ = _fix_duckdb_date_arithmetic(sql)
        self._basic_sql_validation(sql, schema_metadata)

        return sql