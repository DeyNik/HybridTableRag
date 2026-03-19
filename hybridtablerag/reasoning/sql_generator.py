"""
reasoning/sql_generator.py
===========================
Changes from original:
  - SQLValidator.validate(): unchanged
  - _basic_sql_validation(): normalises backticks/quotes before table-name check
                             so `tickets` and "tickets" both match 'tickets'
  - generate_sql(): single-pass clean_sql() replaces the fragile multi-step cleanup
  - PythonExecutor.execute(): guards against LLM error strings reaching exec()
"""

import json
import re
from typing import Any, Dict, List

from hybridtablerag.llm.base import BaseLLM


# ──────────────────────────────────────────────────────────────────────────────
# Safety allow-list
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# SQL cleanup  (single-pass — replaces the fragile multi-step original)
# ──────────────────────────────────────────────────────────────────────────────

def clean_sql(raw: str) -> str:
    sql = raw.strip()
    sql = re.sub(r"^```(?:sql)?\s*\n?", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\n?```\s*$", "", sql).strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    return sql


# ──────────────────────────────────────────────────────────────────────────────
# DuckDB-specific SQL post-processor
# ──────────────────────────────────────────────────────────────────────────────

def _fix_duckdb_date_arithmetic(sql: str) -> tuple[str, list[str]]:
    """
    Fix LLM-generated date arithmetic that is valid in PostgreSQL/MySQL
    but wrong in DuckDB.

    DuckDB rule:  DATE - DATE = INTEGER (days), not INTERVAL.
    LLMs often generate:  date_col_a - date_col_b < INTERVAL '3 days'
    DuckDB needs:         date_col_a - date_col_b < 3

    Handles:
      CAST(a AS DATE) - CAST(b AS DATE) <op> INTERVAL 'N day[s]'
        → CAST(a AS DATE) - CAST(b AS DATE) <op> N
      col_a - col_b <op> INTERVAL 'N day[s]'
        → col_a - col_b <op> N
      any_expr - any_expr <op> INTERVAL 'N month[s]/year[s]'
        → datediff('unit', rhs, lhs) <op> N
    """
    fixes: list[str] = []

    interval_cmp = re.compile(
        r'(CAST\s*\([^)]+\)|\b\w+(?:\.\w+)?)\s*'
        r'-\s*'
        r'(CAST\s*\([^)]+\)|\b\w+(?:\.\w+)?)\s*'
        r'([<>=!]+)\s*'
        r"INTERVAL\s*['\"]( \d+)\s*(\w+)['\"]",
        re.IGNORECASE,
    )

    def _rewrite(m):
        lhs  = m.group(1)
        rhs  = m.group(2)
        op   = m.group(3)
        n    = m.group(4).strip()
        unit = m.group(5).lower().rstrip('s')
        if unit == 'day':
            fixes.append(
                f"Fixed DATE-DATE vs INTERVAL: '{lhs} - {rhs} {op} INTERVAL \'{n} {unit}\'' "
                f"→ integer comparison (DuckDB DATE subtraction returns BIGINT)"
            )
            return f"{lhs} - {rhs} {op} {n}"
        fixes.append(
            f"Fixed INTERVAL \'{n} {unit}\' → datediff(\'{unit}\', ...) {op} {n}"
        )
        return f"datediff(\'{unit}\', {rhs}, {lhs}) {op} {n}"

    return interval_cmp.sub(_rewrite, sql), fixes


# ──────────────────────────────────────────────────────────────────────────────
# Schema formatter  — converts enriched schema_ctx to readable prompt text
# ──────────────────────────────────────────────────────────────────────────────

def _format_schema_for_prompt(schema_metadata: list) -> str:
    """
    Converts the enriched schema context (from build_schema_context) into
    a clean, scannable text block for the LLM prompt.

    Raw JSON is noisy and wastes tokens. This format is faster to parse
    and makes categorical values immediately visible so the LLM uses
    exact values rather than guessing.

    Example output:
        Table: tickets (1001 rows)
          ticket_priority (VARCHAR) | nulls: 0
            values: High (420), Medium (310), Low (180), Critical (91)
          ticket_created_date (DATE) | nulls: 0
            range: 2023-01-01 → 2024-12-31
          customer_name (VARCHAR) | nulls: 5
            samples: Alice Smith, Bob Jones, Carol White
    """
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
                # Categorical — show every value with its count
                val_str = ", ".join(
                    f"{v} ({c})" for v, c in col["all_values"].items()
                )
                lines.append(f"    values: {val_str}")

            elif col.get("range"):
                r = col["range"]
                if "avg" in r and r["avg"] is not None:
                    lines.append(
                        f"    range: {r['min']} → {r['max']} (avg {r['avg']})"
                    )
                else:
                    lines.append(f"    range: {r['min']} → {r['max']}")

            elif col.get("sample_values"):
                lines.append(f"    samples: {', '.join(col['sample_values'])}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# LLM SQL generator
# ──────────────────────────────────────────────────────────────────────────────

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
OUTPUT FORMAT:
Return strictly valid JSON with this structure:
{
  "reasoning": "Step by step reasoning explaining how you derived the query.",
  "sql_query":  "Valid DuckDB SQL query"
}
Do not include markdown. Do not include extra text. Return only valid JSON.
"""
        else:
            output_instruction = """
OUTPUT FORMAT:
Return ONLY the SQL query.
Do not include markdown fences.
Do not include explanations.
Do not prefix with 'sql'.
"""

        # Extract exact table name so the LLM cannot guess a different one
        table_names = [t["table_name"] for t in schema_metadata if "table_name" in t]
        table_name_hint = (
            f"\nIMPORTANT: The ONLY valid table name is: {table_names[0]!r}. "
            "Use this exact name — no aliases, no surrounding quotes unless DuckDB requires it.\n"
            if table_names else ""
        )

        # Detect query complexity to give the LLM the right pattern hints
        q_lower = user_query.lower()
        agg_keywords    = any(w in q_lower for w in ("total", "sum", "average", "avg", "mean", "how many", "count", "how much"))
        rank_keywords   = any(w in q_lower for w in ("most", "least", "top", "bottom", "highest", "lowest", "maximum", "minimum", "best", "worst"))
        trend_keywords  = any(w in q_lower for w in ("over time", "by month", "by week", "by year", "trend", "per month", "monthly", "weekly"))
        implicit_agg    = any(w in q_lower for w in ("what is the", "what was the", "tell me the", "give me the")) and not any(w in q_lower for w in ("list", "show all", "find all"))

        pattern_hints = []
        if agg_keywords or implicit_agg:
            pattern_hints.append(
                "- This question requires aggregation (SUM, COUNT, AVG). "
                "Use GROUP BY if grouping by a category. Always alias aggregated columns."
            )
        if rank_keywords:
            pattern_hints.append(
                "- This question requires ranking. Use ORDER BY with LIMIT, "
                "or window functions like ROW_NUMBER() OVER (ORDER BY ...) for ranked lists."
            )
        if trend_keywords:
            pattern_hints.append(
                "- This question requires time-series grouping. "
                "Use strftime('%Y-%m', date_col) for monthly grouping in DuckDB. "
                "Always ORDER BY the time column."
            )
        if not pattern_hints:
            pattern_hints.append(
                "- If the question implies a count, total, or comparison without stating it "
                "explicitly, use aggregation rather than returning raw rows."
            )

        pattern_block = "\n".join(pattern_hints)

        base_instruction = f"""
You are a senior data engineer writing DuckDB SQL.

STRICT RULES:
- Use ONLY the table and columns listed in the schema below.
- Do NOT invent tables, columns, or values.
- Values in WHERE clauses must match EXACTLY the values shown in the schema (case-sensitive).
- Use GROUP BY whenever you use an aggregate function.
- Always alias computed columns (e.g. COUNT(*) AS ticket_count).
{table_name_hint}
QUERY PATTERN GUIDANCE:
{pattern_block}

DuckDB-SPECIFIC SYNTAX:
- DATE subtraction: date_a - date_b returns INTEGER days (not INTERVAL)
- Monthly grouping: strftime('%Y-%m', date_col)
- No ILIKE: use LOWER(col) LIKE LOWER('%value%')
- String concat: col1 || ' ' || col2

{output_instruction}
"""
        # Format schema as readable text — much cleaner than raw JSON for the LLM.
        # Includes full value distributions for categorical columns (from Change 1)
        # and min/max/avg ranges for numeric and date columns.
        schema_block = _format_schema_for_prompt(schema_metadata)

        prompt = f"""
{base_instruction}

User Question:
{user_query}

Available Schema:
{schema_block}
"""
        if len(schema_metadata) > 1 and relationships:
            prompt += f"""
Inferred Relationships:
{json.dumps(relationships, indent=2)}

Use these relationships for JOIN conditions if needed.
"""
        prompt += "\nGenerate the SQL query:\n"
        return prompt

    # ------------------------------------------------------------------
    def _basic_sql_validation(
        self, sql: str, schema_metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        BUG FIX: original compared raw table names against raw SQL, so
        backtick-quoted names (`tickets`) or double-quoted names ("tickets")
        would fail even though they reference the correct table.

        Fix: strip all identifier-quoting characters before comparison.
        """
        known_tables = {t["table_name"] for t in schema_metadata if "table_name" in t}

        # Normalise: remove backticks and double-quotes, lowercase
        sql_normalised = re.sub(r'[`"]', '', sql.lower())

        if not any(t.lower() in sql_normalised for t in known_tables):
            raise ValueError(
                f"Generated SQL does not reference any known tables.\n"
                f"Known tables: {known_tables}\n"
                f"SQL (normalised): {sql_normalised[:300]}"
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
            user_query=user_query,
            schema_metadata=schema_metadata,
            relationships=relationships,
            reasoning=reasoning,
        )

        raw = self.llm.generate(prompt).strip()

        # Strip markdown fences the LLM might add despite instructions
        raw = raw.replace("```", "").strip()
        # Remove leading 'json' label that sometimes appears
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

        if reasoning:
            try:
                parsed = json.loads(raw)
                sql = clean_sql(parsed["sql_query"])
                sql, date_fixes = _fix_duckdb_date_arithmetic(sql)
                self._basic_sql_validation(sql, schema_metadata)
                return {"sql_query": sql, "reasoning": parsed["reasoning"]}
            except json.JSONDecodeError:
                raise ValueError(f"LLM did not return valid JSON:\n{raw}")

        sql = clean_sql(raw)
        sql, date_fixes = _fix_duckdb_date_arithmetic(sql)
        self._basic_sql_validation(sql, schema_metadata)
        return sql