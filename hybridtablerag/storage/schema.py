"""
storage/schema.py
=================
Builds enriched schema context for LLM prompts.
"""

from typing import Any, Dict, List
import pandas as pd


CATEGORICAL_THRESHOLD = 25


def _safe_val(v):
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
    except Exception:
        pass

    return v


def build_schema_context(conn, table_name: str, bts_log: List[str]) -> Dict[str, Any]:
    """
    Build schema context for ONE table.
    """

    # --- row count ---
    row_count = conn.execute(
        f"SELECT COUNT(*) FROM {table_name}"
    ).fetchone()[0]

    # --- schema ---
    schema_rows = conn.execute(
        f"DESCRIBE {table_name}"
    ).fetchall()

    columns = [row[0] for row in schema_rows]

    # --- null counts (batched) ---
    null_exprs = [
        f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS {col}_nulls"
        for col in columns
    ]
    null_query = f"SELECT {', '.join(null_exprs)} FROM {table_name}"
    null_counts = conn.execute(null_query).fetchone()

    # --- distinct counts (batched) ---
    distinct_exprs = [
        f"COUNT(DISTINCT {col}) AS {col}_distinct"
        for col in columns
    ]
    distinct_query = f"SELECT {', '.join(distinct_exprs)} FROM {table_name}"
    distinct_counts = conn.execute(distinct_query).fetchone()

    # --- sample ---
    sample_df = conn.execute(
        f"SELECT * FROM {table_name} LIMIT 3"
    ).fetchdf()

    col_contexts = []

    for i, col in enumerate(columns):
        col_type = schema_rows[i][1]
        null_count = null_counts[i]
        distinct_count = distinct_counts[i]

        col_ctx = {
            "name": col,
            "type": col_type,
            "null_count": int(null_count),
            "distinct": int(distinct_count),
        }

        col_upper = col_type.upper()

        # --- categorical ---
        if distinct_count <= CATEGORICAL_THRESHOLD:
            try:
                dist_df = conn.execute(f"""
                    SELECT {col}, COUNT(*) as cnt
                    FROM {table_name}
                    GROUP BY {col}
                    ORDER BY cnt DESC
                """).fetchdf()

                col_ctx["all_values"] = [
                    {"value": _safe_val(row[col]), "count": int(row["cnt"])}
                    for _, row in dist_df.iterrows()
                ]

            except Exception:
                pass

        # --- numeric ---
        elif any(t in col_upper for t in ["INT", "DOUBLE", "FLOAT", "DECIMAL", "BIGINT"]):
            try:
                stats = conn.execute(f"""
                    SELECT MIN({col}), MAX({col}), AVG({col})
                    FROM {table_name}
                """).fetchone()

                col_ctx["range"] = {
                    "min": _safe_val(stats[0]),
                    "max": _safe_val(stats[1]),
                    "avg": _safe_val(stats[2]),
                }
            except Exception:
                pass

        # --- date ---
        elif any(t in col_upper for t in ["DATE", "TIMESTAMP"]):
            try:
                stats = conn.execute(f"""
                    SELECT MIN({col}), MAX({col})
                    FROM {table_name}
                """).fetchone()

                col_ctx["range"] = {
                    "min": _safe_val(stats[0]),
                    "max": _safe_val(stats[1]),
                }
            except Exception:
                pass

        # --- fallback sample ---
        else:
            if col in sample_df.columns:
                col_ctx["sample_values"] = [
                    _safe_val(v) for v in sample_df[col].dropna().tolist()
                ]

        col_contexts.append(col_ctx)

    bts_log.append(f"Built schema context for table: {table_name}")

    return {
        "table_name": table_name,
        "row_count": int(row_count),
        "columns": col_contexts,
    }


def build_multi_table_schema_context(
    conn,
    table_names: List[str],
    relationships: List[dict],
    bts_log: List[str],
) -> Dict[str, Any]:
    """
    Multi-table schema context.
    """

    tables_ctx = []

    for t in table_names:
        ctx = build_schema_context(conn, t, bts_log)
        tables_ctx.append(ctx)

    return {
        "tables": tables_ctx,
        "relationships": relationships or [],
    }


def format_schema_for_prompt(schema_context: Dict[str, Any]) -> str:
    """
    Convert schema context to LLM-friendly text.
    """

    lines = []

    # --- MULTI TABLE ---
    if "tables" in schema_context:
        for table in schema_context["tables"]:
            lines.append(f"Table: {table['table_name']} ({table['row_count']} rows)")

            for col in table["columns"]:
                line = f"  {col['name']} ({col['type']})"

                if col["null_count"] == 0:
                    line += " | no nulls"
                else:
                    line += f" | {col['null_count']} nulls"

                lines.append(line)

                # categorical
                if "all_values" in col:
                    vals = ", ".join([
                        f"{v['value']} ({v['count']})"
                        for v in col["all_values"]
                    ])
                    lines.append(f"    values: {vals}")

                # range
                elif "range" in col:
                    r = col["range"]
                    if "avg" in r:
                        lines.append(
                            f"    range: {r['min']} → {r['max']} (avg {r['avg']})"
                        )
                    else:
                        lines.append(
                            f"    range: {r['min']} → {r['max']}"
                        )

                # sample
                elif "sample_values" in col:
                    vals = ", ".join(map(str, col["sample_values"]))
                    lines.append(f"    sample: {vals}")

        # relationships
        if schema_context.get("relationships"):
            lines.append("Relationships:")
            for r in schema_context["relationships"]:
                lines.append(
                    f"  {r['from_table']}.{r['from_column']} → "
                    f"{r['to_table']}.{r['to_column']} ({r['type']})"
                )

    # --- SINGLE TABLE ---
    else:
        table = schema_context

        lines.append(f"Table: {table['table_name']} ({table['row_count']} rows)")

        for col in table["columns"]:
            line = f"  {col['name']} ({col['type']})"

            if col["null_count"] == 0:
                line += " | no nulls"
            else:
                line += f" | {col['null_count']} nulls"

            lines.append(line)

            if "all_values" in col:
                vals = ", ".join([
                    f"{v['value']} ({v['count']})"
                    for v in col["all_values"]
                ])
                lines.append(f"    values: {vals}")

            elif "range" in col:
                r = col["range"]
                if "avg" in r:
                    lines.append(
                        f"    range: {r['min']} → {r['max']} (avg {r['avg']})"
                    )
                else:
                    lines.append(
                        f"    range: {r['min']} → {r['max']}"
                    )

            elif "sample_values" in col:
                vals = ", ".join(map(str, col["sample_values"]))
                lines.append(f"    sample: {vals}")

    return "\n".join(lines)