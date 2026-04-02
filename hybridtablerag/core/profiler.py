"""
core/profiler.py
================
Pure column-level statistics. No cleaning, no I/O.

Moved from: metadata/schema_profiler.py → profile_dataframe()
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return column-level statistics for a cleaned DataFrame.

    Each column entry contains:
        dtype         — pandas dtype string
        num_nulls     — count of null values
        pct_null      — percentage null (0–100)
        num_unique    — count of distinct non-null values
        sample_values — up to 5 non-null sample values
        is_multi_valued — True if values contain ';' or '|' separators
                          (used by the normalizer to identify bridge table candidates)
    """
    columns: Dict[str, Dict] = {}

    for col in df.columns:
        series      = df[col]
        non_null    = series.dropna()

        # Detect multi-valued strings (semicolon or pipe separated)
        is_multi = False
        if series.dtype == object and not non_null.empty:
            sample_vals = non_null.head(10).astype(str)
            has_semi = sample_vals.str.contains(';', regex=False).any()
            has_pipe = sample_vals.str.contains(r'\|', regex=True).any()
            # Only flag as multi-valued if separator appears in most values
            # (avoids flagging free text that happens to contain semicolons)
            if has_semi:
                ratio = sample_vals.str.contains(';', regex=False).mean()
                is_multi = ratio >= 0.5
            elif has_pipe:
                ratio = sample_vals.str.contains(r'\|', regex=True).mean()
                is_multi = ratio >= 0.5

        columns[col] = {
            'dtype':          str(series.dtype),
            'num_nulls':      int(series.isna().sum()),
            'pct_null':       round(float(series.isna().mean() * 100), 2),
            'num_unique':     int(series.nunique(dropna=True)),
            'sample_values':  non_null.head(5).tolist(),
            'is_multi_valued': is_multi,
        }

    return {
        'num_rows':    df.shape[0],
        'num_columns': df.shape[1],
        'columns':     columns,
    }


def profile_summary(profile: Dict[str, Any]) -> str:
    """
    Human-readable summary of a profile dict.
    Used in BTS logs and debug output.
    """
    lines = [
        f"Rows: {profile['num_rows']} | Columns: {profile['num_columns']}",
    ]
    multi_valued = [
        col for col, stats in profile['columns'].items()
        if stats.get('is_multi_valued')
    ]
    high_null = [
        col for col, stats in profile['columns'].items()
        if stats['pct_null'] >= 50
    ]
    if multi_valued:
        lines.append(f"Multi-valued columns (bridge table candidates): {multi_valued}")
    if high_null:
        lines.append(f"High null columns (>=50%): {high_null}")
    return '\n'.join(lines)