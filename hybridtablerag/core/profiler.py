"""
core/profiler.py
================
Pure column-level statistics. No cleaning, no I/O.

Moved from: metadata/schema_profiler.py → profile_dataframe()
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd



def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return column-level statistics with robust multi-value detection.
    
    Detects: ; | , separators, JSON lists [...], Python lists [...], and slash-separated values.
    """
    columns = {}

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        is_multi = False
        multi_type = None

        # Check for ANY string-like dtype (object, string, str)
        is_string_dtype = (
            series.dtype == 'object' or 
            series.dtype == 'string' or 
            str(series.dtype).lower() in ('object', 'string', 'str') or
            pd.api.types.is_string_dtype(series)
        )

        if is_string_dtype and not non_null.empty:
            sample = non_null.head(20).astype(str)
            
            # 1. JSON/List structures (High confidence) 
            # Matches: ["a", "b"] or [{"k":"v"}] or ['a', 'b']
            starts_with_bracket = sample.str.strip().str.startswith('[').mean()
            if starts_with_bracket > 0.3:  # Lowered threshold for inclusivity
                is_multi = True
                multi_type = 'json_list'

            # 2. Pipe separator | (High confidence)
            elif sample.str.contains('|', regex=False).mean() > 0.3:
                is_multi = True
                multi_type = 'pipe'

            # 3. Semicolon separator ; (High confidence) 
            elif sample.str.contains(';', regex=False).mean() > 0.3:
                is_multi = True
                multi_type = 'semicolon'

            # 4. Comma-separated lists (Medium confidence) 
            # Heuristic: multiple commas + high unique count = likely list, not prose
            comma_ratio = sample.str.count(',').mean()
            if comma_ratio > 1.0 and series.nunique() > 10:  # Lowered threshold
                is_multi = True
                multi_type = 'comma'

            # 5. Forward slash / (Common in categories: "A/B/C") 
            elif sample.str.contains('/', regex=False).mean() > 0.4:
                is_multi = True
                multi_type = 'slash'

        columns[col] = {
            'dtype': str(series.dtype),
            'num_nulls': int(series.isna().sum()),
            'pct_null': round(float(series.isna().mean() * 100), 2),
            'num_unique': int(series.nunique(dropna=True)),
            'sample_values': non_null.head(5).tolist(),
            'is_multi_valued': is_multi,
            'multi_val_type': multi_type,  # NEW: helps normalizer choose split strategy
        }

    return {
        'num_rows': df.shape[0],
        'num_columns': df.shape[1],
        'columns': columns,
    }


def profile_summary(profile: Dict[str, Any]) -> str:
    """Human-readable summary for logs/debug output."""
    lines = [f"Rows: {profile['num_rows']} | Columns: {profile['num_columns']}"]
    
    multi_valued = [
        col for col, stats in profile['columns'].items()
        if stats.get('is_multi_valued')
    ]
    if multi_valued:
        details = [
            f"{col}({profile['columns'][col]['multi_val_type']})" 
            for col in multi_valued
        ]
        lines.append(f"🔗 Multi-valued: {', '.join(details)}")
    
    return '\n'.join(lines)