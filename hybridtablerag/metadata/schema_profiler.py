"""
data_cleaner.py
===============
Robust DataFrame cleaning: normalise columns, clean nulls, flatten JSON,
parse mixed-format dates, infer numerics, drop duplicates.
"""

import json
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column name normalisation
# ---------------------------------------------------------------------------

def normalize_column_name(name: str) -> str:
    """Lowercase, strip, replace whitespace / non-alphanumeric with underscore."""
    name = str(name).strip().lower()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^0-9a-z_]', '', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name or 'unnamed'


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict, prefix: str, sep: str = '_') -> dict:
    """Recursively flatten a nested dict into a flat dict with compound keys."""
    out = {}
    for k, v in d.items():
        new_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, new_key, sep))
        else:
            out[new_key] = v
    return out


def _parse_json(val):
    """Try to parse a value as JSON; return the original value if it is not."""
    if isinstance(val, str):
        stripped = val.strip()
        if stripped.startswith(('{', '[')):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return val


def _is_json_column(series: pd.Series) -> bool:
    """
    Return True if any of the first 10 non-null values look like JSON.
    Checking only the first value misses columns where row 0 is NaN.
    """
    for val in series.dropna().head(10):
        if isinstance(val, str) and val.strip().startswith(('{', '[')):
            return True
    return False


def flatten_json_column(
    df: pd.DataFrame,
    col: str,
    prefix: str = None,
    log: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Flatten a JSON column into multiple new columns, preserving row count
    and row order exactly.

    Handles:
      - dict              -> one column per key (recursive / nested dicts)
      - list of dicts     -> one column per key; multiple values joined with '; '
      - list of scalars   -> single column; values joined with '; '
      - scalar            -> single column with the value
      - null/unparseable  -> all new columns get None for that row
    """
    log = log or []
    prefix = prefix or col

    row_records: List[Optional[dict]] = []
    for val in df[col]:
        if val is None:
            row_records.append(None)
            continue
        try:
            if pd.isna(val):
                row_records.append(None)
                continue
        except (TypeError, ValueError):
            pass

        parsed = _parse_json(val)

        if isinstance(parsed, dict):
            row_records.append(_flatten_dict(parsed, prefix))
        elif isinstance(parsed, list):
            if not parsed:
                row_records.append(None)
            elif all(isinstance(x, dict) for x in parsed):
                keys = set().union(*(d.keys() for d in parsed))
                merged = {
                    f"{prefix}_{k}": '; '.join(str(d.get(k, '')) for d in parsed)
                    for k in keys
                }
                row_records.append(merged)
            else:
                row_records.append({prefix: '; '.join(str(x) for x in parsed)})
        else:
            row_records.append({prefix: parsed})

    seen: set = set()
    all_keys: List[str] = []
    for rec in row_records:
        if rec:
            for k in rec:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

    for k in all_keys:
        df[k] = [rec.get(k) if rec else None for rec in row_records]
        log.append(f"Flattened '{col}' -> '{k}'")

    df = df.drop(columns=[col])
    log.append(f"Dropped original column '{col}' after flattening")
    return df


# ---------------------------------------------------------------------------
# Null / na-like cleaning
# ---------------------------------------------------------------------------

_NULL_STRINGS = frozenset({
    '', 'na', 'n/a', 'null', 'none', '-', 'nan', 'nil',
    '#n/a', '#null!', 'missing', 'unknown',
})


def clean_column_value(val):
    """
    Normalise a single cell value:
      - True NaN / None  -> None
      - String na-likes  -> None
      - Strip whitespace and HTML tags from strings
    """
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(val, str):
        val = re.sub(r'<[^>]+>', '', val)
        val = val.strip()
        if val.lower() in _NULL_STRINGS:
            return None

    return val


# ---------------------------------------------------------------------------
# Date parsing - element-wise to handle mixed formats in the same column
# ---------------------------------------------------------------------------

def _parse_date_elementwise(series: pd.Series) -> pd.Series:
    """
    Parse each cell independently so that a column with mixed date formats
    (e.g. "May 03, 2024", "31/03/25", "01/08/24", "2025-04-28") is handled
    correctly.

    Why not pd.to_datetime(series)?
    Pandas infers ONE format from the first non-null value and silently
    returns NaT for every row that uses a different format.
    Parsing row-by-row avoids this completely.
    """
    def _try_one(val):
        if val is None:
            return None
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        for dayfirst in (True, False):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    return pd.to_datetime(val, errors='raise', dayfirst=dayfirst)
                except Exception:
                    continue
        return None

    return series.apply(_try_one)


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
    log: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean a DataFrame:

      1. Flatten MultiIndex columns
      2. Normalise column names
      3. Drop duplicate columns (keep first)
      4. Clean cell values (nulls, HTML, whitespace)
      5. Drop duplicate rows
      6. Flatten JSON columns (row-aligned, handles nested dicts)
      7. Parse & normalise date columns (element-wise, handles mixed formats)
      8. Infer numeric columns

    Returns (cleaned_df, log).
    """
    log = log or []

    # 1. Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            '_'.join(str(i) for i in col if str(i) != '').strip()
            for col in df.columns
        ]
        log.append('Flattened MultiIndex columns')

    # 2. Normalise column names
    original_cols = df.columns.tolist()
    df.columns = [normalize_column_name(c) for c in df.columns]
    renamed = {o: n for o, n in zip(original_cols, df.columns) if o != n}
    if renamed:
        log.append(f'Normalized column names: {renamed}')

    # 3a. Drop duplicate column names (keep first occurrence)
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        df = df.loc[:, ~df.columns.duplicated()]
        log.append(f'Dropped duplicate column names: {dupes}')

    # 3b. Drop columns whose VALUES are identical to an already-seen column
    seen_fingerprints: dict = {}   # fingerprint -> first column name
    content_dupes: List[str] = []
    for col in df.columns.tolist():
        fingerprint = tuple(df[col].astype(str).tolist())
        if fingerprint in seen_fingerprints:
            content_dupes.append((col, seen_fingerprints[fingerprint]))
        else:
            seen_fingerprints[fingerprint] = col
    if content_dupes:
        drop_cols = [c for c, _ in content_dupes]
        df = df.drop(columns=drop_cols)
        for dup, orig in content_dupes:
            log.append(f"Dropped content-duplicate column '{dup}' (identical to '{orig}')")

    # 4. Cell-level cleaning
    for col in df.columns:
        df[col] = df[col].apply(clean_column_value)
    log.append('Applied cell cleaning (nulls, HTML, whitespace)')

    # 5. Drop duplicate rows
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        log.append(f'Dropped {n_dropped} duplicate row(s)')

    # 6. Flatten JSON columns
    for col in df.columns.tolist():
        if _is_json_column(df[col]):
            df = flatten_json_column(df, col, prefix=col, log=log)

    # 7 & 8. Date and numeric inference
    date_pattern = re.compile(
        r'(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4})'
        r'|(\d{1,2}\s?[A-Za-z]{3,9}\s?\d{2,4})'
        r'|([A-Za-z]{3,9}\s?\d{1,2},?\s?\d{2,4})'
    )

    for col in df.columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue

        sample_str = non_null.astype(str).head(20)

        # Date detection
        if sample_str.str.match(date_pattern).any():
            try:
                parsed = _parse_date_elementwise(df[col])
                recovered = sum(1 for v in parsed if v is not None and pd.notna(v))
                if recovered > 0:
                    df[col] = [
                        v.strftime('%Y-%m-%d') if (v is not None and pd.notna(v)) else None
                        for v in parsed
                    ]
                    log.append(
                        f"Normalised dates in '{col}' -> YYYY-MM-DD "
                        f"({recovered}/{len(df)} values parsed)"
                    )
            except Exception as e:
                log.append(f"Could not parse dates in '{col}': {e}")
            continue

        # Numeric detection (>=90% of samples must look numeric)
        numeric_matches = sample_str.str.match(r'^-?\d*\.?\d+$').sum()
        if numeric_matches >= max(1, len(sample_str) * 0.9):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                log.append(f"Inferred numeric for column '{col}'")
            except Exception as e:
                log.append(f"Could not convert '{col}' to numeric: {e}")

    return df, log


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_dataframe(df: pd.DataFrame) -> Dict:
    """Return column-level statistics for a cleaned DataFrame."""
    return {
        'num_rows': df.shape[0],
        'num_columns': df.shape[1],
        'columns': {
            col: {
                'dtype': str(df[col].dtype),
                'num_nulls': int(df[col].isna().sum()),
                'pct_null': round(float(df[col].isna().mean() * 100), 2),
                'num_unique': int(df[col].nunique(dropna=True)),
                'sample_values': df[col].dropna().head(5).tolist(),
            }
            for col in df.columns
        },
    }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def clean_and_profile(
    df: pd.DataFrame,
    log: Optional[List[str]] = None,
) -> Dict:
    """
    Clean then profile a DataFrame.

    Returns a dict with:
      - 'cleaned_df': the cleaned DataFrame
      - 'profile':    column-level statistics
      - 'log':        list of actions taken / warnings
    """
    df_cleaned, log = clean_dataframe(df.copy(), log=log or [])
    return {
        'cleaned_df': df_cleaned,
        'profile': profile_dataframe(df_cleaned),
        'log': log,
    }