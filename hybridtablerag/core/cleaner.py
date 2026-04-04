"""
core/cleaner.py
===============
DataFrame cleaning pipeline.

SIMPLIFIED: Manual header_rows only (auto-detect commented out for now)
Preserved: All cell cleaning logic (JSON, dates, nulls, deduplication)
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Dict, List, Optional, Tuple
import os
import hashlib
import pandas as pd


# Column name normalisation

def normalize_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^0-9a-z_]', '', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name or 'unnamed'


# Multi-level header merging (manual only)

def _merge_multi_level_headers(df: pd.DataFrame, header_rows: list[int]) -> pd.DataFrame:
    """
    Combine multi-level headers into single-level column names.
    Called ONLY when user explicitly passes header_rows=[0,1], etc.
    """
    if len(header_rows) == 1:
        new_columns = df.iloc[header_rows[0]].astype(str).str.strip()
        df = df.drop(index=header_rows[0]).reset_index(drop=True)
    else:
        merged = []
        for col_idx in range(len(df.columns)):
            parts = []
            for row_idx in header_rows:
                val = str(df.iloc[row_idx, col_idx]).strip()
                if val and val.lower() not in ('nan', 'none', '', 'unnamed'):
                    parts.append(val)
            if not parts:
                parts = [f"col_{col_idx}"]
            name = "_".join(parts)
            name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            name = re.sub(r'_+', '_', name).strip('_')
            merged.append(name)

        seen = {}
        final_columns = []
        for col in merged:
            if col in seen:
                seen[col] += 1
                final_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                final_columns.append(col)

        df = df.copy()
        df.columns = final_columns
        df = df.drop(index=header_rows).reset_index(drop=True)

    return df


# Auto-detection heuristics (disabled)

# def _detect_multi_level_headers(df: pd.DataFrame, max_check: int = 5) -> tuple[list[int], bool]:
#     return [0], False


# File I/O

def read_file(
    filepath_or_buffer,
    header_rows: Optional[List[int]] = None,
    sheet_name: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:

    if hasattr(filepath_or_buffer, 'name'):
        name = filepath_or_buffer.name
    elif isinstance(filepath_or_buffer, str):
        name = filepath_or_buffer
    else:
        name = 'upload.csv'

    is_excel = name.lower().endswith(('.xlsx', '.xls', '.xlsm'))

    if is_excel:
        if header_rows is not None:
            if len(header_rows) == 1:
                # Single header: pandas handles it
                sheets = pd.read_excel(filepath_or_buffer, sheet_name=sheet_name or None, header=header_rows[0])
            else:
                # Multi-level: read raw, then merge
                df_raw = pd.read_excel(filepath_or_buffer, sheet_name=sheet_name or 0, header=None)
                sheets = _merge_multi_level_headers(df_raw, header_rows)
                if sheet_name and isinstance(sheets, dict) and sheet_name in sheets:
                    sheets = {sheet_name: sheets[sheet_name]}
        else:
            # Default: single header
            sheets = pd.read_excel(filepath_or_buffer, sheet_name=sheet_name or None, header=0)
        
        if isinstance(sheets, pd.DataFrame):
            return {sheet_name or 'sheet': sheets}
        return sheets

    else:
        # CSV handling
        if header_rows is not None:
            if len(header_rows) == 1:
                # Single header: let pandas handle it
                df = pd.read_csv(filepath_or_buffer, header=header_rows[0])
            else:
                # Multi-level: read raw, then merge headers manually
                df_raw = pd.read_csv(filepath_or_buffer, header=None)
                df = _merge_multi_level_headers(df_raw, header_rows)
            return {'sheet': df}
        
        # Default: single header at row 0
        return {'sheet': pd.read_csv(filepath_or_buffer, header=0)}


# JSON helpers

def _flatten_dict(d: dict, prefix: str, sep: str = '_') -> dict:
    out = {}
    for k, v in d.items():
        new_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, new_key, sep))
        else:
            out[new_key] = v
    return out

def _parse_json(val):
    if isinstance(val, str):
        stripped = val.strip()
        if stripped.startswith(('{', '[')):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return val

def _is_json_column(series: pd.Series) -> bool:
    import json as _json
    for val in series.dropna().head(10):
        if not isinstance(val, str):
            continue
        stripped = val.strip()
        if stripped.startswith('{'):
            return True
        if stripped.startswith('['):
            try:
                parsed = _json.loads(stripped)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return False
                return True
            except (ValueError, _json.JSONDecodeError):
                pass
    return False

def flatten_json_column(df: pd.DataFrame, col: str, prefix: str = None, log: Optional[List[str]] = None) -> pd.DataFrame:
    log = log or []
    prefix = prefix or col
    row_records: List[Optional[dict]] = []

    for val in df[col]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            row_records.append(None)
            continue

        parsed = _parse_json(val)

        if isinstance(parsed, dict):
            row_records.append(_flatten_dict(parsed, prefix))
        elif isinstance(parsed, list):
            if not parsed:
                row_records.append(None)
            elif all(isinstance(x, dict) for x in parsed):
                keys = set().union(*(d.keys() for d in parsed))
                merged = {f"{prefix}_{k}": '; '.join(str(d.get(k, '')) for d in parsed) for k in keys}
                row_records.append(merged)
            else:
                row_records.append({prefix: '; '.join(str(x) for x in parsed)})
        else:
            row_records.append({prefix: parsed})

    seen, all_keys = set(), []
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
    log.append(f"Dropped original JSON column '{col}'")
    return df


# Null / date cleaning

_NULL_STRINGS = frozenset({'', 'na', 'n/a', 'null', 'none', '-', 'nan', 'nil', '#n/a', '#null!', 'missing', 'unknown'})

def clean_column_value(val):
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(val, str):
        val = re.sub(r'<[^>]+>', '', val).strip()
        if val.lower() in _NULL_STRINGS:
            return None

    return val

def _parse_date_elementwise(series: pd.Series) -> pd.Series:
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


# Main cleaning pipeline

def clean_dataframe(
    df: pd.DataFrame,
    log: Optional[List[str]] = None,
    header_rows: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, List[str]]:

    log = log or []

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(i) for i in col if str(i).strip()).strip() for col in df.columns]
        log.append('Flattened pandas MultiIndex columns')

    original_cols = df.columns.tolist()
    df.columns = [normalize_column_name(c) for c in df.columns]

    renamed = {o: n for o, n in zip(original_cols, df.columns) if str(o) != n}
    if renamed:
        log.append(f'Normalised column names: {renamed}')

    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        df = df.loc[:, ~df.columns.duplicated()]
        log.append(f'Dropped duplicate column names: {dupes}')

    seen_fingerprints: dict = {}
    content_dupes = []

    for col in df.columns.tolist():
        fingerprint = tuple(df[col].astype(str).tolist())
        if fingerprint in seen_fingerprints:
            content_dupes.append((col, seen_fingerprints[fingerprint]))
        else:
            seen_fingerprints[fingerprint] = col

    if content_dupes:
        df = df.drop(columns=[c for c, _ in content_dupes])
        for dup, orig in content_dupes:
            log.append(f"Dropped content-duplicate '{dup}' (identical to '{orig}')")

    for col in df.columns:
        df[col] = df[col].apply(clean_column_value)

    log.append('Applied cell cleaning')

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    if n_before != len(df):
        log.append(f'Dropped {n_before - len(df)} duplicate row(s)')

    for col in df.columns.tolist():
        if _is_json_column(df[col]):
            df = flatten_json_column(df, col, prefix=col, log=log)

    date_pattern = re.compile(r'(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4})|(\d{1,2}\s?[A-Za-z]{3,9}\s?\d{2,4})|([A-Za-z]{3,9}\s?\d{1,2},?\s?\d{2,4})')

    for col in df.columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue

        sample_str = non_null.astype(str).head(20)

        if sample_str.str.match(date_pattern).any():
            try:
                parsed = _parse_date_elementwise(df[col])
                recovered = sum(1 for v in parsed if v is not None and pd.notna(v))

                if recovered > 0:
                    df[col] = [v.strftime('%Y-%m-%d') if (v is not None and pd.notna(v)) else None for v in parsed]
                    log.append(f"Normalised dates in '{col}' ({recovered}/{len(df)} parsed)")
            except Exception as e:
                log.append(f"Could not parse dates in '{col}': {e}")
            continue

        if sample_str.str.match(r'^-?\d*\.?\d+$').sum() >= max(1, len(sample_str) * 0.9):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                log.append(f"Inferred numeric for '{col}'")
            except Exception as e:
                log.append(f"Could not convert '{col}' to numeric: {e}")

    return df, log


def clean_and_profile(df: pd.DataFrame, log: Optional[List[str]] = None) -> Dict:
    from hybridtablerag.core.profiler import profile_dataframe
    df_cleaned, log = clean_dataframe(df.copy(), log=log or [])
    return {
        'cleaned_df': df_cleaned,
        'profile': profile_dataframe(df_cleaned),
        'log': log
    }