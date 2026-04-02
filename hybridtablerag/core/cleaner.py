"""
core/cleaner.py
===============
DataFrame cleaning pipeline.

Moved from: metadata/schema_profiler.py
Logic:      unchanged —
New:        auto_detect_header_rows() for generic multi-row header handling
            (removes the need for the Streamlit manual header-row input)
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



# Header detection ---------------------------------------------------

# cache and signature for automatic header detection
_HEADER_CACHE: Dict[str, List[int]] = {}


def _file_signature(filepath: str) -> str:
    try:
        stat = os.stat(filepath)
        with open(filepath, 'rb') as f:
            head = f.read(4096)

        return hashlib.md5(
            (filepath + str(stat.st_size)).encode() + head
        ).hexdigest()
    except Exception:
        return filepath


def _evaluate_df_quality(df: pd.DataFrame) -> float:
    score = 0

    if df.empty or df.shape[1] == 0:
        return -999

    cols = list(df.columns)

    unnamed_count = sum('unnamed' in str(c).lower() for c in cols)
    duplicate_count = len(cols) - len(set(cols))

    score -= unnamed_count * 3
    score -= duplicate_count * 3

    score += len(set(cols)) / len(cols)

    score -= df.head(5).isna().mean().mean() * 2

    score += min(df.dtypes.nunique(), 5) * 0.3

    if unnamed_count == 0 and duplicate_count == 0:
        score += 2

    return score
  

def _try_read_csv(filepath, header_option):
    try:
        return pd.read_csv(filepath, header=header_option)
    except Exception:
        return None
    
def _select_best_csv_parse(filepath: str) -> pd.DataFrame:
    file_sig = _file_signature(filepath)

    # 1. Cache hit
    if file_sig in _HEADER_CACHE:
        header_rows = _HEADER_CACHE[file_sig]
        hdr_arg = header_rows if len(header_rows) > 1 else header_rows[0]
        return pd.read_csv(filepath, header=hdr_arg)

    # 2. Fast path (clean data)
    df0 = _try_read_csv(filepath, 0)
    if df0 is not None:
        try:
            cleaned0, _ = clean_dataframe(df0.copy(), log=[])
            score0 = _evaluate_df_quality(cleaned0)

            if score0 >= 2:
                _HEADER_CACHE[file_sig] = [0]
                return cleaned0
        except Exception:
            pass

    # 3. Full search
    candidates = [[0], [0, 1], [0, 1, 2], None]

    best_df = None
    best_score = -999
    best_header = None

    for header_rows in candidates:
        if isinstance(header_rows, list):
            hdr_arg = header_rows if len(header_rows) > 1 else header_rows[0]
        else:
            hdr_arg = header_rows

        df = _try_read_csv(filepath, hdr_arg)
        if df is None:
            continue

        try:
            cleaned_df, _ = clean_dataframe(df.copy(), log=[])
        except Exception:
            continue

        score = _evaluate_df_quality(cleaned_df)

        if score > best_score:
            best_score = score
            best_df = cleaned_df
            best_header = header_rows

    if best_df is None:
        raise ValueError("CSV parsing failed")

    if best_score < 0:
        raise ValueError("Low-confidence parse")

    # 4. Cache
    if isinstance(best_header, list):
        _HEADER_CACHE[file_sig] = best_header
    elif best_header is None:
        _HEADER_CACHE[file_sig] = [0]
    else:
        _HEADER_CACHE[file_sig] = [best_header]

    return best_df


def read_file(
    filepath_or_buffer,
    header_rows: Optional[List[int]] = None,
    sheet_name: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Read a CSV or Excel file into a dict of {sheet_name: DataFrame}.

    For CSV, returns {"sheet": df}.
    For Excel, returns one entry per sheet (or just the selected one).
    header_rows: if None, auto-detected for CSV. For Excel always [0] unless overridden.

    This is the single entry point for file I/O in the entire system.
    """
    if hasattr(filepath_or_buffer, 'name'):
        name = filepath_or_buffer.name
    elif isinstance(filepath_or_buffer, str):
        name = filepath_or_buffer
    else:
        name = 'upload.csv'

    is_excel = name.lower().endswith(('.xlsx', '.xls', '.xlsm'))

    if is_excel:
        hdr = header_rows or [0]
        hdr_arg = hdr if len(hdr) > 1 else hdr[0]
        sheets = pd.read_excel(
            filepath_or_buffer,
            sheet_name=sheet_name or None,
            header=hdr_arg,
        )
        if isinstance(sheets, pd.DataFrame):
            return {sheet_name or 'sheet': sheets}
        return sheets  # dict of sheet_name → df

    else:
        if header_rows is not None:
            hdr_arg = header_rows if len(header_rows) > 1 else header_rows[0]
            df = pd.read_csv(filepath_or_buffer, header=hdr_arg)
            return {'sheet': df}

        if isinstance(filepath_or_buffer, str):
            df = _select_best_csv_parse(filepath_or_buffer)
            return {'sheet': df}

        df = pd.read_csv(filepath_or_buffer, header=0)
        return {'sheet': df}
        


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
    """
    Return True only for JSON dict columns (single objects like {"key": "val"}).

    Deliberately returns False for JSON list-of-dicts ('[{"key":"val"},...]').
    Those are handled by the normalizer which builds proper bridge tables.
    Flattening them here with "; " joins would destroy the relational structure.
    """
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
                    return False  # normalizer handles list-of-dicts
                return True       # list of scalars: cleaner can flatten
            except (ValueError, _json.JSONDecodeError):
                pass
    return False


def flatten_json_column(
    df: pd.DataFrame,
    col: str,
    prefix: str = None,
    log: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Flatten a JSON column into multiple new columns, preserving row order exactly.
    Handles: dict, list-of-dicts, list-of-scalars, scalar, null.
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
        log.append(f"Flattened '{col}' → '{k}'")

    df = df.drop(columns=[col])
    log.append(f"Dropped original JSON column '{col}'")
    return df


# Null / na-like cleaning 

_NULL_STRINGS = frozenset({
    '', 'na', 'n/a', 'null', 'none', '-', 'nan', 'nil',
    '#n/a', '#null!', 'missing', 'unknown',
})


def clean_column_value(val):
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


#  Date parsing — element-wise for mixed formats

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


# ── Main cleaning pipeline ────────────────────────────────────────────────────

def clean_dataframe(
    df: pd.DataFrame,
    log: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean a DataFrame through 8 sequential steps:

      1. Flatten MultiIndex columns
      2. Normalise column names
      3a. Drop duplicate column names
      3b. Drop content-duplicate columns (identical values, different name)
      4. Clean cell values (nulls, HTML, whitespace)
      5. Drop duplicate rows
      6. Flatten JSON columns
      7. Normalise dates (element-wise, handles mixed formats)
      8. Infer numeric columns

    Returns (cleaned_df, log).
    NOTE: multi-valued columns (semicolon/pipe-separated) are NOT exploded here.
    That is the normalizer's job. The cleaner keeps them intact so the normalizer
    can inspect and decide what to do with them.
    """
    log = log or []

    # 1. Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            '_'.join(str(i) for i in col if str(i).strip() != '').strip()
            for col in df.columns
        ]
        log.append('Flattened MultiIndex columns')

    # 2. Normalise column names
    original_cols = df.columns.tolist()
    df.columns = [normalize_column_name(c) for c in df.columns]
    renamed = {o: n for o, n in zip(original_cols, df.columns) if str(o) != n}
    if renamed:
        log.append(f'Normalised column names: {renamed}')

    # 3a. Drop duplicate column names
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        df = df.loc[:, ~df.columns.duplicated()]
        log.append(f'Dropped duplicate column names: {dupes}')

    # 3b. Drop content-duplicate columns
    seen_fingerprints: dict = {}
    content_dupes: List[Tuple[str, str]] = []
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
            log.append(f"Dropped content-duplicate '{dup}' (identical to '{orig}')")

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
    # Note: multi-valued string columns (semicolons etc.) are left intact here.
    # The normalizer will handle those separately and produce bridge tables.
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
                        f"Normalised dates in '{col}' → YYYY-MM-DD "
                        f"({recovered}/{len(df)} parsed)"
                    )
            except Exception as e:
                log.append(f"Could not parse dates in '{col}': {e}")
            continue

        # Numeric detection (>=90% threshold)
        numeric_matches = sample_str.str.match(r'^-?\d*\.?\d+$').sum()
        if numeric_matches >= max(1, len(sample_str) * 0.9):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                log.append(f"Inferred numeric for '{col}'")
            except Exception as e:
                log.append(f"Could not convert '{col}' to numeric: {e}")

    return df, log


def clean_and_profile(
    df: pd.DataFrame,
    log: Optional[List[str]] = None,
) -> Dict:
    """
    Convenience wrapper: clean then profile.

    Returns:
        cleaned_df: cleaned DataFrame
        profile:    column-level statistics (from core/profiler.py)
        log:        list of actions taken
    """
    from hybridtablerag.core.profiler import profile_dataframe

    df_cleaned, log = clean_dataframe(df.copy(), log=log or [])
    return {
        'cleaned_df': df_cleaned,
        'profile':    profile_dataframe(df_cleaned),
        'log':        log,
    }