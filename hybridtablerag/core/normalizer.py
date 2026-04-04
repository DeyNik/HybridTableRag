"""
core/normalizer.py
==================
Automatic normalization of flat cleaned DataFrames into relational tables.

What this does
--------------
Takes a cleaned DataFrame (output of core/cleaner.py) and produces a
NormalizationPlan: a set of named DataFrames with referential integrity,
ready to register as DuckDB tables.

Detection strategy (rule-based first, LLM-assisted second)
-----------------------------------------------------------
  Rule-based (zero LLM cost, handles ~80% of cases):
    - Multi-valued string columns: detect separator patterns (;, |, comma-in-categorical)
    - JSON list-of-dicts columns: each dict becomes a bridge table row
    - High-cardinality columns that look like tags/categories

  LLM-assisted (only called for ambiguous cases):
    - Ambiguous separator: comma could be decimal, list, or part of a sentence
    - Column naming: LLM produces clean table/column names with context awareness
    - Relationship naming: LLM decides the bridge table name from context

Referential integrity
---------------------
  - The primary key column is auto-detected (first ID-like column, or row index)
  - All bridge tables get a FK column pointing to the main table PK
  - DuckDB doesn't enforce FK constraints, but they are declared in
    NormalizationPlan.relationships for the query orchestrator to use

Output
------
  NormalizationPlan
    .main_table_name       str
    .main_df               pd.DataFrame     — the fact table
    .bridge_tables         list[BridgeTable] — one per multi-valued column
    .relationships         list[dict]        — FK declarations
    .log                   list[str]         — audit trail

Usage
-----
  from hybridtablerag.core.normalizer import Normalizer

  plan = Normalizer(llm=llm).normalize(cleaned_df, table_hint="tickets")
  # plan.main_df → the fact table
  # plan.bridge_tables[0].df → first bridge table
  # plan.relationships → [{"from": "ticket_departments.ticket_id",
  #                         "to": "tickets.ticket_id", "type": "many_to_one"}]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BridgeTable:
    """One bridge table extracted from a multi-valued column."""
    name:       str              # e.g. "ticket_departments"
    df:         pd.DataFrame     # the bridge table data
    source_col: str              # original column name in the main table
    pk_col:     str              # FK column name (points to main table PK)
    value_cols: List[str]        # the meaningful columns in this bridge table
    separator:  Optional[str]    # ';', '|', 'json', or None


@dataclass
class NormalizationPlan:
    """Complete normalization output for one DataFrame."""
    main_table_name: str
    main_df:         pd.DataFrame
    bridge_tables:   List[BridgeTable]  = field(default_factory=list)
    relationships:   List[Dict]         = field(default_factory=list)
    log:             List[str]          = field(default_factory=list)

    @property
    def all_tables(self) -> Dict[str, pd.DataFrame]:
        """Returns all tables as {table_name: df} — for bulk DuckDB registration."""
        result = {self.main_table_name: self.main_df}
        for bt in self.bridge_tables:
            result[bt.name] = bt.df
        return result

    @property
    def normalized(self) -> bool:
        return len(self.bridge_tables) > 0


# ── Separator / column type detection ────────────────────────────────────────

# Separators to check in order of specificity
_SEPARATORS = [';', '|']

# Patterns that strongly suggest a column is a free-text field
# (even if it contains semicolons, it shouldn't be split)
_FREE_TEXT_PATTERNS = [
    re.compile(r'\s{2,}'),          # multiple spaces → prose
    re.compile(r'[.!?]\s'),         # sentence-ending punctuation → prose
    re.compile(r'\b(the|a|an|is|are|was|were|and|or|but)\b', re.I),
]

def _looks_like_free_text(val: str) -> bool:
    return any(p.search(val) for p in _FREE_TEXT_PATTERNS)


def _detect_separator(series: pd.Series) -> Optional[str]:
    """
    Return the separator used in a multi-valued column, or None if it's scalar.

    Only returns a separator if:
    - At least 40% of non-null values contain it
    - The values don't look like free text (prose sentences)
    - Average number of parts after splitting is between 1.5 and 20
      (avoids splitting comma-separated sentences or single-value columns)
    """
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None

    # Quick free-text check on a sample
    sample = non_null.head(10)
    free_text_count = sum(1 for v in sample if _looks_like_free_text(v))
    if free_text_count / len(sample) > 0.5:
        return None

    for sep in _SEPARATORS:
        has_sep = non_null.str.contains(re.escape(sep), regex=False)
        if has_sep.mean() >= 0.4:
            # Check average number of parts — too many or too few is suspicious
            avg_parts = non_null[has_sep].apply(
                lambda v: len(v.split(sep))
            ).mean()
            if 1.5 <= avg_parts <= 20:
                return sep

    return None


def _is_json_list_of_dicts(series: pd.Series) -> bool:
    """Return True if most non-null values are JSON arrays of objects."""
    sample = series.dropna().head(10)
    if sample.empty:
        return False
    hits = 0
    for val in sample:
        if not isinstance(val, str):
            continue
        stripped = val.strip()
        if not stripped.startswith('['):
            continue
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                hits += 1
        except (json.JSONDecodeError, ValueError):
            continue
    return hits / len(sample) >= 0.4


def _detect_primary_key(df: pd.DataFrame) -> Optional[str]:
    """
    Find the primary key column.

    Priority:
    1. Column whose name ends in '_id' and has all unique non-null values
    2. First column with all unique non-null values and short values (IDs not UUIDs prose)
    3. Row index (synthetic PK)
    """
    for col in df.columns:
        if col.lower().endswith('_id') or col.lower() == 'id':
            non_null = df[col].dropna()
            if non_null.nunique() == len(non_null) and len(non_null) > 0:
                return col

    for col in df.columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        if non_null.nunique() == len(non_null):
            # Check values are short (ID-like) not long (text-like)
            avg_len = non_null.astype(str).str.len().mean()
            if avg_len <= 30:
                return col

    return None


# ── LLM naming ────────────────────────────────────────────────────────────────

_NAMING_PROMPT = """
You are a database schema designer.

Given information about a column being extracted into a bridge table,
suggest clean snake_case names.

Main table: {main_table}
Source column: {source_col}
Sample values after splitting: {sample_values}
Column content description: {description}

Respond with ONLY valid JSON in this exact structure:
{{
  "bridge_table_name": "snake_case name for the bridge table",
  "value_column_name": "snake_case name for the value column in the bridge table"
}}

Rules:
- bridge_table_name should be: main_table + '_' + singular noun describing the values
- value_column_name should be a clear singular noun for one value
- Keep names short (under 30 chars), descriptive, lowercase
- No explanations, no markdown, just the JSON object
"""


def _llm_name_bridge_table(
    llm,
    main_table: str,
    source_col: str,
    sample_values: List[str],
    description: str,
) -> Tuple[str, str]:
    """
    Use LLM to generate sensible bridge table and value column names.
    Falls back to rule-based names if LLM fails or is not provided.
    """
    # Rule-based fallback — always computed first
    # Strip the main table prefix from the source column if present
    col_stem = source_col
    if col_stem.startswith(main_table + '_'):
        col_stem = col_stem[len(main_table) + 1:]
    col_stem = col_stem.rstrip('s')  # crude singularise

    fallback_table = f"{main_table}_{col_stem}"
    fallback_value = col_stem

    if llm is None:
        return fallback_table, fallback_value

    try:
        prompt = _NAMING_PROMPT.format(
            main_table=main_table,
            source_col=source_col,
            sample_values=sample_values[:5],
            description=description,
        )
        raw = llm.generate(prompt).strip()
        # Strip any accidental markdown fences
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'\s*```$', '', raw).strip()
        parsed = json.loads(raw)
        bridge_name = parsed.get('bridge_table_name', fallback_table)
        value_name  = parsed.get('value_column_name', fallback_value)
        # Sanitise: enforce snake_case, no spaces
        bridge_name = re.sub(r'[^a-z0-9_]', '_', bridge_name.lower()).strip('_')
        value_name  = re.sub(r'[^a-z0-9_]', '_', value_name.lower()).strip('_')
        return bridge_name or fallback_table, value_name or fallback_value
    except Exception:
        return fallback_table, fallback_value


# ── Bridge table builders ─────────────────────────────────────────────────────

def _build_bridge_from_separator(
    df: pd.DataFrame,
    col: str,
    pk_col: str,
    separator: str,
    bridge_table_name: str,
    value_col_name: str,
    log: List[str],
) -> BridgeTable:
    """
    Explode a semicolon/pipe-separated column into a bridge table.

    Input row:  ticket_id=TCKT-1000, impacted_departments="Finance;IT;Operations"
    Output rows:
      ticket_id=TCKT-1000, department=Finance
      ticket_id=TCKT-1000, department=IT
      ticket_id=TCKT-1000, department=Operations
    """
    rows = []
    for _, row in df.iterrows():
        pk_val = row[pk_col]
        val    = row[col]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        parts = [p.strip() for p in str(val).split(separator) if p.strip()]
        for part in parts:
            rows.append({pk_col: pk_val, value_col_name: part})

    bridge_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[pk_col, value_col_name])
    log.append(
        f"Bridge table '{bridge_table_name}': {len(bridge_df)} rows "
        f"from '{col}' (separator='{separator}')"
    )
    return BridgeTable(
        name=bridge_table_name,
        df=bridge_df,
        source_col=col,
        pk_col=pk_col,
        value_cols=[value_col_name],
        separator=separator,
    )


def _build_bridge_from_json_list(
    df: pd.DataFrame,
    col: str,
    pk_col: str,
    bridge_table_name: str,
    log: List[str],
) -> Optional[BridgeTable]:
    """
    Explode a JSON list-of-dicts column into a bridge table.

    Input row:  ticket_id=TCKT-1000,
                affected_systems='[{"system":"Payroll","impact":"Full"}]'
    Output row: ticket_id=TCKT-1000, system=Payroll, impact=Full
    """
    rows = []
    all_keys: set = set()

    for _, row in df.iterrows():
        pk_val = row[pk_col]
        val    = row[col]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        try:
            parsed = json.loads(str(val).strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(parsed, list):
            continue
        for item in parsed:
            if not isinstance(item, dict):
                continue
            flat = {pk_col: pk_val}
            flat.update(item)
            rows.append(flat)
            all_keys.update(item.keys())

    if not rows:
        return None

    bridge_df = pd.DataFrame(rows)

    # Rename value columns to clean snake_case
    rename_map = {}
    for k in all_keys:
        clean = re.sub(r'[^a-z0-9_]', '_', k.lower()).strip('_')
        if clean and clean != k:
            rename_map[k] = clean
    if rename_map:
        bridge_df = bridge_df.rename(columns=rename_map)

    value_cols = [c for c in bridge_df.columns if c != pk_col]
    log.append(
        f"Bridge table '{bridge_table_name}': {len(bridge_df)} rows "
        f"from '{col}' (JSON list-of-dicts), columns: {value_cols}"
    )
    return BridgeTable(
        name=bridge_table_name,
        df=bridge_df,
        source_col=col,
        pk_col=pk_col,
        value_cols=value_cols,
        separator='json',
    )


# ── Ambiguity check for comma separator ───────────────────────────────────────

_COMMA_AMBIGUITY_PROMPT = """
A CSV column named "{col}" has values that contain commas.
Sample values: {samples}

Is this column a list of multiple values separated by commas,
or is each value a single item that happens to contain commas?

Answer with ONLY one word: list OR scalar
"""


def _resolve_comma_ambiguity(llm, col: str, series: pd.Series) -> bool:
    """
    Returns True if the comma-separated column should be treated as a list.
    Uses LLM if available, otherwise applies heuristics.
    """
    non_null = series.dropna().astype(str)
    sample   = non_null.head(5).tolist()

    # Heuristic: if values look like prose (sentence structure), it's scalar
    if any(_looks_like_free_text(v) for v in sample):
        return False

    # Heuristic: if average parts after comma split is between 2 and 8
    # and none of the parts look like sentences, it's a list
    avg_parts = non_null.apply(lambda v: len(v.split(','))).mean()
    if not (1.8 <= avg_parts <= 8):
        return False

    # At this point it's genuinely ambiguous — ask the LLM
    if llm is None:
        return False  # safe default: don't split without LLM confirmation

    try:
        prompt = _COMMA_AMBIGUITY_PROMPT.format(col=col, samples=sample)
        response = llm.generate(prompt).strip().lower()
        return 'list' in response
    except Exception:
        return False


# ── Main Normalizer class ─────────────────────────────────────────────────────

class Normalizer:
    """
    Automatic normalization of a cleaned flat DataFrame into relational tables.

    Usage:
        plan = Normalizer(llm=llm).normalize(df, table_hint="tickets")

    llm is optional — normalization works without it using rule-based detection.
    LLM is used for:
      - Generating clean bridge table and column names (always attempted)
      - Resolving comma ambiguity (only for genuinely ambiguous columns)
    """

    def __init__(self, llm=None):
        self.llm = llm

    def normalize(
        self,
        df: pd.DataFrame,
        table_hint: str = 'table',
        log: Optional[List[str]] = None,
    ) -> NormalizationPlan:
        """
        Analyse df and produce a NormalizationPlan.

        table_hint: base name for the main table (e.g. "tickets")
                    used as prefix for bridge table names.

        Returns NormalizationPlan — call .all_tables to get all DataFrames
        for DuckDB registration.
        """
        log = log or []
        main_table_name = re.sub(r'[^a-z0-9_]', '_', table_hint.lower()).strip('_')

        plan = NormalizationPlan(
            main_table_name=main_table_name,
            main_df=df.copy(),
            log=log,
        )

        # Detect primary key
        pk_col = _detect_primary_key(df)
        if pk_col is None:
            # Synthesise a PK from the row index
            pk_col = f"{main_table_name}_row_id"
            plan.main_df.insert(0, pk_col, range(len(df)))
            log.append(
                f"No natural PK found — synthesised '{pk_col}' from row index"
            )
        else:
            log.append(f"Primary key: '{pk_col}'")

        # Track which columns get extracted (to remove from main table)
        cols_to_drop: List[str] = []

        for col in df.columns:
            if col == pk_col:
                continue

            series = df[col]

            # ── JSON list-of-dicts ─────────────────────────────────────────
            if _is_json_list_of_dicts(series):
                bridge_name, _ = _llm_name_bridge_table(
                    llm=self.llm,
                    main_table=main_table_name,
                    source_col=col,
                    sample_values=series.dropna().head(5).astype(str).tolist(),
                    description=f"JSON list of objects from column '{col}'",
                )
                bridge = _build_bridge_from_json_list(
                    df=plan.main_df,
                    col=col,
                    pk_col=pk_col,
                    bridge_table_name=bridge_name,
                    log=log,
                )
                if bridge is not None:
                    plan.bridge_tables.append(bridge)
                    cols_to_drop.append(col)
                    plan.relationships.append({
                        'from_table':  bridge_name,
                        'from_column': pk_col,
                        'to_table':    main_table_name,
                        'to_column':   pk_col,
                        'type':        'many_to_one',
                    })
                continue

            # ── Semicolon / pipe separated ────────────────────────────────
            sep = _detect_separator(series)

            if sep is None:
                # Check comma as a special case (ambiguous)
                non_null = series.dropna().astype(str)
                if not non_null.empty and ',' in non_null.iloc[0]:
                    if _resolve_comma_ambiguity(self.llm, col, series):
                        sep = ','

            if sep is not None:
                # Get LLM-assisted names
                sample_parts = []
                for v in series.dropna().head(5).astype(str):
                    sample_parts.extend([p.strip() for p in v.split(sep)][:3])

                bridge_name, value_col_name = _llm_name_bridge_table(
                    llm=self.llm,
                    main_table=main_table_name,
                    source_col=col,
                    sample_values=sample_parts[:6],
                    description=(
                        f"Multi-valued column '{col}' with "
                        f"'{sep}'-separated values like: {sample_parts[:3]}"
                    ),
                )
                bridge = _build_bridge_from_separator(
                    df=plan.main_df,
                    col=col,
                    pk_col=pk_col,
                    separator=sep,
                    bridge_table_name=bridge_name,
                    value_col_name=value_col_name,
                    log=log,
                )
                plan.bridge_tables.append(bridge)
                cols_to_drop.append(col)
                plan.relationships.append({
                    'from_table':  bridge_name,
                    'from_column': pk_col,
                    'to_table':    main_table_name,
                    'to_column':   pk_col,
                    'type':        'many_to_one',
                })

        # Remove extracted columns from main table
        if cols_to_drop:
            plan.main_df = plan.main_df.drop(columns=cols_to_drop)
            log.append(
                f"Removed {len(cols_to_drop)} columns from main table "
                f"(moved to bridge tables): {cols_to_drop}"
            )

        log.append(
            f"Normalization complete: 1 main table + "
            f"{len(plan.bridge_tables)} bridge table(s)"
        )
        return plan

    def normalize_sheet_dict(
        self,
        sheets: Dict[str, pd.DataFrame],
        table_hint: str = 'table',
    ) -> Dict[str, NormalizationPlan]:
        """
        Normalize multiple sheets. Returns one NormalizationPlan per sheet.
        """
        return {
            sheet_name: self.normalize(df, table_hint=f"{table_hint}_{sheet_name}")
            for sheet_name, df in sheets.items()
        }