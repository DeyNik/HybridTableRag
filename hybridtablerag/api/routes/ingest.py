"""
api/routes/ingest.py
====================
POST /ingest → clean → normalize → register in DuckDB → embed → rebuild orchestrator
"""
from __future__ import annotations

import io
from typing import Optional, List

from fastapi import APIRouter, File, Form, UploadFile

from hybridtablerag.api.main import app_state, _rebuild_orchestrator
from hybridtablerag.api.models import (
    BridgeTableInfo,
    ColumnProfile,
    IngestResponse,
)
from hybridtablerag.core.cleaner import clean_dataframe, read_file
from hybridtablerag.core.normalizer import NormalizationPlan, Normalizer
from hybridtablerag.core.profiler import profile_dataframe

router = APIRouter()


def _parse_header_rows(header_rows: Optional[str]) -> Optional[List[int]]:
    if header_rows is None: return None
    if isinstance(header_rows, list): return header_rows
    try:
        return [int(x.strip()) for x in str(header_rows).split(",")]
    except Exception:
        raise ValueError(f"header_rows must be comma-separated integers, got: {header_rows}")


def _build_profile_response(profile_data: dict) -> dict:
    return {
        col: ColumnProfile(
            dtype=stats["dtype"],
            num_nulls=stats["num_nulls"],
            pct_null=stats["pct_null"],
            num_unique=stats["num_unique"],
            sample_values=[str(v) for v in stats["sample_values"][:3]],
            is_multi_valued=stats.get("is_multi_valued", False),
        )
        for col, stats in profile_data["columns"].items()
    }


def _detect_text_columns(profile_data: dict) -> List[str]:
    cols = []
    for col, stats in profile_data["columns"].items():
        if stats["dtype"] != "object": continue
        if stats.get("is_multi_valued"): continue
        if stats["num_unique"] <= 5: continue  # Skip low-cardinality categoricals
        cols.append(col)
    return cols


def _find_pk_column(df, table_name: str) -> str:
    """Heuristic: prefer *_id, fallback to first column."""
    for c in df.columns:
        if c.lower().endswith("_id") or c.lower() == "id":
            return c
    return df.columns[0]


@router.post("/", response_model=IngestResponse)
async def ingest_file(
    file:        UploadFile        = File(...),
    table_name:  str               = Form("table"),
    normalize:   bool              = Form(True),
    header_rows: Optional[str]     = Form(None),
    sheet_name:  Optional[str]     = Form(None),
):
    cleaning_log: list = []
    norm_log:     list = []

    try:
        parsed_header_rows = _parse_header_rows(header_rows)
    except ValueError as e:
        return IngestResponse(success=False, table_name=table_name, row_count=0, column_count=0,
                              tables_created=[], bridge_tables=[], relationships=[], profile={},
                              cleaning_log=[], norm_log=[], error=str(e))

    # Read file
    try:
        content = await file.read()
        file_like = io.BytesIO(content)
        file_like.name = file.filename or "upload.csv"
        sheets = read_file(file_like, header_rows=parsed_header_rows, sheet_name=sheet_name)
    except Exception as e:
        return IngestResponse(success=False, table_name=table_name, row_count=0, column_count=0,
                              tables_created=[], bridge_tables=[], relationships=[], profile={},
                              cleaning_log=[], norm_log=[], error=f"Failed to read file: {e}")

    all_table_names = []
    all_relationships = []
    all_bridge_info = []
    final_main_df = None
    final_profile = None

    # Process EACH sheet
    for sheet_name_key, sheet_df in sheets.items():
        current_table_name = table_name if len(sheets) == 1 else f"{table_name}_{sheet_name_key}"

        # Clean
        try:
            cleaned_df, log = clean_dataframe(sheet_df, log=[])
            cleaning_log.extend([f"[{sheet_name_key}] {l}" for l in log])
        except Exception as e:
            return IngestResponse(success=False, table_name=current_table_name, row_count=0, column_count=0,
                                  tables_created=[], bridge_tables=[], relationships=[], profile={},
                                  cleaning_log=cleaning_log, norm_log=norm_log, error=f"Cleaning failed: {e}")

        # Profile
        profile_data = profile_dataframe(cleaned_df)

        # Normalize
        if normalize:
            try:
                plan = Normalizer(llm=app_state.llm).normalize(cleaned_df, table_hint=current_table_name, log=norm_log)
            except Exception as e:
                norm_log.append(f"[{sheet_name_key}] Normalization failed (fallback to main): {e}")
                plan = NormalizationPlan(main_table_name=current_table_name, main_df=cleaned_df, log=norm_log)
        else:
            plan = NormalizationPlan(main_table_name=current_table_name, main_df=cleaned_df, log=norm_log)
            norm_log.append(f"[{sheet_name_key}] Normalization skipped")

        # Register
        try:
            app_state.store.register_normalization_plan(plan, norm_log)
        except Exception as e:
            return IngestResponse(success=False, table_name=current_table_name, row_count=len(cleaned_df), column_count=len(cleaned_df.columns),
                                  tables_created=[], bridge_tables=[], relationships=plan.relationships,
                                  profile=_build_profile_response(profile_data), cleaning_log=cleaning_log, norm_log=norm_log,
                                  error=f"DuckDB registration failed: {e}")

        # Collect metadata
        all_table_names.extend(list(plan.all_tables.keys()))
        all_relationships.extend(plan.relationships)

        all_bridge_info.extend([
            BridgeTableInfo(
                name=bt.name,
                row_count=len(bt.df),
                columns=bt.df.columns.tolist(),
                source_col=getattr(bt, "source_col", "unknown"),
                separator=getattr(bt, "separator", "unknown"),
            )
            for bt in plan.bridge_tables
        ])

        if final_main_df is None:
            final_main_df = plan.main_df
            final_profile = profile_data

        # Vector embedding
        if app_state.vector_store is not None:
            try:
                text_cols = _detect_text_columns(profile_data)
                pk_col = _find_pk_column(plan.main_df, plan.main_table_name)
                if text_cols:
                    app_state.vector_store.embed_table(
                        table_name=plan.main_table_name,
                        text_columns=text_cols,
                        pk_column=pk_col,
                        bts_log=norm_log,
                    )
            except Exception as ve:
                norm_log.append(f"[{sheet_name_key}] Vector embedding skipped: {ve}")

    # Rebuild orchestrator ONCE
    if all_table_names:
        _rebuild_orchestrator(
            table_names=list(dict.fromkeys(all_table_names)),  # preserve order, dedupe
            relationships=all_relationships,
            default_table=all_table_names[0],
        )

    return IngestResponse(
        success=True,
        table_name=all_table_names[0] if all_table_names else table_name,
        row_count=len(final_main_df) if final_main_df is not None else 0,
        column_count=len(final_main_df.columns) if final_main_df is not None else 0,
        tables_created=list(dict.fromkeys(all_table_names)),
        bridge_tables=all_bridge_info,
        relationships=all_relationships,
        profile=_build_profile_response(final_profile) if final_profile else {},
        cleaning_log=cleaning_log,
        norm_log=norm_log,
    )