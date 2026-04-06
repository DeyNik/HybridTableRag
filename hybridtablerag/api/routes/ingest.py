"""
api/routes/ingest.py  — DEBUG EDITION
Every pipeline stage wrapped in its own try/except.
The final except now returns a JSONResponse (never an empty body).
"""
from __future__ import annotations

import io
import re
import traceback
from typing import Optional, List

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from hybridtablerag.api.main import app_state, _rebuild_orchestrator
from hybridtablerag.api.models import BridgeTableInfo, ColumnProfile, IngestResponse
from hybridtablerag.core.cleaner import clean_dataframe, read_file
from hybridtablerag.core.normalizer import Normalizer, NormalizationPlan
from hybridtablerag.core.profiler import profile_dataframe

router = APIRouter()


# ---------- Helpers ----------

def _parse_header_rows(header_rows: Optional[str]) -> Optional[List[int]]:
    if not header_rows or not header_rows.strip():
        return None
    try:
        return [int(x.strip()) for x in header_rows.split(",")]
    except ValueError as e:
        raise ValueError(f"Invalid header_rows value '{header_rows}': {e}")


def _build_profile_response(profile_data: dict) -> dict:
    result = {}
    for col, stats in profile_data.get("columns", {}).items():
        try:
            result[col] = ColumnProfile(
                dtype=stats["dtype"],
                num_nulls=stats["num_nulls"],
                pct_null=stats["pct_null"],
                num_unique=stats["num_unique"],
                sample_values=[str(v) for v in stats["sample_values"][:3]],
                is_multi_valued=stats.get("is_multi_valued", False),
            )
        except Exception as e:
            result[col] = ColumnProfile(
                dtype=str(stats.get("dtype", "unknown")),
                num_nulls=0, pct_null=0.0, num_unique=0,
                sample_values=[f"<profile error: {e}>"],
                is_multi_valued=False,
            )
    return result


def _is_semantic_column(store, table_name: str, col: str) -> bool:
    try:
        rows = store.conn.execute(
            f'SELECT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL LIMIT 50'
        ).fetchall()
        values = [str(r[0]) for r in rows if r[0] is not None]
        if not values:
            return False
        email_pattern = re.compile(r"\S+@\S+\.\S+")
        date_pattern  = re.compile(r"^\d{4}-\d{2}-\d{2}")
        if all(email_pattern.match(v) for v in values):
            return False
        if all(date_pattern.match(v) for v in values):
            return False
        total_chars   = sum(len(v) for v in values)
        alpha_chars   = sum(sum(c.isalpha() for c in v) for v in values)
        alpha_ratio   = alpha_chars / total_chars if total_chars else 0
        avg_token_len = sum(len(v.split()) for v in values) / len(values)
        distinct      = store.conn.execute(
            f'SELECT COUNT(DISTINCT "{col}") FROM "{table_name}"'
        ).fetchone()[0]
        if alpha_ratio > 0.5 and avg_token_len >= 1:
            return True
        if distinct < 50 and alpha_ratio > 0.6:
            return True
        return False
    except Exception:
        return False


def _detect_text_columns(store, table_name: str):
    try:
        schema    = store.get_table_schema(table_name)
        pk_column = None
        for col_info in schema:
            if col_info["column_name"].endswith("_id"):
                pk_column = col_info["column_name"]
                break
        text_cols = [
            col_info["column_name"]
            for col_info in schema
            if col_info["data_type"] in ("VARCHAR", "TEXT")
            and col_info["column_name"] != pk_column
            and _is_semantic_column(store, table_name, col_info["column_name"])
        ]
        return text_cols, pk_column
    except Exception:
        return [], None


# ---------- Ingest Endpoint ----------

@router.post("/")
async def ingest_file(
    file: UploadFile = File(...),
    table_name: str = Form("table"),
    normalize: bool = Form(True),
    header_rows: Optional[str] = Form(None),
):
    cleaning_log: List[str] = []
    norm_log:     List[str] = []
    bts_log:      List[str] = []

    # ── STAGE 0: app-state sanity check ──────────────────────────────────────
    if app_state.store is None:
        return JSONResponse(status_code=503, content={
            "success": False,
            "error": "DuckDB store not initialised (app_state.store is None). "
                     "Check startup logs.",
            "cleaning_log": [], "norm_log": [],
        })
    if app_state.llm is None:
        return JSONResponse(status_code=503, content={
            "success": False,
            "error": "LLM not initialised (app_state.llm is None). "
                     "Check your .env / API key.",
            "cleaning_log": [], "norm_log": [],
        })

    try:
        # ── STAGE 1: read bytes ───────────────────────────────────────────────
        cleaning_log.append("📂 Stage 1: reading uploaded file…")
        try:
            content = await file.read()
            if not content:
                return JSONResponse(status_code=400, content={
                    "success": False, "error": "Uploaded file is empty.",
                    "cleaning_log": cleaning_log, "norm_log": norm_log,
                })
            file_like      = io.BytesIO(content)
            file_like.name = file.filename or "upload.csv"
            cleaning_log.append(
                f"✅ File received: '{file.filename}', {len(content):,} bytes"
            )
        except Exception as e:
            cleaning_log.append(f"❌ Stage 1 failed: {e}")
            return JSONResponse(status_code=500, content={
                "success": False, "error": f"File read failed: {e}",
                "traceback": traceback.format_exc(),
                "cleaning_log": cleaning_log, "norm_log": norm_log,
            })

        # ── STAGE 2: parse header rows ────────────────────────────────────────
        cleaning_log.append(f"📂 Stage 2: parsing header_rows='{header_rows}'…")
        try:
            parsed_headers = _parse_header_rows(header_rows)
            cleaning_log.append(f"✅ Parsed header rows: {parsed_headers}")
        except Exception as e:
            cleaning_log.append(f"❌ Stage 2 failed: {e}")
            return JSONResponse(status_code=400, content={
                "success": False, "error": str(e),
                "cleaning_log": cleaning_log, "norm_log": norm_log,
            })

        # ── STAGE 3: read_file ────────────────────────────────────────────────
        cleaning_log.append("📂 Stage 3: calling read_file()…")
        try:
            sheets = read_file(file_like, header_rows=parsed_headers)
            cleaning_log.append(
                f"✅ read_file returned {len(sheets)} sheet(s): {list(sheets.keys())}"
            )
        except Exception as e:
            cleaning_log.append(f"❌ Stage 3 (read_file) failed: {e}")
            return JSONResponse(status_code=500, content={
                "success": False, "error": f"read_file failed: {e}",
                "traceback": traceback.format_exc(),
                "cleaning_log": cleaning_log, "norm_log": norm_log,
            })

        if not sheets:
            return JSONResponse(status_code=422, content={
                "success": False, "error": "No sheets loaded from file.",
                "cleaning_log": cleaning_log, "norm_log": norm_log,
            })

        all_table_names   = []
        all_relationships = []
        all_bridge_info   = []
        profile_data      = {"columns": {}}
        plan              = None

        for sheet_name, df in sheets.items():
            pfx = f"[sheet={sheet_name}]"

            # ── STAGE 4: clean ────────────────────────────────────────────────
            cleaning_log.append(
                f"📂 {pfx} Stage 4: clean_dataframe "
                f"({len(df)} rows × {len(df.columns)} cols)…"
            )
            try:
                cleaned_df, sheet_clean_log = clean_dataframe(df, log=[])
                cleaning_log.extend(sheet_clean_log)
                cleaning_log.append(
                    f"✅ {pfx} cleaned → "
                    f"{len(cleaned_df)} rows × {len(cleaned_df.columns)} cols"
                )
            except Exception as e:
                cleaning_log.append(f"❌ {pfx} Stage 4 (clean) failed: {e}")
                cleaning_log.append(traceback.format_exc())
                continue

            if cleaned_df.empty:
                cleaning_log.append(f"⚠️ {pfx} empty after cleaning — skipping")
                continue

            # ── STAGE 5: profile ──────────────────────────────────────────────
            cleaning_log.append(f"📂 {pfx} Stage 5: profile_dataframe…")
            try:
                profile_data = profile_dataframe(cleaned_df)
                cleaning_log.append(
                    f"✅ {pfx} profiled {len(profile_data['columns'])} columns"
                )
            except Exception as e:
                cleaning_log.append(
                    f"⚠️ {pfx} Stage 5 (profile) non-fatal: {e}"
                )
                profile_data = {"columns": {}}

            # ── STAGE 6: normalize ────────────────────────────────────────────
            norm_log.append(
                f"📂 {pfx} Stage 6: normalize (normalize={normalize})…"
            )
            try:
                if normalize:
                    plan = Normalizer(llm=app_state.llm).normalize(
                        cleaned_df,
                        table_hint=table_name,
                        profile_hints=profile_data["columns"],
                        log=norm_log,
                    )
                else:
                    plan = NormalizationPlan(
                        main_table_name=table_name,
                        main_df=cleaned_df,
                        log=norm_log,
                    )
                norm_log.append(
                    f"✅ {pfx} normalize done — "
                    f"main='{getattr(plan, 'main_table_name', '?')}', "
                    f"bridges={len(getattr(plan, 'bridge_tables', []))}"
                )
            except Exception as e:
                norm_log.append(f"❌ {pfx} Stage 6 (normalize) failed: {e}")
                norm_log.append(traceback.format_exc())
                continue

            if plan is None or not plan.all_tables:
                norm_log.append(f"⚠️ {pfx} no tables from normalization — skipping")
                continue

            # ── STAGE 7: register in DuckDB ───────────────────────────────────
            norm_log.append(f"📂 {pfx} Stage 7: register_normalization_plan…")
            try:
                app_state.store.register_normalization_plan(plan, norm_log)
                norm_log.append(
                    f"✅ {pfx} registered: {list(plan.all_tables.keys())}"
                )
                all_table_names.extend(list(plan.all_tables.keys()))
                if plan.relationships:
                    all_relationships.extend(plan.relationships)
            except Exception as e:
                norm_log.append(f"❌ {pfx} Stage 7 (register) failed: {e}")
                norm_log.append(traceback.format_exc())
                continue

            # ── STAGE 8: embed ────────────────────────────────────────────────
            bts_log.append(
                f"📂 {pfx} Stage 8: embed "
                f"(vector_store={app_state.vector_store is not None})…"
            )
            if app_state.vector_store:
                try:
                    text_cols, pk_col = _detect_text_columns(
                        app_state.store, plan.main_table_name
                    )
                    bts_log.append(
                        f"   text_cols={text_cols}, pk_col={pk_col}"
                    )
                    if pk_col and text_cols:
                        app_state.vector_store.embed_table(
                            plan.main_table_name, text_cols, pk_col, bts_log
                        )
                        bts_log.append(f"✅ {pfx} embedding complete")
                    else:
                        bts_log.append(
                            f"⚠️ {pfx} skipping embed (pk_col={pk_col}, "
                            f"text_cols={text_cols})"
                        )
                except Exception as e:
                    bts_log.append(
                        f"⚠️ {pfx} Stage 8 (embed) non-fatal: {e}"
                    )
                    bts_log.append(traceback.format_exc())
            else:
                bts_log.append(f"⚠️ {pfx} no vector_store — skipping embed")

            # bridge metadata
            for bt in plan.bridge_tables:
                all_bridge_info.append(
                    BridgeTableInfo(
                        name=bt.name,
                        row_count=len(bt.df),
                        columns=bt.df.columns.tolist(),
                        source_col=bt.source_col,
                        separator=bt.separator,
                    )
                )

        # ── nothing produced? ─────────────────────────────────────────────────
        if not all_table_names:
            return JSONResponse(status_code=422, content={
                "success":        False,
                "table_name":     table_name,
                "row_count":      0,
                "column_count":   0,
                "tables_created": [],
                "bridge_tables":  [],
                "relationships":  [],
                "profile":        {},
                "error":          "Pipeline produced no tables. See logs.",
                "cleaning_log":   cleaning_log,
                "norm_log":       norm_log + bts_log,
            })

        # ── STAGE 9: rebuild orchestrator ─────────────────────────────────────
        norm_log.append("📂 Stage 9: rebuilding orchestrator…")
        try:
            _rebuild_orchestrator(
                list(dict.fromkeys(all_table_names)),
                all_relationships,
                all_table_names[0],
            )
            norm_log.append(
                f"✅ Orchestrator ready — "
                f"tables={list(dict.fromkeys(all_table_names))}"
            )
        except Exception as e:
            norm_log.append(f"❌ Stage 9 (orchestrator) failed: {e}")
            return JSONResponse(status_code=500, content={
                "success":        False,
                "error":          f"Orchestrator rebuild failed: {e}",
                "traceback":      traceback.format_exc(),
                "tables_created": all_table_names,
                "cleaning_log":   cleaning_log,
                "norm_log":       norm_log + bts_log,
            })

        # ── Success ───────────────────────────────────────────────────────────
        return IngestResponse(
            success=True,
            table_name=all_table_names[0],
            row_count=len(plan.main_df) if plan is not None else 0,
            column_count=len(plan.main_df.columns) if plan is not None else 0,
            tables_created=all_table_names,
            bridge_tables=all_bridge_info,
            relationships=all_relationships,
            profile=_build_profile_response(profile_data),
            cleaning_log=cleaning_log,
            norm_log=norm_log + bts_log,
        )

    except Exception as e:
        # Safety-net — guarantees non-empty JSON body no matter what
        return JSONResponse(status_code=500, content={
            "success":      False,
            "error":        str(e),
            "traceback":    traceback.format_exc(),
            "cleaning_log": cleaning_log,
            "norm_log":     norm_log + bts_log,
        })