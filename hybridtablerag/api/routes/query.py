"""
api/routes/query.py
===================
POST /query — natural language question → structured answer.
"""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from hybridtablerag.api.main import app_state
from hybridtablerag.api.models import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    if app_state.orchestrator is None:
        raise HTTPException(
            status_code=400,
            detail="No data loaded. Call POST /ingest first.",
        )
    try:
        result = app_state.orchestrator.run(
            user_query=request.query,
            session_id=request.session_id,
            reasoning=request.reasoning,
            debug_mode=request.debug_mode,
            force_intent=request.force_intent,
        )
    except Exception as e:
        return QueryResponse(
            success=False, intent="unknown",
            session_id=request.session_id, error=str(e),
        )

    rows, columns, row_count = None, None, None
    python_rows, python_columns = None, None
    chart_json = None
    vector_results = None

    if result.dataframe is not None and not result.dataframe.empty:
        rows      = _safe_records(result.dataframe.to_dict("records"))
        columns   = result.dataframe.columns.tolist()
        row_count = len(result.dataframe)

    if result.python_dataframe is not None and not result.python_dataframe.empty:
        python_rows    = _safe_records(result.python_dataframe.to_dict("records"))
        python_columns = result.python_dataframe.columns.tolist()

    if result.chart is not None:
        try:
            chart_json = result.chart.to_json()
        except Exception:
            chart_json = None

    if getattr(result, "vector_results", None) is not None:
        if not result.vector_results.empty:
            vector_results = _safe_records(result.vector_results.to_dict("records"))

    return QueryResponse(
        success=result.success,
        intent=result.intent,
        session_id=request.session_id,
        sql=result.sql,
        rows=rows, columns=columns, row_count=row_count,
        python_code=result.python_code,
        python_rows=python_rows, python_columns=python_columns,
        chart_json=chart_json,
        vector_results=vector_results,
        vector_query=getattr(result, "vector_query", None),
        llm_answer=getattr(result, "llm_answer", None),
        reasoning=result.reasoning,
        context_used=getattr(result, "context_used", None),
        bts_log=result.bts_log,
        debug_info=result.debug_info if request.debug_mode else None,
        error=result.error,
        python_error=getattr(result, "python_error", None),
    )


def _safe_records(records: list) -> list:
    import datetime
    from decimal import Decimal
    def _safe(v):
        if v is None: return None
        if isinstance(v, (datetime.date, datetime.datetime)): return v.isoformat()
        if isinstance(v, Decimal): return float(v)
        try:
            import numpy as np
            if isinstance(v, (np.integer, np.floating)): return v.item()
        except ImportError:
            pass
        return v
    return [{k: _safe(val) for k, val in row.items()} for row in records]