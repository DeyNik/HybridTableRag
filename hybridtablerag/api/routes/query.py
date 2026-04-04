"""
api/routes/query.py
===================
POST /query — natural language question → structured answer.
"""
from __future__ import annotations
from fastapi import APIRouter
from hybridtablerag.api.main import app_state
from hybridtablerag.api.models import QueryRequest, QueryResponse
import datetime
from decimal import Decimal
from typing import Any, Dict, List

router = APIRouter()


def _safe_records(records: List[Dict]) -> List[Dict]:
    """Make DataFrame records JSON-serializable."""
    def _safe(v: Any) -> Any:
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


@router.post("/", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    if app_state.orchestrator is None:
        return QueryResponse(
            success=False, intent="unknown",
            session_id=request.session_id,
            error="No data loaded. Call POST /ingest first.",
        )

    try:
        result = app_state.orchestrator.run(
            user_query=request.query,
            session_id=request.session_id,
            reasoning=request.reasoning,
            debug_mode=request.debug_mode,
            force_intent=request.force_intent,
            vector_top_k=request.top_k_vector,  
        )
    except Exception as e:
        return QueryResponse(
            success=False, intent="unknown",
            session_id=request.session_id, error=str(e),
        )

    # Serialize SQL result
    rows, columns, row_count = None, None, None
    if result.dataframe is not None and not result.dataframe.empty:
        rows      = _safe_records(result.dataframe.to_dict("records"))
        columns   = result.dataframe.columns.tolist()
        row_count = len(result.dataframe)

    # Serialize Python result
    python_rows, python_columns = None, None
    if result.python_dataframe is not None and not result.python_dataframe.empty:
        python_rows    = _safe_records(result.python_dataframe.to_dict("records"))
        python_columns = result.python_dataframe.columns.tolist()

    # Serialize chart
    chart_json = None
    if result.chart is not None:
        try:
            chart_json = result.chart.to_json()
        except Exception:
            pass

    # Serialize vector results
    vector_results = None
    if getattr(result, "vector_results", None) is not None and not result.vector_results.empty:
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
        reasoning=getattr(result, "reasoning", None),
        context_used=getattr(result, "context_used", None),
        bts_log=result.bts_log,
        debug_info=result.debug_info if request.debug_mode else None,
        error=result.error,
        python_error=getattr(result, "python_error", None),
    )