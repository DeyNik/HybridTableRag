"""
api/routes/health.py
====================
GET /health — liveness + readiness check.
"""

from fastapi import APIRouter
from hybridtablerag.api.models import HealthResponse
from hybridtablerag.api.main import app_state

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Returns system status. Used by load balancers and for startup confirmation.
    Does not raise HTTP errors — always returns 200 with status field.
    """
    duckdb_ok      = False
    llm_ok         = False
    vector_ok      = False
    tables:  list  = []
    row_counts: dict = {}

    # DuckDB check
    try:
        if app_state.store and app_state.store.conn:
            tables     = app_state.store.list_tables()
            non_system = [t for t in tables if t != "chat_history"]
            for t in non_system:
                try:
                    n = app_state.store.conn.execute(
                        f"SELECT COUNT(*) FROM {t}"
                    ).fetchone()[0]
                    row_counts[t] = int(n)
                except Exception:
                    row_counts[t] = -1
            duckdb_ok = True
    except Exception:
        pass

    # LLM check
    try:
        llm_ok = app_state.llm is not None
    except Exception:
        pass

    # Vector store check
    try:
        vector_ok = app_state.vector_store is not None
    except Exception:
        pass

    if duckdb_ok and llm_ok:
        status = "ok"
    elif duckdb_ok or llm_ok:
        status = "degraded"
    else:
        status = "not_ready"

    return HealthResponse(
        status=status,
        duckdb=duckdb_ok,
        llm=llm_ok,
        vector_store=vector_ok,
        tables=tables,
        row_counts=row_counts,
    )