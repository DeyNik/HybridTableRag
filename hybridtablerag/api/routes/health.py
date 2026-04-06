"""
api/routes/health.py  — fixed
Bug fixed: removed import of non-existent storage.utils._escape_identifier
"""
from fastapi import APIRouter
from hybridtablerag.api.models import HealthResponse
from hybridtablerag.api.main import app_state

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check():
    duckdb_ok = llm_ok = vector_ok = False
    tables, row_counts = [], {}

    try:
        if app_state.store and app_state.store.conn:
            tables = app_state.store.list_tables()
            for t in tables:
                if t == "chat_history":
                    continue
                try:
                    n = app_state.store.conn.execute(
                        f'SELECT COUNT(*) FROM "{t}"'   # FIXED: inline quoting, no import needed
                    ).fetchone()[0]
                    row_counts[t] = int(n)
                except Exception:
                    row_counts[t] = -1
            duckdb_ok = True
    except Exception:
        pass

    llm_ok    = app_state.llm is not None
    vector_ok = app_state.vector_store is not None
    status    = "ok" if (duckdb_ok and llm_ok) else ("degraded" if (duckdb_ok or llm_ok) else "not_ready")

    return HealthResponse(
        status=status, duckdb=duckdb_ok, llm=llm_ok,
        vector_store=vector_ok, tables=tables, row_counts=row_counts,
    )