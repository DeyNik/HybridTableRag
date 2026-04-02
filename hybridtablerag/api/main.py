"""
api/main.py
===========
FastAPI application factory.

Run:
    uvicorn hybridtablerag.api.main:app --reload --port 8000

Or standalone:
    python -m hybridtablerag.api.main
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on path when running directly
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ── Shared application state ──────────────────────────────────────────────────
# One instance per process. Stored here so all routes can import it.

class _AppState:
    """
    Container for all shared resources.
    Initialised in lifespan(), used by all route handlers.
    """
    store:          Optional[object] = None   # DuckDBStore
    llm:            Optional[object] = None   # BaseLLM
    sql_generator:  Optional[object] = None   # LLMSQLGenerator
    orchestrator:   Optional[object] = None   # QueryOrchestrator  (rebuilt on each ingest)
    context_store:  Optional[object] = None   # ContextStore
    vector_store:   Optional[object] = None   # VectorStore (optional)

    # Set when a file is ingested
    table_names:    list = []
    relationships:  list = []
    default_table:  Optional[str] = None


app_state = _AppState()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise all shared resources.
    Shutdown: close connections cleanly.
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    try:
        from hybridtablerag.llm.factory import get_llm
        from hybridtablerag.storage.store import DuckDBStore
        from hybridtablerag.storage.context import ContextStore
        from hybridtablerag.reasoning.sql import LLMSQLGenerator

        # LLM client — reads LLM_PROVIDER from .env
        app_state.llm = get_llm("azure-openai","gpt-5-mini")
        print("[Startup] LLM initialised")

        # DuckDB — creates the file if it doesn't exist
        db_path = os.getenv("DUCKDB_PATH", "data/hybridtablerag.duckdb")
        app_state.store = DuckDBStore(db_path=db_path)
        print(f"[Startup] DuckDB connected: {app_state.store.db_path}")

        # Conversation history store
        app_state.context_store = ContextStore(app_state.store.conn)
        print("[Startup] ContextStore ready")

        # SQL generator
        app_state.sql_generator = LLMSQLGenerator(llm=app_state.llm)

        # Optional: vector store
        try:
            from hybridtablerag.storage.vectors import VectorStore, get_embedding_provider
            provider = get_embedding_provider()
            vs = VectorStore(app_state.store.conn, provider)
            vs.setup()   # installs/loads DuckDB vss extension
            app_state.vector_store = vs
            print("[Startup] VectorStore ready")
        except Exception as ve:
            print(f"[Startup] VectorStore not available: {ve} (continuing without it)")

        # If tables already exist from a previous session, try to restore orchestrator
        existing_tables = app_state.store.list_tables()
        non_system = [t for t in existing_tables if not t.startswith("chat_history")]
        if non_system:
            _rebuild_orchestrator(non_system, [], non_system[0])
            print(f"[Startup] Restored orchestrator for tables: {non_system}")

    except Exception as e:
        print(f"[Startup ERROR] {e}")
        # Don't crash — health endpoint will report degraded

    yield   # ← app runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    if app_state.store:
        app_state.store.close()
        print("[Shutdown] DuckDB closed")


def _rebuild_orchestrator(
    table_names: list,
    relationships: list,
    default_table: str,
) -> None:
    """
    (Re)build the QueryOrchestrator with the current table state.
    Called after every successful ingest so the orchestrator always
    reflects the most recently loaded data.
    """
    from hybridtablerag.reasoning.orchestrator import QueryOrchestrator

    app_state.orchestrator = QueryOrchestrator(
        llm=app_state.llm,
        store=app_state.store,
        context_store=app_state.context_store,
        sql_generator=app_state.sql_generator,
        table_names=table_names,
        relationships=relationships,
        vector_store=app_state.vector_store,
        default_table=default_table,
    )
    app_state.table_names   = table_names
    app_state.relationships = relationships
    app_state.default_table = default_table


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="HybridTableRAG",
    description=(
        "Chat with your CSV/Excel data using natural language. "
        "Cleans, normalises, and queries tabular data via LLM + DuckDB."
    ),
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Streamlit (running on different port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ───────────────────────────────────────────────────────────
from hybridtablerag.api.routes import health, ingest, query, history   # noqa: E402

app.include_router(health.router,  prefix="/health",  tags=["health"])
app.include_router(ingest.router,  prefix="/ingest",  tags=["ingest"])
app.include_router(query.router,   prefix="/query",   tags=["query"])
app.include_router(history.router, prefix="/history", tags=["history"])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "hybridtablerag.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("ENV", "development") == "development",
    )