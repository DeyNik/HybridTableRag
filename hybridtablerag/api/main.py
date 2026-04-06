from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class _AppState:
    store: Optional[object] = None
    llm: Optional[object] = None
    sql_generator: Optional[object] = None
    orchestrator: Optional[object] = None
    context_store: Optional[object] = None
    vector_store: Optional[object] = None

    table_names: list = []
    relationships: list = []
    default_table: Optional[str] = None


app_state = _AppState()


def _rebuild_orchestrator(table_names: list, relationships: list, default_table: str):
    if not table_names:
        raise ValueError("Cannot build orchestrator without tables")

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

    app_state.table_names = table_names
    app_state.relationships = relationships
    app_state.default_table = default_table


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from hybridtablerag.llm.factory import get_llm
        from hybridtablerag.storage.store import DuckDBStore
        from hybridtablerag.storage.context import ContextStore
        from hybridtablerag.reasoning.sql import LLMSQLGenerator

        app_state.llm = get_llm()

        db_path = os.getenv("DUCKDB_PATH", "data/hybridtablerag.duckdb")
        db_file = Path(db_path)
        db_dir = db_file.parent
        # Ensure directory exists
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
                print(f"[Startup INFO] Created missing directory: {db_dir}")
            except Exception as e:
                print(f"[Startup ERROR] Cannot create directory {db_dir}: {e}")
                raise

        # Initialize DuckDBStore
        try:
            app_state.store = DuckDBStore(db_path=str(db_file))
            print(f"[Startup INFO] DuckDBStore initialized at {db_file}")
        except Exception as e:
            print(f"[Startup ERROR] Failed to init DuckDBStore: {e}")
            app_state.store = None
            raise RuntimeError(f"DuckDB initialization failed: {e}")

        app_state.context_store = ContextStore(app_state.store.conn)
        app_state.sql_generator = LLMSQLGenerator(llm=app_state.llm)

        try:
            from hybridtablerag.storage.vectors import VectorStore, get_embedding_provider

            provider = get_embedding_provider()
            vs = VectorStore(app_state.store.conn, provider)
            vs.setup()
            app_state.vector_store = vs
        except Exception:
            app_state.vector_store = None

        existing_tables = app_state.store.list_tables()
        non_system = [t for t in existing_tables if not t.startswith("chat_history")]

        if non_system:
            _rebuild_orchestrator(non_system, [], non_system[0])

    except Exception as e:
        print(f"[Startup ERROR] {e}")

    yield

    if app_state.store:
        app_state.store.close()


app = FastAPI(
    title="HybridTableRAG",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from hybridtablerag.api.routes import health, ingest, query, history

app.include_router(health.router, prefix="/health")
app.include_router(ingest.router, prefix="/ingest")
app.include_router(query.router, prefix="/query")
app.include_router(history.router, prefix="/history")