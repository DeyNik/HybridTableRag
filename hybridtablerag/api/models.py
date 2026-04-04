"""
api/models.py
=============
Pydantic request/response contracts for all API routes.
These are the schemas your colleague's microservice sees when calling yours.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# Ingest 

class IngestOptions(BaseModel):
    """Options sent alongside the file upload."""
    table_name:  str        = Field("table",  description="Base name for the DuckDB table")
    normalize:   bool       = Field(True,     description="Auto-normalize multi-valued columns into bridge tables")
    header_rows: Optional[List[int]] = Field(None, description="Header row indices e.g. [0,1] for two-row header. Auto-detected if None.")
    sheet_name:  Optional[str]       = Field(None, description="Specific Excel sheet. Ingests all sheets if None.")


class ColumnProfile(BaseModel):
    dtype:            str
    num_nulls:        int
    pct_null:         float
    num_unique:       int
    sample_values:    List[Any]
    is_multi_valued:  bool = False


class BridgeTableInfo(BaseModel):
    name:        str
    row_count:   int
    columns:     List[str]
    source_col:  str          # original column that was split
    separator:   str          # ';' | '|' | 'json'


class IngestResponse(BaseModel):
    success:          bool
    table_name:       str
    row_count:        int
    column_count:     int
    tables_created:   List[str]          # [main_table, bridge1, bridge2, ...]
    bridge_tables:    List[BridgeTableInfo]
    relationships:    List[Dict[str, str]]
    profile:          Dict[str, ColumnProfile]
    cleaning_log:     List[str]
    norm_log:         List[str]
    error:            Optional[str] = None


# Query

class QueryRequest(BaseModel):
    query:         str
    session_id:    str   = Field(...,   description="UUID for conversation tracking")
    reasoning:     bool  = Field(False, description="Ask LLM for chain-of-thought")
    debug_mode:    bool  = Field(False, description="Return full internals in response")
    force_intent:  Optional[str] = Field(None, description="sql | python | vector | conversational")
    top_k_vector:  int   = Field(10,    description="Max vector search results")


class QueryResponse(BaseModel):
    success:          bool
    intent:           str            # sql | python | vector | conversational
    session_id:       str

    # SQL result
    sql:              Optional[str]         = None
    rows:             Optional[List[Dict]]  = None   # df.to_dict('records')
    columns:          Optional[List[str]]   = None
    row_count:        Optional[int]         = None

    # Python result
    python_code:      Optional[str]         = None
    python_rows:      Optional[List[Dict]]  = None
    python_columns:   Optional[List[str]]   = None
    chart_json:       Optional[str]         = None   # plotly fig.to_json()

    # Vector result
    vector_results:   Optional[List[Dict]]  = None
    vector_query:     Optional[str]         = None

    # Conversational result
    llm_answer:       Optional[str]         = None

    # Meta
    reasoning:        Optional[str]         = None
    context_used:     Optional[str]         = None
    bts_log:          List[str]             = []
    debug_info:       Optional[Dict]        = None
    error:            Optional[str]         = None
    python_error:     Optional[str]         = None


# Session / History

class SessionTurn(BaseModel):
    session_id:     str
    turn:           int
    timestamp:      str
    user_query:     str
    intent:         str
    result_summary: str
    sql_generated:  Optional[str] = None
    error:          Optional[str] = None


class HistoryResponse(BaseModel):
    session_id: str
    turns:      List[SessionTurn]


class ClearHistoryResponse(BaseModel):
    success:    bool
    session_id: str
    turns_deleted: int


# Health

class HealthResponse(BaseModel):
    status:        str             # "ok" | "degraded" | "not_ready"
    duckdb:        bool
    llm:           bool
    vector_store:  bool
    tables:        List[str]
    row_counts:    Dict[str, int]  # {table_name: row_count}
    version:       str = "0.2.0"