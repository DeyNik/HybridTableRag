"""
api/routes/history.py
=====================
GET  /history/{session_id}          — get conversation turns
DELETE /history/{session_id}        — clear session history
GET  /history/sessions/all          — list all sessions (admin/debug)
"""

from fastapi import APIRouter, HTTPException
from hybridtablerag.api.main import app_state
from hybridtablerag.api.models import HistoryResponse, ClearHistoryResponse, SessionTurn

router = APIRouter()


@router.get("/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str, last_n: int = 20):
    """Return the last N conversation turns for a session."""
    if app_state.context_store is None:
        raise HTTPException(503, "Context store not ready")
    try:
        turns_raw = app_state.context_store.get_history(session_id, last_n=last_n)
        turns = [
            SessionTurn(
                session_id=t.get("session_id", session_id),
                turn=t.get("turn", 0),
                timestamp=str(t.get("timestamp", "")),
                user_query=t.get("user_query", ""),
                intent=t.get("intent", ""),
                result_summary=t.get("result_summary", ""),
                sql_generated=t.get("sql_generated"),
                error=t.get("error"),
            )
            for t in turns_raw
        ]
        return HistoryResponse(session_id=session_id, turns=turns)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/{session_id}", response_model=ClearHistoryResponse)
async def clear_history(session_id: str):
    """Clear all conversation turns for a session."""
    if app_state.context_store is None:
        raise HTTPException(503, "Context store not ready")
    try:
        before = len(app_state.context_store.get_history(session_id, last_n=9999))
        app_state.context_store.clear_session(session_id)
        return ClearHistoryResponse(
            success=True,
            session_id=session_id,
            turns_deleted=before,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/sessions/all")
async def all_sessions():
    """Return a summary of all sessions (for admin/debug)."""
    if app_state.context_store is None:
        raise HTTPException(503, "Context store not ready")
    try:
        df = app_state.context_store.get_all_sessions()
        return df.to_dict("records")
    except Exception as e:
        raise HTTPException(500, str(e))