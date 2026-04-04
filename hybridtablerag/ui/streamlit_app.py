"""
ui/streamlit_app.py — API-INTEGRATED VERSION
✅ Calls real /ingest endpoint
✅ Shows real API responses
✅ Debug-by-default, multi-file, multi-sheet
"""

import io
import os
import sys
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="HybridTableRAG — Pipeline", layout="wide", initial_sidebar_state="expanded")

# ── CSS (same as before, omitted for brevity) ─────────────────────────────────
st.markdown("""
<style>
/* ... same CSS as previous version ... */
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "session_id": None,
    "upload_queue": [],
    "api_ingest_result": None,
    "loaded_tables": [],
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.session_id:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# ── Sidebar: API Health ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 API Status")
    try:
        health = requests.get(f"{API_BASE}/health/", timeout=3).json()
        status_color = "ok" if health["status"] == "ok" else "err"
        st.markdown(f"**Status:** <span style='color:{'green' if status_color=='ok' else 'red'};font-weight:bold'>{health['status'].upper()}</span>", unsafe_allow_html=True)
        
        if health.get("duckdb"):
            st.success("✅ DuckDB connected")
        else:
            st.error("❌ DuckDB disconnected")
        
        if health.get("llm"):
            st.success("✅ LLM ready")
        else:
            st.error("❌ LLM not ready")
        
        if health.get("tables"):
            non_sys = [t for t in health["tables"] if t != "chat_history"]
            if non_sys:
                st.markdown("**📊 Loaded tables:**")
                for t in non_sys:
                    cnt = health.get("row_counts", {}).get(t, "?")
                    st.caption(f"📋 `{t}` — {cnt:,} rows" if isinstance(cnt, int) else f"📋 `{t}`")
    except Exception as e:
        st.error(f"🔌 API unreachable: {e}")
    
    st.divider()
    st.caption(f"Session: `{st.session_state.session_id}`")
    if st.button("🗑 Clear session"):
        for k in list(st.session_state.keys()):
            if k not in ["session_id"]:
                del st.session_state[k]
        st.rerun()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _pills(**kwargs) -> str:
    return "<div class='stat-row'>" + "".join(f"<div class='stat-pill'>{k}<span>{v}</span></div>" for k,v in kwargs.items()) + "</div>"

def _render_log(log: List[str], title: str = "Log"):
    if not log:
        st.caption("No log entries.")
        return
    html = "".join(f"<div>{l}</div>" for l in log)
    st.markdown(f"<div style='background:#f8f9fb;border:1px solid #e2e6ea;border-left:3px solid #2563eb;border-radius:0 6px 6px 0;padding:.7rem 1rem;max-height:280px;overflow-y:auto;font-family:monospace;font-size:.72rem'>{html}</div>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("HybridTableRAG")
st.caption("Upload → Clean → Normalize → Load → Query  |  Debug always visible")
st.divider()

# ── 1. UPLOAD ─────────────────────────────────────────────────────────────────
st.markdown("<div style='font-family:monospace;font-size:.8rem;font-weight:600;color:#2563eb;text-transform:uppercase;letter-spacing:.08em;margin:.8rem 0 .3rem'>① Upload Files</div>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop CSV or Excel files (multiple supported)",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    for uf in uploaded_files:
        if uf.name not in [f["name"] for f in st.session_state.upload_queue]:
            content = uf.read()
            # Preview sheets
            sheets_preview = {}
            if uf.name.lower().endswith((".xlsx", ".xls")):
                try:
                    xls = pd.ExcelFile(io.BytesIO(content))
                    for sn in xls.sheet_names:
                        sheets_preview[sn] = pd.read_excel(xls, sheet_name=sn, nrows=3)
                except Exception as e:
                    st.warning(f"Could not preview {uf.name}: {e}")
            else:
                sheets_preview["__default__"] = pd.read_csv(io.BytesIO(content), nrows=3)
            
            st.session_state.upload_queue.append({
                "name": uf.name,
                "bytes": content,
                "sheets_preview": sheets_preview,
                "size_kb": len(content) // 1024,
            })
            st.success(f"✅ Added `{uf.name}` ({len(sheets_preview)} sheet(s))")

if st.session_state.upload_queue:
    st.markdown("**Upload Queue:**")
    for item in st.session_state.upload_queue:
        with st.expander(f"📄 {item['name']} ({item['size_kb']}KB)", expanded=False):
            for sn, df in item["sheets_preview"].items():
                st.caption(f"Sheet: `{sn}`")
                st.dataframe(df, use_container_width=True, height=100)
    
    col_clear, col_process = st.columns([1, 4])
    with col_clear:
        if st.button("🗑 Clear"):
            st.session_state.upload_queue = []
            st.rerun()
    with col_process:
        run_ingest = st.button("▶ Send to API /ingest", type="primary", use_container_width=True, disabled=not st.session_state.upload_queue)
else:
    st.info("Upload files to begin.")

st.divider()

# ── 2. CALL API /ingest ───────────────────────────────────────────────────────
st.markdown("<div style='font-family:monospace;font-size:.8rem;font-weight:600;color:#2563eb;text-transform:uppercase;letter-spacing:.08em;margin:.8rem 0 .3rem'>② Process via API</div>", unsafe_allow_html=True)

if run_ingest and st.session_state.upload_queue:
    with st.spinner("📤 Sending files to API /ingest endpoint…"):
        try:
            # Prepare multipart form for FIRST file (API currently handles one file at a time)
            first_file = st.session_state.upload_queue[0]
            files = {"file": (first_file["name"], io.BytesIO(first_file["bytes"]), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
            
            form_data = {
                "table_name": first_file["name"].replace(".xlsx", "").replace(".csv", ""),
                "normalize": "true",
                "header_rows": "",  # auto-detect
                "sheet_name": "",   # all sheets
            }
            
            resp = requests.post(f"{API_BASE}/ingest/", files=files, data=form_data, timeout=180)
            
            if resp.status_code != 200:
                st.error(f"❌ API Error {resp.status_code}")
                st.code(resp.text)
                st.stop()
            
            result = resp.json()
            st.session_state.api_ingest_result = result
            
            if result.get("success"):
                st.success(f"✅ Ingest complete: {result['row_count']:,} rows in `{result['table_name']}`")
                st.markdown(_pills(
                    rows=f"{result['row_count']:,}",
                    cols=result["column_count"],
                    tables=len(result["tables_created"]),
                    bridges=len(result["bridge_tables"]),
                ), unsafe_allow_html=True)
            else:
                st.error(f"❌ Ingest failed: {result.get('error', 'Unknown')}")
                
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to API. Is it running at `http://localhost:8000`?")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# Show API ingest result
if st.session_state.api_ingest_result:
    r = st.session_state.api_ingest_result
    
    with st.expander("📋 Full API Response", expanded=True):
        st.json(r)
    
    # Logs
    col_l, col_r = st.columns(2)
    with col_l:
        with st.expander("🧹 Cleaning Log", expanded=False):
            _render_log(r.get("cleaning_log", []))
    with col_r:
        with st.expander("🔀 Normalization Log", expanded=False):
            _render_log(r.get("norm_log", []))
    
    # Bridge tables
    if r.get("bridge_tables"):
        st.markdown("**🔗 Bridge tables created:**")
        for bt in r["bridge_tables"]:
            st.markdown(
                f"<div style='background:#f7f8fa;border:1px solid #e2e6ea;border-radius:8px;padding:.8rem;margin:.4rem 0'>"
                f"<strong>📋 {bt['name']}</strong><br/>"
                f"<small>{bt['row_count']:,} rows | cols: {', '.join(bt['columns'])} | from: <code>{bt['source_col']}</code></small>"
                f"</div>",
                unsafe_allow_html=True,
            )
    
    # Relationships
    if r.get("relationships"):
        with st.expander("🔗 Relationships", expanded=True):
            for rel in r["relationships"]:
                st.caption(f"`{rel['from_table']}.{rel['from_column']}` → `{rel['to_table']}.{rel['to_column']}` ({rel.get('type','')})")

st.divider()

# ── 3. QUICK TEST QUERY ───────────────────────────────────────────────────────
st.markdown("<div style='font-family:monospace;font-size:.8rem;font-weight:600;color:#2563eb;text-transform:uppercase;letter-spacing:.08em;margin:.8rem 0 .3rem'>③ Quick Test Query</div>", unsafe_allow_html=True)

if st.session_state.api_ingest_result and st.session_state.api_ingest_result.get("success"):
    query = st.text_input("Ask a question", placeholder='e.g. "How many rows?" or "Find tickets about login"')
    
    col_opts, col_run = st.columns([3, 1])
    with col_opts:
        force_intent = st.selectbox("Force intent", ["auto", "sql", "python", "vector", "conversational"])
        top_k = st.slider("Vector top-K", 3, 20, 10)
    
    if col_run.button("▶ Run Query") and query:
        with st.spinner("🤔 Querying…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/query/",
                    json={
                        "query": query,
                        "session_id": st.session_state.session_id,
                        "reasoning": True,
                        "debug_mode": True,
                        "force_intent": None if force_intent == "auto" else force_intent,
                        "top_k_vector": top_k,
                    },
                    timeout=90,
                )
                qr = resp.json()
            except Exception as e:
                qr = {"success": False, "error": f"API error: {e}", "intent": "error", "bts_log": []}
        
        # Render result
        if not qr.get("success"):
            st.error(f"❌ {qr.get('error', 'Unknown error')}")
        else:
            st.caption(f"🎯 Intent: `{qr['intent'].upper()}`")
            
            if qr.get("rows"):
                st.dataframe(pd.DataFrame(qr["rows"]), use_container_width=True)
            if qr.get("chart_json"):
                import plotly.io as pio
                fig = pio.from_json(qr["chart_json"])
                st.plotly_chart(fig, use_container_width=True, key=f"quick_{uuid.uuid4().hex[:6]}")
            if qr.get("vector_results"):
                st.dataframe(pd.DataFrame(qr["vector_results"]), use_container_width=True)
            if qr.get("llm_answer"):
                st.info(qr["llm_answer"])
            
            # Debug panel (always visible)
            with st.expander("🔍 Debug Details", expanded=True):
                if qr.get("sql"):
                    st.markdown("**Generated SQL:**")
                    st.code(qr["sql"], language="sql")
                if qr.get("python_code"):
                    st.markdown("**Generated Python:**")
                    st.code(qr["python_code"], language="python")
                if qr.get("context_used"):
                    st.markdown("**Context Injected:**")
                    st.caption(qr["context_used"])
                if qr.get("bts_log"):
                    _render_log(qr["bts_log"], "Execution Log")
else:
    st.info("Complete ingest in step ② to enable queries.")

st.divider()

# ── 4. DOWNLOAD ───────────────────────────────────────────────────────────────
st.markdown("<div style='font-family:monospace;font-size:.8rem;font-weight:600;color:#2563eb;text-transform:uppercase;letter-spacing:.08em;margin:.8rem 0 .3rem'>④ Download</div>", unsafe_allow_html=True)

if st.session_state.api_ingest_result and st.session_state.api_ingest_result.get("success"):
    for tbl in st.session_state.api_ingest_result.get("tables_created", []):
        try:
            resp = requests.post(
                f"{API_BASE}/query/",
                json={"query": f"SELECT * FROM {tbl} LIMIT 100", "session_id": st.session_state.session_id, "force_intent": "sql"},
                timeout=30,
            ).json()
            if resp.get("rows"):
                df = pd.DataFrame(resp["rows"])
                csv = df.to_csv(index=False).encode()
                st.download_button(f"⬇️ {tbl}.csv", data=csv, file_name=f"{tbl}.csv", mime="text/csv", key=f"dl_{tbl}")
        except Exception:
            st.caption(f"⚠️ Could not prepare `{tbl}`")
else:
    st.info("No data loaded yet.")