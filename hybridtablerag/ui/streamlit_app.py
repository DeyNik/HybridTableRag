"""
ui/streamlit_app.py
====================
Pipeline page: Upload → Clean → Normalize → Load → Download
Calls the FastAPI backend at API_BASE_URL.
Can also run in standalone mode (direct Python imports) when API is unavailable.
"""

import io
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HybridTableRAG",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{--bg:#ffffff;--surface:#f7f8fa;--border:#e2e6ea;--accent:#2563eb;
     --ok:#16a34a;--warn:#b45309;--err:#dc2626;--text:#0f172a;--muted:#64748b;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;
  color:var(--text)!important;font-family:'IBM Plex Sans',sans-serif;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
h1,h2,h3{font-family:'IBM Plex Mono',monospace;letter-spacing:-.02em;}
.step-header{font-family:'IBM Plex Mono',monospace;font-size:.85rem;font-weight:600;
  color:var(--accent);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;}
.stat-row{display:flex;gap:.5rem;flex-wrap:wrap;margin:.5rem 0;}
.stat-pill{background:var(--surface);border:1px solid var(--border);border-radius:6px;
  padding:.2rem .65rem;font-size:.72rem;color:var(--muted);}
.stat-pill span{color:var(--text);font-weight:600;margin-left:.25rem;}
.log-wrap{background:#f8f9fb;border:1px solid var(--border);border-left:3px solid var(--accent);
  border-radius:0 6px 6px 0;padding:.75rem 1rem;max-height:260px;overflow-y:auto;
  font-family:'IBM Plex Mono',monospace;font-size:.73rem;line-height:1.7;}
.log-drop{color:var(--err);}.log-ok{color:var(--ok);}.log-warn{color:var(--warn);}.log-dim{color:var(--muted);}
.bridge-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:.75rem 1rem;margin:.4rem 0;}
.bridge-card .name{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:.85rem;}
.bridge-card .meta{color:var(--muted);font-size:.75rem;}
.status-ok{color:var(--ok);font-weight:600;}.status-err{color:var(--err);font-weight:600;}
.stButton>button{background:var(--accent);color:#fff;border:none;border-radius:6px;
  font-family:'IBM Plex Mono',monospace;font-weight:600;padding:.45rem 1.2rem;}
.stButton>button:hover{opacity:.88;}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:6px;}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "session_id":     None,
    "ingest_result":  None,
    "uploaded_bytes": None,
    "uploaded_name":  None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# Generate persistent session ID
if not st.session_state.session_id:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.session_state["debug_mode"] = st.checkbox(
        "🐛 Debug mode",
        value=st.session_state.get("debug_mode", False),
        help="Show BTS logs, SQL, and intent classification in all views",
    )
    st.caption(f"Session: `{st.session_state.session_id}`")
    st.divider()

    # API health indicator
    st.markdown("**API Status**")
    try:
        h = requests.get(f"{API_BASE}/health/", timeout=2).json()
        status_color = "🟢" if h["status"] == "ok" else "🟡"
        st.markdown(f"{status_color} `{h['status'].upper()}`")
        st.caption(f"DuckDB: {'✅' if h['duckdb'] else '❌'}  LLM: {'✅' if h['llm'] else '❌'}  Vector: {'✅' if h['vector_store'] else '❌'}")
        if h["tables"]:
            st.caption(f"Tables: {', '.join(h['tables'])}")
    except Exception:
        st.markdown("🔴 `OFFLINE`")
        st.caption(f"API not reachable at {API_BASE}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _pills(**kwargs) -> str:
    pills = "".join(
        f"<div class='stat-pill'>{k}<span>{v}</span></div>"
        for k, v in kwargs.items()
    )
    return f"<div class='stat-row'>{pills}</div>"

def _log_class(line: str) -> str:
    l = line.lower()
    if any(k in l for k in ("dropped", "error", "❌", "failed")): return "log-drop"
    if any(k in l for k in ("normalised", "✅", "complete", "registered")): return "log-ok"
    if any(k in l for k in ("warn", "⚠️", "skipped")): return "log-warn"
    return "log-dim"

def _render_log(log: list):
    html = "".join(f"<div class='{_log_class(l)}'>› {l}</div>" for l in log)
    st.markdown(f"<div class='log-wrap'>{html}</div>", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("HybridTableRAG")
st.caption("Upload → Clean → Normalize → Query  |  Switch to **Chat** in the sidebar")
st.divider()


# ── ① Upload ──────────────────────────────────────────────────────────────────
st.markdown("<div class='step-header'>① Upload file</div>", unsafe_allow_html=True)

col_file, col_opts = st.columns([3, 2])

with col_opts:
    table_name = st.text_input("Table name", value="tickets",
        help="Used in DuckDB and SQL queries")
    normalize = st.checkbox("Auto-normalize multi-valued columns", value=True,
        help="Detects semicolon/pipe-separated columns and JSON arrays → bridge tables")
    header_override = st.text_input("Header rows (blank = auto-detect)",
        value="", placeholder="e.g. 0,1  for two-row header")

with col_file:
    uploaded = st.file_uploader(
        "Drop a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

if uploaded:
    content = uploaded.read()
    st.session_state.uploaded_bytes = content
    st.session_state.uploaded_name  = uploaded.name
    # Preview
    try:
        if uploaded.name.lower().endswith(".csv"):
            preview_df = pd.read_csv(io.BytesIO(content), nrows=5)

        else:
            xls = pd.ExcelFile(io.BytesIO(content))
            sheet_names = xls.sheet_names

            selected_sheet = st.selectbox("Select sheet", sheet_names)

            preview_df = pd.read_excel(
                io.BytesIO(content),
                sheet_name=selected_sheet,
                nrows=5
            )

        st.markdown(_pills(file=uploaded.name, size=f"{len(content)//1024}KB"), unsafe_allow_html=True)

        with st.expander("▸ Raw preview (first 5 rows)", expanded=True):
            st.dataframe(preview_df, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not preview: {e}")

st.divider()


# ── ② Ingest (Clean + Normalize + Load) ──────────────────────────────────────
st.markdown("<div class='step-header'>② Clean · Normalize · Load</div>", unsafe_allow_html=True)

if st.session_state.uploaded_bytes is None:
    st.info("Upload a file above to continue.")
else:
    if st.button("🚀 Run Pipeline", type="primary"):
        with st.spinner("Running full pipeline…"):
            try:
                files   = {"file": (st.session_state.uploaded_name, io.BytesIO(st.session_state.uploaded_bytes))}
                data    = {
                    "table_name":  table_name,
                    "normalize":   str(normalize).lower(),
                    "header_rows": header_override.strip() or "",
                }
                resp = requests.post(f"{API_BASE}/ingest/", files=files, data=data, timeout=120)
                result = resp.json()
                st.session_state.ingest_result = result
            except Exception as e:
                st.error(f"API call failed: {e}")
                st.stop()

    if st.session_state.ingest_result:
        r = st.session_state.ingest_result

        if not r["success"]:
            st.error(f"Pipeline failed: {r.get('error', 'Unknown error')}")
        else:
            st.success(f"✅ Loaded {r['row_count']:,} rows into `{r['table_name']}`")
            st.markdown(
                _pills(
                    rows=f"{r['row_count']:,}",
                    cols=r["column_count"],
                    tables=len(r["tables_created"]),
                    bridges=len(r["bridge_tables"]),
                ),
                unsafe_allow_html=True,
            )

            col_l, col_r = st.columns(2)

            with col_l:
                with st.expander("🧹 Cleaning log", expanded=False):
                    _render_log(r.get("cleaning_log", []))

                with st.expander("📐 Normalization log", expanded=False):
                    _render_log(r.get("norm_log", []))

            with col_r:
                with st.expander(f"📊 Column profile ({r['column_count']} columns)", expanded=False):
                    profile_rows = []
                    for col_name, stats in r.get("profile", {}).items():
                        profile_rows.append({
                            "column":     col_name,
                            "dtype":      stats["dtype"],
                            "nulls":      stats["num_nulls"],
                            "% null":     f"{stats['pct_null']:.1f}",
                            "unique":     stats["num_unique"],
                            "multi?":     "⚠️" if stats.get("is_multi_valued") else "",
                            "samples":    " | ".join(str(s) for s in stats["sample_values"][:2]),
                        })
                    st.dataframe(pd.DataFrame(profile_rows).set_index("column"), use_container_width=True)

            # Bridge tables
            if r["bridge_tables"]:
                st.markdown("**Bridge tables created:**")
                for bt in r["bridge_tables"]:
                    st.markdown(
                        f"<div class='bridge-card'>"
                        f"<div class='name'>📎 {bt['name']}</div>"
                        f"<div class='meta'>{bt['row_count']:,} rows · "
                        f"columns: {', '.join(bt['columns'])} · "
                        f"from: <code>{bt['source_col']}</code> (sep: <code>{bt['separator']}</code>)</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Relationships
            if r["relationships"]:
                with st.expander("🔗 Inferred relationships", expanded=False):
                    for rel in r["relationships"]:
                        st.caption(
                            f"`{rel['from_table']}.{rel['from_column']}` → "
                            f"`{rel['to_table']}.{rel['to_column']}` ({rel.get('type','')})"
                        )

st.divider()


# ── ③ Quick query (test without going to Chat page) ───────────────────────────
st.markdown("<div class='step-header'>③ Quick test query</div>", unsafe_allow_html=True)

if st.session_state.ingest_result and st.session_state.ingest_result.get("success"):
    q = st.text_input("Ask a quick question", placeholder="e.g. how many tickets are high priority?")
    col_run, col_opts2 = st.columns([2, 2])
    with col_opts2:
        q_reasoning = st.checkbox("Reasoning mode")
        q_debug     = st.checkbox("Debug mode", value=st.session_state.get("debug_mode", False))

    if st.button("▶ Run query") and q:
        with st.spinner("Querying…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/query/",
                    json={
                        "query":      q,
                        "session_id": st.session_state.session_id,
                        "reasoning":  q_reasoning,
                        "debug_mode": q_debug,
                    },
                    timeout=60,
                )
                qr = resp.json()
            except Exception as e:
                st.error(f"Query failed: {e}")
                qr = None

        if qr:
            if not qr["success"]:
                st.error(qr.get("error", "Unknown error"))
            else:
                st.caption(f"Intent: `{qr['intent']}`")
                if qr.get("rows"):
                    st.dataframe(pd.DataFrame(qr["rows"]), use_container_width=True)
                if qr.get("chart_json"):
                    import plotly.io as pio
                    fig = pio.from_json(qr["chart_json"])
                    st.plotly_chart(fig, use_container_width=True, key="quick_chart")
                if qr.get("llm_answer"):
                    st.info(qr["llm_answer"])
                if q_debug and qr.get("bts_log"):
                    with st.expander("🪵 BTS log"):
                        _render_log(qr["bts_log"])
                if qr.get("sql") and q_debug:
                    with st.expander("🗄️ SQL"):
                        st.code(qr["sql"], language="sql")
else:
    st.info("Load data in step ② to enable test queries.")

st.divider()


# ── ④ Download ────────────────────────────────────────────────────────────────
st.markdown("<div class='step-header'>④ Download</div>", unsafe_allow_html=True)

if st.session_state.ingest_result and st.session_state.ingest_result.get("success"):
    r = st.session_state.ingest_result
    st.caption(f"Tables available: {', '.join(r['tables_created'])}")

    col_dl = st.columns(min(len(r["tables_created"]), 4))
    for i, tbl in enumerate(r["tables_created"]):
        with col_dl[i % 4]:
            try:
                rows_resp = requests.post(
                    f"{API_BASE}/query/",
                    json={"query": f"SELECT * FROM {tbl}",
                          "session_id": st.session_state.session_id,
                          "force_intent": "sql"},
                    timeout=30,
                ).json()
                if rows_resp.get("rows"):
                    csv = pd.DataFrame(rows_resp["rows"]).to_csv(index=False).encode()
                    st.download_button(f"⬇ {tbl}.csv", data=csv,
                                       file_name=f"{tbl}_cleaned.csv",
                                       mime="text/csv", key=f"dl_{tbl}")
            except Exception:
                st.caption(f"Could not prepare {tbl}")
else:
    st.info("No data loaded yet.")