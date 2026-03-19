# streamlit_app.py
import sys
import io
from pathlib import Path
import os
import pandas as pd
import streamlit as st

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
#sys.path.append(str(Path(__file__).resolve().parent.parent))
"""
ui/streamlit_app.py
====================
Full Streamlit UI:
  Upload CSV / Excel
  Clean & Profile  (metadata/schema_profiler.py)
  Load into DuckDB (storage/duckdb_manager.py)
  Natural language query  (reasoning/query_orchestrator.py)
  Download cleaned data
"""
from hybridtablerag.llm.factory import get_llm
from hybridtablerag.reasoning.sql_generator import LLMSQLGenerator
from hybridtablerag.storage.duckdb_manager import DuckDBManager
from hybridtablerag.metadata.schema_profiler import clean_and_profile
from hybridtablerag.reasoning.query_orchestrator import (
    QueryOrchestrator,
    QueryResult,
    register_cleaned_df,
)

# from hybridtablerag.llm.gemini_client  import GeminiClient  as LLMClient
# from llm.ollama_client  import OllamaClient  as LLMClient
# from llm.openai_client  import OpenAIClient  as LLMClient
from hybridtablerag.llm.azureopenai_client  import AzureOpenAIClient  as LLMClient


#css config
st.set_page_config(page_title="Hybrid Table RAG", layout="wide")
import streamlit as st

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');

:root {
  --bg:#ffffff; --surface:#f5f5f5; --border:#cccccc;
  --ok:#34d399; --warn:#f59e0b; --err:#f87171;
  --text:#111111; --muted:#555555;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Mono', monospace;
}

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  color: var(--text) !important;
}

h1, h2, h3 {
  color: var(--text) !important;
  font-weight: 600;
  font-family: 'IBM Plex Mono', monospace;
}

.stat-row {
  display:flex; gap:.6rem; flex-wrap:wrap; margin:.6rem 0;
}

.stat-pill {
  background:#e0e0e0; 
  border:1px solid var(--border);
  border-radius:6px; padding:.25rem .7rem; 
  font-size:.75rem; color:var(--muted);
}

.stat-pill span {
  color:var(--text); font-weight:700; margin-left:.3rem;
}

.log-block {
  background:#f0f0f0; border:1px solid var(--border); 
  border-radius:8px; padding:1rem; max-height:320px; 
  overflow-y:auto; font-size:.78rem; line-height:1.7;
}

.log-drop { color: var(--err); } 
.log-norm { color: var(--ok); }
.log-flat { color: var(--muted); } 
.log-num { color: var(--warn); }
.log-default { color: var(--muted); }

.stButton > button {
  background:#888888; 
  color:#fff; 
  border:none; 
  border-radius:8px;
  font-family:'IBM Plex Mono', monospace; font-weight:600;
  padding:.5rem 1.4rem; letter-spacing:.02em; transition:opacity .2s;
}

.stButton > button:hover { opacity:.85; }

[data-testid="stFileUploader"] {
  border:2px dashed var(--border) !important; 
  border-radius:10px !important; 
  background: var(--surface) !important;
}

[data-testid="stDataFrame"] {
  border:1px solid var(--border) !important; 
  border-radius:8px; 
  background: var(--bg) !important;
}

</style>
""", unsafe_allow_html=True)

# Session state defaults
_DEFAULTS = {
    "uploaded_df": None,
    "cleaned_df":  None,
    "profile":     None,
    "log":         None,
    "db":          None,
    "orchestrator": None,
    "table_name":  None,
    "query_history": [],
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# Helpers
def _resolve_single(obj: dict, key: str):
    """
    Session state stores profiles/logs/cleaned_dfs as either:
      A) single-sheet: the object itself   e.g. {"columns": {...}}  or  [log lines]
      B) multi-sheet:  {"sheet1": obj, "sheet2": obj}

    For dicts, distinguish by checking for a known top-level key that only
    exists in the real object (not in the sheet wrapper).
    For lists (logs), they are never wrapped — always returned directly.
    """
    if isinstance(obj, list):
        return obj                      # logs are always plain lists
    if key in obj:
        return obj                      # already the real object (single-sheet)
    return list(obj.values())[0]        # multi-sheet wrapper — return first sheet


def _css_log(line: str) -> str:
    l = line.lower()
    if "dropped" in l:     return "log-drop"
    if "normalised" in l:  return "log-norm"
    if "flattened" in l:   return "log-flat"
    if "numeric" in l:     return "log-num"
    return "log-default"


def _render_log(log: list):
    lines = "".join(
        f"<div class='{_css_log(l)}'> {l}</div>" for l in log
    )
    st.markdown(f"<div class='log-block'>{lines}</div>", unsafe_allow_html=True)


def _stat_pills(**kwargs) -> str:
    pills = "".join(
        f"<div class='stat-pill'>{k}<span>{v}</span></div>"
        for k, v in kwargs.items()
    )
    return f"<div class='stat-row'>{pills}</div>"


def _render_query_result(result: QueryResult):
    """Render a QueryResult — dataframe, chart, SQL/code, BTS log."""
    if not result.success:
        st.error(f"{result.error}")
    else:
        badge = "SQL" if result.intent == "sql" else "Python"
        st.markdown(
            _stat_pills(intent=badge, rows=len(result.dataframe) if result.dataframe is not None else 0),
            unsafe_allow_html=True,
        )

        if result.reasoning:
            with st.expander("Reasoning", expanded=False):
                st.markdown(result.reasoning)

        if result.chart is not None:
            st.plotly_chart(result.chart, use_container_width=True)

        if result.dataframe is not None and not result.dataframe.empty:
            st.dataframe(result.dataframe, use_container_width=True)
        elif result.dataframe is not None:
            st.info("Query returned no rows.")

        if result.sql:
            with st.expander("SQL / Code", expanded=False):
                lang = "sql" if result.intent == "sql" else "python"
                st.code(result.sql, language=lang)

    with st.expander("Behind-the-scenes log", expanded=False):
        _render_log(result.bts_log)



#Upload

st.title("Hybrid Table RAG")
st.markdown(
    "<p style='color:#8892a4;font-size:.9rem;margin-top:-.5rem'>"
    "Upload → Clean → Query with natural language</p>",
    unsafe_allow_html=True,
)
st.divider()

st.subheader("Upload file")

col_up, col_hdr = st.columns([3, 1])
with col_hdr:
    header_rows = st.number_input(
        "Header rows", min_value=1, max_value=4, value=1,
        help="Set to 2 for CSVs with a two-row group header (e.g. Ticket / Ticket ID)",
    )
with col_up:
    uploaded_file = st.file_uploader(
        "CSV or Excel", type=["csv", "xlsx"], label_visibility="collapsed"
    )

if uploaded_file:
    try:
        hdr = list(range(header_rows)) if header_rows > 1 else 0
        if uploaded_file.name.endswith(".csv"):
            st.session_state.uploaded_df = pd.read_csv(uploaded_file, header=hdr)
        else:
            sheets = pd.read_excel(uploaded_file, sheet_name=None, header=hdr)
            st.session_state.uploaded_df = sheets
    except Exception as e:
        st.error(f"Could not read file: {e}")

if st.session_state.uploaded_df is not None:
    df_map = (
        st.session_state.uploaded_df
        if isinstance(st.session_state.uploaded_df, dict)
        else {"sheet": st.session_state.uploaded_df}
    )
    with st.expander("Raw preview", expanded=True):
        sel = st.selectbox("Sheet", list(df_map.keys())) if len(df_map) > 1 else list(df_map.keys())[0]
        raw = df_map[sel]
        if isinstance(raw.columns, pd.MultiIndex):
            disp = raw.copy()
            disp.columns = [" ".join(str(i) for i in c if str(i) != "") for c in disp.columns]
        else:
            disp = raw
        st.markdown(_stat_pills(rows=raw.shape[0], cols=raw.shape[1]), unsafe_allow_html=True)
        st.dataframe(disp.head(10), use_container_width=True)

st.divider()



#Clean & Profile

st.subheader("Clean & Profile")

if st.session_state.uploaded_df is None:
    st.info("Upload a file above to enable cleaning.")
else:
    if st.button("Run Cleaner"):
        df_map = (
            st.session_state.uploaded_df
            if isinstance(st.session_state.uploaded_df, dict)
            else {"sheet": st.session_state.uploaded_df}
        )
        cleaned_map, profile_map, log_map = {}, {}, {}
        prog = st.progress(0, text="Cleaning…")
        for i, (sheet, sdf) in enumerate(df_map.items()):
            prog.progress((i + 1) / len(df_map), text=f"Cleaning: {sheet}")
            try:
                res = clean_and_profile(sdf)
                cleaned_map[sheet] = res["cleaned_df"]
                profile_map[sheet] = res["profile"]
                log_map[sheet]     = res["log"]
            except Exception as e:
                st.error(f"Error on '{sheet}': {e}")
        prog.empty()

        if list(cleaned_map.keys()) == ["sheet"]:
            st.session_state.cleaned_df = cleaned_map["sheet"]
            st.session_state.profile    = profile_map["sheet"]
            st.session_state.log        = log_map["sheet"]
        else:
            st.session_state.cleaned_df = cleaned_map
            st.session_state.profile    = profile_map
            st.session_state.log        = log_map

        # Reset DuckDB + orchestrator so they pick up fresh cleaned data
        st.session_state.db           = None
        st.session_state.orchestrator = None
        st.session_state.table_name   = None
        st.success("Cleaning complete!")

if st.session_state.cleaned_df is not None:
    with st.expander("Cleaned data preview", expanded=False):
        cdf = (
            st.session_state.cleaned_df
            if isinstance(st.session_state.cleaned_df, pd.DataFrame)
            else list(st.session_state.cleaned_df.values())[0]
        )
        st.markdown(_stat_pills(rows=cdf.shape[0], cols=cdf.shape[1]), unsafe_allow_html=True)
        st.dataframe(cdf, use_container_width=True)

    with st.expander("Cleaning log", expanded=False):
        log = _resolve_single(st.session_state.log, None)
        _render_log(log)

    with st.expander("Column profile", expanded=False):
        profile = _resolve_single(st.session_state.profile, "columns")
        rows = []
        for col, stats in profile["columns"].items():
            pct = stats["pct_null"]
            rows.append({
                "column":   col,
                "dtype":    stats["dtype"],
                "nulls":    stats["num_nulls"],
                "% null":   f"{pct:.1f}",
                "unique":   stats["num_unique"],
                "samples":  " | ".join(str(v) for v in stats["sample_values"][:3]),
            })
        st.dataframe(pd.DataFrame(rows).set_index("column"), use_container_width=True)

st.divider()


# Load into DuckDB

st.subheader("Load into DuckDB")

if st.session_state.cleaned_df is None:
    st.info("Clean data first.")
else:
    col_tbl, col_load = st.columns([2, 1])
    with col_tbl:
        table_name_input = st.text_input(
            "Table name", value="tickets",
            help="Name used in SQL queries, e.g. SELECT * FROM tickets"
        )
    with col_load:
        st.markdown("<br>", unsafe_allow_html=True)
        load_btn = st.button("Load into DuckDB")

    if load_btn:
        bts: list = []
        try:
            db = DuckDBManager()
            df_to_load = (
                st.session_state.cleaned_df
                if not isinstance(st.session_state.cleaned_df, dict)
                else list(st.session_state.cleaned_df.values())[0]
            )
            register_cleaned_df(db.conn, df_to_load, table_name_input, bts)

            llm         = get_llm("azure-openai", "charlie-gpt-4.1-mini")          
            sql_gen     = LLMSQLGenerator(llm=llm)
            orchestrator = QueryOrchestrator(
                llm=llm,
                conn=db.conn,
                sql_generator=sql_gen,
                default_table=table_name_input,
            )

            st.session_state.db           = db
            st.session_state.orchestrator = orchestrator
            st.session_state.table_name   = table_name_input

            st.success(f"Table '{table_name_input}' loaded. Ready to query.")
            _render_log(bts)
        except Exception as e:
            st.error(f"DuckDB load failed: {e}")
            _render_log(bts)

st.divider()


# ── Natural language query ────────────────────────────────────────────────────
st.subheader("Ask a question")

if st.session_state.orchestrator is None:
    st.info("Load data into DuckDB (step 3) to enable querying.")
else:
    col_q, col_opts = st.columns([4, 1])
    with col_q:
        user_query = st.text_input(
            "Query",
            placeholder="e.g.  Show tickets resolved in under 3 days  |  Plot ticket volume by priority",
            label_visibility="collapsed",
        )
    with col_opts:
        show_reasoning = st.checkbox("Reasoning", value=False)
        force_path = st.selectbox("Path", ["sql", "python", "both"], index=0,
            help="'both' runs SQL and Python in parallel — useful for testing")

    run_query = st.button("Run", disabled=not bool(user_query))

    if run_query and user_query:

        if force_path == "both":
            # Run both paths in parallel and show side by side
            with st.spinner("Running both paths…"):
                result_sql = st.session_state.orchestrator.run(
                    user_query, reasoning=show_reasoning, force_intent="sql"
                )
                result_py = st.session_state.orchestrator.run(
                    user_query, reasoning=show_reasoning, force_intent="python"
                )

            st.session_state.query_history.insert(0, result_sql)  # store SQL result as canonical

            col_sql, col_py = st.columns(2)
            with col_sql:
                st.markdown("SQL path")
                _render_query_result(result_sql)
            with col_py:
                st.markdown("Python path")
                _render_query_result(result_py)

        else:
            with st.spinner("Running…"):
                result = st.session_state.orchestrator.run(
                    user_query,
                    reasoning=show_reasoning,
                    force_intent=None if force_path == "auto" else force_path,
                )

            st.session_state.query_history.insert(0, result)
            _render_query_result(result)

    # Query history
    if len(st.session_state.query_history) > 1:
        with st.expander(f"Query history ({len(st.session_state.query_history)} queries)", expanded=False):
            for i, past in enumerate(st.session_state.query_history[1:], 1):
                st.markdown(f"**{i}.** `{past.user_query}`")
                _render_query_result(past)

st.divider()


# #  Download
# st.subheader(" Download cleaned data")

# if st.session_state.cleaned_df is None:
#     st.info("No cleaned data yet.")
# else:
#     if isinstance(st.session_state.cleaned_df, dict):
#         cols = st.columns(min(len(st.session_state.cleaned_df), 4))
#         for col_w, (sheet, dfc) in zip(cols, st.session_state.cleaned_df.items()):
#             with col_w:
#                 st.download_button(
#                     f"⬇ {sheet}.csv",
#                     data=dfc.to_csv(index=False).encode(),
#                     file_name=f"{sheet}_cleaned.csv",
#                     mime="text/csv",
#                     key=f"dl_{sheet}",
#                 )
#     else:
#         st.download_button(
#             "⬇ Download cleaned CSV",
#             data=st.session_state.cleaned_df.to_csv(index=False).encode(),
#             file_name="cleaned_data.csv",
#             mime="text/csv",
#         )
