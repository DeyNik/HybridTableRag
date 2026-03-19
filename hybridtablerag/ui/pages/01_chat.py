"""
ui/pages/01_chat.py
====================
Chat interface for Hybrid Table RAG.

- Reads orchestrator + table from st.session_state set by streamlit_app.py
- Displays conversation as chat bubbles
- Each assistant message has a collapsible "How did I get this?" button
  showing the SQL/Python code, reasoning chain, and BTS execution log
- If DuckDB isn't loaded yet, shows a friendly redirect prompt
"""

import sys
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ── path ─────────────────────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from hybridtablerag.reasoning.query_orchestrator import QueryResult

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chat — Hybrid Table RAG", layout="wide", page_icon="")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --bg: #ffffff;
  --surface: #f7f7f8;
  --surface2: #efefef;
  --border: #e0e0e0;
  --accent: #1a1a2e;
  --accent-soft: #f0f0ff;
  --ok: #16a34a;
  --warn: #b45309;
  --err: #dc2626;
  --text: #111111;
  --muted: #6b7280;
  --bubble-user: #1a1a2e;
  --bubble-user-text: #ffffff;
  --bubble-ai: #f7f7f8;
  --bubble-ai-text: #111111;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Sans', sans-serif;
}

[data-testid="stSidebar"] {
  background: var(--surface) !important;
}

/* ── page title ── */
.chat-header {
  padding: 1.5rem 0 0.5rem;
  border-bottom: 2px solid var(--border);
  margin-bottom: 1.5rem;
}
.chat-header h1 {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--accent);
  margin: 0;
  letter-spacing: -0.02em;
}
.chat-header p {
  font-size: 0.82rem;
  color: var(--muted);
  margin: 0.3rem 0 0;
}

/* ── chat bubbles ── */
.bubble-wrap {
  display: flex;
  margin-bottom: 1.2rem;
  align-items: flex-start;
  gap: 0.75rem;
}
.bubble-wrap.user  { flex-direction: row-reverse; }
.bubble-wrap.ai    { flex-direction: row; }

.avatar {
  width: 32px; height: 32px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.75rem; font-weight: 700;
  flex-shrink: 0;
  font-family: 'IBM Plex Mono', monospace;
}
.avatar.user { background: var(--accent); color: #fff; }
.avatar.ai   { background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }

.bubble {
  max-width: 72%;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  font-size: 0.88rem;
  line-height: 1.6;
}
.bubble.user {
  background: var(--bubble-user);
  color: var(--bubble-user-text);
  border-bottom-right-radius: 3px;
}
.bubble.ai {
  background: var(--bubble-ai);
  color: var(--bubble-ai-text);
  border: 1px solid var(--border);
  border-bottom-left-radius: 3px;
}

/* ── intent badge ── */
.intent-badge {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  font-weight: 600;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  margin-bottom: 0.4rem;
  letter-spacing: 0.04em;
}
.intent-sql    { background: #dbeafe; color: #1d4ed8; }
.intent-python { background: #dcfce7; color: #15803d; }
.intent-err    { background: #fee2e2; color: #b91c1c; }

/* ── reasoning panel ── */
.reasoning-panel {
  background: #fafafa;
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 0 8px 8px 0;
  padding: 0.9rem 1rem;
  font-size: 0.8rem;
  color: var(--muted);
  margin-top: 0.6rem;
  line-height: 1.65;
}
.reasoning-panel strong { color: var(--text); }

/* ── log block ── */
.log-mini {
  background: #f9f9f9;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.6rem 0.8rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  max-height: 220px;
  overflow-y: auto;
  line-height: 1.7;
  margin-top: 0.5rem;
}
.ll-drop { color: var(--err); }
.ll-ok   { color: var(--ok); }
.ll-flat { color: #6366f1; }
.ll-warn { color: var(--warn); }
.ll-dim  { color: #9ca3af; }

/* ── input bar ── */
.input-bar-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.78rem;
  color: var(--muted);
  margin-bottom: 0.3rem;
}

/* ── no-data state ── */
.no-data-box {
  background: var(--surface);
  border: 1px dashed var(--border);
  border-radius: 10px;
  padding: 2.5rem 2rem;
  text-align: center;
  color: var(--muted);
  font-size: 0.88rem;
  margin-top: 2rem;
}
.no-data-box strong { color: var(--text); font-size: 1rem; display: block; margin-bottom: 0.5rem; }

/* hide Streamlit chrome on chat page */
[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session state for chat history
# ──────────────────────────────────────────────────────────────────────────────
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []   # list of {"role": "user"|"ai", "result": QueryResult|str}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log_class(line: str) -> str:
    l = line.lower()
    if any(k in l for k in ("dropped", "error")):  return "ll-drop"
    if any(k in l for k in ("normalised", "complete")): return "ll-ok"
    if any(k in l for k in ("flattened", "intent", "routed")): return "ll-flat"
    if any(k in l for k in ("numeric", "warn")):   return "ll-warn"
    return "ll-dim"


def _render_log_mini(log: list[str]):
    html = "".join(
        f"<div class='{_log_class(l)}'>› {l}</div>" for l in log
    )
    st.markdown(f"<div class='log-mini'>{html}</div>", unsafe_allow_html=True)


def _summary_text(result: QueryResult) -> str:
    """One-line plain-text summary of the result for the bubble."""
    if not result.success:
        return f"Something went wrong: {result.error}"
    if result.dataframe is not None:
        r, c = result.dataframe.shape
        noun = "row" if r == 1 else "rows"
        return f"Done — {r} {noun}, {c} columns returned."
    return "Done."


def _render_reasoning_panel(result: QueryResult, uid: str = ''):
    """The expandable 'How did I get this?' panel under each AI bubble."""
    with st.expander("How did I get this?", expanded=False):
        tab_labels = ["Result", "SQL / Code", "Reasoning", "Execution log"]
        tabs = st.tabs(tab_labels)

        # Tab 0 — full dataframe
        with tabs[0]:
            if result.chart is not None:
                st.plotly_chart(result.chart, use_container_width=True, key=f'chart_panel_{uid}')
            if result.dataframe is not None and not result.dataframe.empty:
                st.dataframe(result.dataframe, use_container_width=True)
            elif result.dataframe is not None:
                st.info("Query returned no rows.")
            elif not result.success:
                st.error(result.error)

        # Tab 1 — SQL or Python code
        with tabs[1]:
            if result.sql:
                lang = "sql" if result.intent == "sql" else "python"
                st.code(result.sql, language=lang)
            else:
                st.info("No SQL or code was generated.")

        # Tab 2 — LLM reasoning chain
        with tabs[2]:
            if result.reasoning:
                st.markdown(
                    f"<div class='reasoning-panel'>{result.reasoning}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(
                    "Reasoning not captured. Re-run the query with "
                    "**Reasoning mode** enabled (toggle in the input bar)."
                )

        # Tab 3 — BTS execution log
        with tabs[3]:
            if result.bts_log:
                _render_log_mini(result.bts_log)
            else:
                st.caption("No log available.")


def _render_message(msg: dict, idx: int):
    """Render one chat turn — user, single AI, or dual SQL+Python AI."""
    role = msg["role"]

    if role == "user":
        st.markdown(
            f"<div class='bubble-wrap user'>"
            f"  <div class='avatar user'>YOU</div>"
            f"  <div class='bubble user'>{msg['text']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    elif role == "ai_dual":
        # ── Both SQL + Python results side by side ────────────────────────
        result_sql: QueryResult = msg["result_sql"]
        result_py:  QueryResult = msg["result_py"]

        # Single AI bubble summarising both
        sql_rows = len(result_sql.dataframe) if result_sql.dataframe is not None else 0
        has_chart = result_py.chart is not None
        summary = (
            f"SQL returned {sql_rows} rows · "
            f"Python {'generated a chart' if has_chart else 'computed analysis'}"
        )
        st.markdown(
            f"<div class='bubble-wrap ai'>"
            f"  <div class='avatar ai'>AI</div>"
            f"  <div class='bubble ai'>"
            f"    <span class='intent-badge intent-sql'>SQL</span>&nbsp;"
            f"    <span class='intent-badge intent-python'>PYTHON</span><br>"
            f"    {summary}"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Side-by-side panels
        col_sql, col_py = st.columns(2)

        with col_sql:
            st.markdown("##### SQL answer")
            if not result_sql.success:
                st.error(result_sql.error)
            else:
                if result_sql.dataframe is not None and not result_sql.dataframe.empty:
                    st.dataframe(result_sql.dataframe, use_container_width=True)
                else:
                    st.info("No rows returned.")
            with st.expander("SQL · How did I get this?", expanded=False):
                tabs = st.tabs(["SQL", "Reasoning", "Log"])
                with tabs[0]:
                    if result_sql.sql:
                        st.code(result_sql.sql, language="sql")
                with tabs[1]:
                    if result_sql.reasoning:
                        st.markdown(result_sql.reasoning)
                    else:
                        st.caption("Enable Reasoning mode to see chain-of-thought.")
                with tabs[2]:
                    _render_log_mini(result_sql.bts_log)

        with col_py:
            st.markdown("##### Python analysis")
            if not result_py.success:
                st.error(result_py.error)
            else:
                if result_py.chart is not None:
                    st.plotly_chart(result_py.chart, use_container_width=True, key=f'chart_dual_{idx}')
                if result_py.dataframe is not None and not result_py.dataframe.empty:
                    st.dataframe(result_py.dataframe, use_container_width=True)
                elif result_py.chart is None:
                    st.info("No output produced.")
            with st.expander("Python · How did I get this?", expanded=False):
                tabs = st.tabs(["Code", "Log"])
                with tabs[0]:
                    if result_py.sql:
                        code_block = result_py.sql.replace("-- Python path (SQL pre-filter + pandas/plotly)\n", "")
                        st.code(code_block, language="python")
                with tabs[1]:
                    _render_log_mini(result_py.bts_log)

    else:  # single ai result
        result: QueryResult = msg["result"]

        if not result.success:
            badge = "<span class='intent-badge intent-err'>ERROR</span>"
        elif result.intent == "sql":
            badge = "<span class='intent-badge intent-sql'>SQL</span>"
        else:
            badge = "<span class='intent-badge intent-python'>PYTHON</span>"

        summary = _summary_text(result)

        st.markdown(
            f"<div class='bubble-wrap ai'>"
            f"  <div class='avatar ai'>AI</div>"
            f"  <div class='bubble ai'>"
            f"    {badge}<br>{summary}"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Reasoning panel lives outside the bubble (full-width)
        _render_reasoning_panel(result, uid=str(idx))


# ──────────────────────────────────────────────────────────────────────────────
# Page header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='chat-header'>"
    "<h1>Chat with your data</h1>"
    "<p>Ask anything about your loaded table in plain English.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Guard: require DuckDB to be loaded
# ──────────────────────────────────────────────────────────────────────────────
orchestrator = st.session_state.get("orchestrator")
table_name   = st.session_state.get("table_name", "—")

if orchestrator is None:
    st.markdown(
        "<div class='no-data-box'>"
        "<strong>No data loaded yet</strong>"
        "Go to the <b>main page</b>, upload a file, clean it, "
        "and click <em>Load into DuckDB</em> before using the chat."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Active table indicator + clear button
# ──────────────────────────────────────────────────────────────────────────────
col_info, col_clear = st.columns([5, 1])
with col_info:
    cleaned = st.session_state.get("cleaned_df")
    if cleaned is not None:
        df_ref = cleaned if isinstance(cleaned, pd.DataFrame) else list(cleaned.values())[0]
        r, c = df_ref.shape
        st.caption(f"Active table: **{table_name}** — {r} rows × {c} columns")
    else:
        st.caption(f"Active table: **{table_name}**")

with col_clear:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Chat history
# ──────────────────────────────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_messages:
        st.markdown(
            "<p style='color:#9ca3af;font-size:.85rem;text-align:center;"
            "padding:2rem 0'>No messages yet — ask something below ↓</p>",
            unsafe_allow_html=True,
        )
    else:
        for idx, msg in enumerate(st.session_state.chat_messages):
            _render_message(msg, idx)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Input bar
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='input-bar-label'>Ask a question about your data</div>", unsafe_allow_html=True)

col_input, col_send = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        "user_input",
        placeholder='e.g. "What is the correlation between priority and resolution time?" or "Plot ticket trends"',
        label_visibility="collapsed",
        key="chat_input",
    )
with col_send:
    send = st.button("Send", use_container_width=True, type="primary")

# Options row
col_r, col_p, _ = st.columns([1, 1, 3])
with col_r:
    reasoning_mode = st.checkbox("Reasoning mode", value=False,
        help="Ask the LLM to explain its SQL derivation step by step")
with col_p:
    force_path = st.selectbox(
        "Path",
        ["auto (sql + python)", "sql only", "python only"],
        help=(
            "auto — runs both SQL and Python, shows answers side by side\n"
            "sql only — structured queries, counts, filters\n"
            "python only — analytics, stats, numpy, charts"
        ),
    )

# ──────────────────────────────────────────────────────────────────────────────
# Handle submission
# ──────────────────────────────────────────────────────────────────────────────
if send and user_input.strip():
    query = user_input.strip()

    st.session_state.chat_messages.append({"role": "user", "text": query})

    auto_mode = force_path.startswith("auto")

    if auto_mode:
        # Run both paths — show SQL and Python side by side
        with st.spinner("Running SQL + Python analysis…"):
            result_sql = orchestrator.run(
                query, reasoning=reasoning_mode, force_intent="sql"
            )
            result_py = orchestrator.run(
                query, reasoning=reasoning_mode, force_intent="python"
            )
        st.session_state.chat_messages.append({
            "role":       "ai_dual",
            "result_sql": result_sql,
            "result_py":  result_py,
        })

    else:
        intent = "sql" if force_path.startswith("sql") else "python"
        with st.spinner(f"Running {intent} path…"):
            result = orchestrator.run(
                query, reasoning=reasoning_mode, force_intent=intent
            )
        st.session_state.chat_messages.append({"role": "ai", "result": result})

    st.rerun()