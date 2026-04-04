"""
ui/pages/01_chat.py
====================
Full context-aware chat interface.
Debug-by-default (no toggles)
Persistent history via API + session state
Auto-chart rendering with unique keys
Intent badges + inline result previews
Error visibility at every step
"""

import io
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

#  Page config 
st.set_page_config(
    page_title="Chat — HybridTableRAG",
    layout="wide",
    page_icon="",
    initial_sidebar_state="expanded",
)

#  CSS: Chat bubbles, intent badges, debug panels 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{--bg:#ffffff;--surface:#f7f8fa;--border:#e2e6ea;--accent:#2563eb;
     --ok:#16a34a;--warn:#b45309;--err:#dc2626;--text:#0f172a;--muted:#64748b;
     --bubble-user:#1e3a5f;--bubble-ai:#f0f4ff;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;
  color:var(--text)!important;font-family:'IBM Plex Sans',sans-serif;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}

/* Chat bubbles */
.bubble-wrap{display:flex;margin:1rem 0;align-items:flex-start;gap:.8rem;}
.bubble-wrap.user{flex-direction:row-reverse;}
.avatar{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:.7rem;font-weight:700;flex-shrink:0;
  font-family:'IBM Plex Mono',monospace;}
.avatar.user{background:var(--bubble-user);color:#fff;}
.avatar.ai{background:#e2e8f0;color:var(--muted);border:1px solid var(--border);}
.bubble{max-width:80%;padding:.7rem 1rem;border-radius:12px;font-size:.9rem;line-height:1.6;}
.bubble.user{background:var(--bubble-user);color:#fff;border-bottom-right-radius:4px;}
.bubble.ai{background:var(--bubble-ai);border:1px solid var(--border);border-bottom-left-radius:4px;}

/* Intent badges */
.intent-badge{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:.65rem;
  font-weight:700;padding:.15rem .6rem;border-radius:5px;margin-bottom:.4rem;letter-spacing:.04em;}
.intent-sql{background:#dbeafe;color:#1d4ed8;}
.intent-python{background:#dcfce7;color:#15803d;}
.intent-vector{background:#fef3c7;color:#92400e;}
.intent-conv{background:#f3e8ff;color:#6b21a8;}
.intent-err{background:#fee2e2;color:#991b1b;}

/* Debug panels */
.log-panel{background:#f8f9fb;border:1px solid var(--border);border-left:3px solid var(--accent);
  border-radius:0 6px 6px 0;padding:.7rem 1rem;max-height:240px;overflow-y:auto;
  font-family:'IBM Plex Mono',monospace;font-size:.72rem;line-height:1.65;}
.log-drop{color:var(--err);}.log-ok{color:var(--ok);}.log-warn{color:var(--warn);}.log-dim{color:var(--muted);}
.context-pill{background:#fffbeb;border:1px solid #fde68a;border-radius:6px;
  padding:.45rem .75rem;font-size:.75rem;color:#78350f;margin:.4rem 0;}

/* Buttons & inputs */
.stButton>button{background:var(--accent);color:#fff;border:none;border-radius:6px;
  font-family:'IBM Plex Mono',monospace;font-weight:600;padding:.45rem 1.2rem;}
.stButton>button:hover{opacity:.88;}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:6px;}
.debug-always{display:inline-block;background:#dbeafe;color:#1d4ed8;
  font-family:'IBM Plex Mono',monospace;font-size:.65rem;font-weight:700;
  padding:.1rem .45rem;border-radius:4px;margin-left:.4rem;}
</style>
""", unsafe_allow_html=True)

#  Session state 
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "chat_messages" not in st.session_state:
    # Load from API on first visit
    try:
        hist = requests.get(f"{API_BASE}/history/{st.session_state.session_id}", timeout=2).json()
        st.session_state.chat_messages = [
            {"role": "user", "text": t["user_query"]}
            if t.get("intent") != "conversational" or "answer" not in t.get("result_summary", "").lower()
            else {"role": "ai", "result": {"success": True, "intent": "conversational", "llm_answer": t.get("result_summary", "")}}
            for t in hist.get("turns", [])
        ]
    except:
        st.session_state.chat_messages = []

# DEBUG IS ALWAYS ON — no checkbox
DEBUG_MODE = True

#  Sidebar: Health + Settings 
with st.sidebar:
    st.markdown("### Chat Settings")
    
    # Always-visible debug badge
    st.markdown("<span class='debug-always'>DEBUG MODE: ON</span>", unsafe_allow_html=True)
    
    reasoning_mode = st.checkbox("Reasoning mode", value=True,
        help="Ask LLM to explain its query derivation step by step")
    force_path = st.selectbox("Force intent", ["auto", "sql", "python", "vector", "conversational"])
    top_k = st.slider("Vector top-K", 3, 20, 10)
    
    st.divider()
    st.caption(f"Session: `{st.session_state.session_id}`")
    
    # Clear chat button
    if st.button("🗑 Clear chat history"):
        st.session_state.chat_messages = []
        try:
            requests.delete(f"{API_BASE}/history/{st.session_state.session_id}", timeout=5)
        except:
            pass
        st.rerun()
    
    st.divider()
    
    # Live health check
    try:
        health = requests.get(f"{API_BASE}/health/", timeout=2).json()
        non_sys = [t for t in health.get("tables", []) if t != "chat_history"]
        if non_sys:
            st.markdown("**Active tables:**")
            for t in non_sys:
                cnt = health.get("row_counts", {}).get(t, "?")
                st.caption(f"`{t}` — {cnt:,} rows" if isinstance(cnt, int) else f"`{t}`")
        else:
            st.warning("No data loaded.\nGo to the **main page** to upload a file.")
    except Exception as e:
        st.caption(f"API status: unreachable ({e})")

#  Helpers 
def _log_class(line: str) -> str:
    l = line.lower()
    if any(k in l for k in ("error", "failed", "exception")): return "log-drop"
    if any(k in l for k in ("complete", "returned", "succeeded")): return "log-ok"
    if any(k in l for k in ("warn", "retry", "0 rows")): return "log-warn"
    return "log-dim"

def _render_log(log: List[str]):
    if not log:
        st.caption("No execution log.")
        return
    html = "".join(f"<div class='{_log_class(l)}'>› {l}</div>" for l in log)
    st.markdown(f"<div class='log-panel'>{html}</div>", unsafe_allow_html=True)

def _intent_badge(intent: str) -> str:
    mapping = {
        "sql": ('<span class="intent-badge intent-sql">SQL</span>'),
        "python": ('<span class="intent-badge intent-python">PYTHON</span>'),
        "vector": ('<span class="intent-badge intent-vector">VECTOR</span>'),
        "conversational": ('<span class="intent-badge intent-conv">CHAT</span>'),
    }
    badge, icon = mapping.get(intent, ('<span class="intent-badge intent-err">?</span>'))
    return badge, icon

def _result_summary(qr: Dict) -> str:
    if not qr.get("success"):
        return f"Error: {qr.get('error', 'Unknown')}"
    if qr.get("llm_answer"):
        words = qr["llm_answer"].split()
        return " ".join(words[:30]) + ("…" if len(words) > 30 else "")
    if qr.get("rows"):
        return f"{qr.get('row_count', len(qr['rows'])):,} row(s) returned"
    if qr.get("chart_json"):
        return "Chart generated"
    if qr.get("vector_results"):
        return f"{len(qr['vector_results'])} similar result(s)"
    return "Done."

def _render_debug_panel(qr: Dict, msg_idx: int):
    """5-tab debug panel — always expanded in debug mode."""
    with st.expander("How did I get this? (Debug)", expanded=True):
        tabs = st.tabs(["Result", "SQL / Code", "Reasoning", "Context", "Execution Log"])
        
        with tabs[0]:
            if qr.get("chart_json"):
                import plotly.io as pio
                fig = pio.from_json(qr["chart_json"])
                # UNIQUE KEY to avoid Streamlit duplicate element error
                st.plotly_chart(fig, use_container_width=True, key=f"chart_dbg_{msg_idx}_{uuid.uuid4().hex[:6]}")
            if qr.get("rows"):
                st.dataframe(pd.DataFrame(qr["rows"]), use_container_width=True)
            elif qr.get("python_rows"):
                st.dataframe(pd.DataFrame(qr["python_rows"]), use_container_width=True)
            elif qr.get("vector_results"):
                st.dataframe(pd.DataFrame(qr["vector_results"]), use_container_width=True)
            elif qr.get("llm_answer"):
                st.markdown(qr["llm_answer"])
            elif not qr.get("success"):
                st.error(qr.get("error", "Unknown error"))
            else:
                st.info("No result data returned.")
        
        with tabs[1]:
            if qr.get("sql") and qr.get("intent") == "sql":
                st.markdown("**Generated SQL:**")
                st.code(qr["sql"], language="sql")
            elif qr.get("python_code"):
                st.markdown("**Generated Python:**")
                st.code(qr["python_code"], language="python")
            else:
                st.caption("No SQL or code generated for this intent.")
        
        with tabs[2]:
            if qr.get("reasoning"):
                st.markdown(qr["reasoning"])
            else:
                st.caption("Enable **Reasoning mode** in sidebar to capture chain-of-thought.")
        
        with tabs[3]:
            ctx = qr.get("context_used")
            if ctx:
                st.markdown(f"<div class='context-pill'>📌 Context injected:<br/>{ctx}</div>", unsafe_allow_html=True)
            else:
                st.caption("No prior context was injected for this turn.")
        
        with tabs[4]:
            log = qr.get("bts_log", [])
            if log:
                _render_log(log)
            else:
                st.caption("No execution log captured.")

def _render_message(msg: Dict, idx: int):
    """Render one conversation turn with full debug visibility."""
    if msg["role"] == "user":
        st.markdown(
            f"<div class='bubble-wrap user'>"
            f"<div class='avatar user'>YOU</div>"
            f"<div class='bubble user'>{msg['text']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return
    
    # AI message
    qr = msg["result"]
    badge_html, icon = _intent_badge(qr.get("intent", "sql"))
    summary = _result_summary(qr)
    
    st.markdown(
        f"<div class='bubble-wrap ai'>"
        f"<div class='avatar ai'>AI</div>"
        f"<div class='bubble ai'>{badge_html}<br/>{summary}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    
    # Inline result preview (main view)
    if qr.get("chart_json") and qr.get("intent") == "python":
        import plotly.io as pio
        fig = pio.from_json(qr["chart_json"])
        st.plotly_chart(fig, use_container_width=True, key=f"chart_main_{idx}_{uuid.uuid4().hex[:6]}")
    
    if qr.get("rows") and qr.get("intent") == "sql":
        st.dataframe(pd.DataFrame(qr["rows"]).head(20), use_container_width=True)
    
    if qr.get("vector_results") and qr.get("intent") == "vector":
        st.dataframe(pd.DataFrame(qr["vector_results"]).head(10), use_container_width=True)
    
    if qr.get("llm_answer") and qr.get("intent") == "conversational":
        st.markdown(qr["llm_answer"])
    
    if not qr.get("success"):
        st.error(qr.get("error", "Unknown error"))
    
    # ALWAYS SHOW DEBUG PANEL (debug-by-default)
    _render_debug_panel(qr, idx)

# Header
st.markdown(
    "<h2 style='margin:0;padding-top:1rem'>💬 Chat with your data</h2>"
    "<p style='color:#64748b;font-size:.85rem;margin-top:.2rem'>"
    "Ask anything — SQL, charts, semantic search, or follow-ups. Debug info always visible.</p>",
    unsafe_allow_html=True,
)

# Check data is loaded
data_ready = False
try:
    h = requests.get(f"{API_BASE}/health/", timeout=2).json()
    non_sys = [t for t in h.get("tables", []) if t != "chat_history"]
    data_ready = bool(non_sys)
except:
    pass

if not data_ready:
    st.warning("No data loaded. Go to the **main page**, upload a file, and click **Run Pipeline**.")
    st.stop()

st.divider()

# Chat history
if not st.session_state.chat_messages:
    st.markdown(
        "<p style='text-align:center;color:#94a3b8;padding:3rem 0'>No messages yet — ask something below ↓</p>",
        unsafe_allow_html=True,
    )
else:
    for i, msg in enumerate(st.session_state.chat_messages):
        _render_message(msg, i)

st.divider()

# Input bar
col_q, col_send = st.columns([5, 1])
with col_q:
    user_input = st.text_input(
        "Ask a question",
        placeholder='e.g. "How many High priority tickets?" · "Plot volume by department" · "Find tickets about login" · "What did I ask earlier?"',
        label_visibility="collapsed",
        key="chat_input",
    )
with col_send:
    send_btn = st.button("Send", type="primary", use_container_width=True)

# Suggested questions
st.caption("Try: "
    "`How many tickets per priority?` | "
    "`Plot ticket volume over time` | "
    "`Find tickets about system failures` | "
    "`What did I ask earlier?`")

# ── Handle submission ─────────────────────────────────────────────────────────
if send_btn and user_input.strip():
    query = user_input.strip()
    
    # Add user message
    st.session_state.chat_messages.append({"role": "user", "text": query})
    
    with st.spinner("Thinking…"):
        try:
            resp = requests.post(
                f"{API_BASE}/query/",
                json={
                    "query": query,
                    "session_id": st.session_state.session_id,
                    "reasoning": reasoning_mode,
                    "debug_mode": DEBUG_MODE,  # always True
                    "force_intent": None if force_path == "auto" else force_path,
                    "top_k_vector": top_k,
                },
                timeout=90,
            )
            qr = resp.json()
        except Exception as e:
            qr = {"success": False, "intent": "error", "error": f"API error: {e}", "bts_log": [f"Request failed: {e}"]}
    
    # Add AI response
    st.session_state.chat_messages.append({"role": "ai", "result": qr})
    
    # Auto-scroll to bottom via rerun
    st.rerun()