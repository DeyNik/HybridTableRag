"""
ui/pages/01_chat.py
====================
Full context-aware chat interface.
Features:
  - Four intent types: sql / python / vector / conversational
  - Debug mode toggle (shows SQL, BTS log, intent, context injected)
  - Persistent conversation history via API
  - "How did I get this?" expander with 5 tabs
  - Unique keys on all plotly charts (no duplicate element errors)
  - Clear chat wipes both UI state and DuckDB history
"""

import io
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat — HybridTableRAG",
    layout="wide",
    page_icon="💬",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{--bg:#ffffff;--surface:#f7f8fa;--border:#e2e6ea;--accent:#2563eb;
     --ok:#16a34a;--warn:#b45309;--err:#dc2626;--text:#0f172a;--muted:#64748b;
     --bubble-user:#1e3a5f;--bubble-ai:#f0f4ff;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;
  color:var(--text)!important;font-family:'IBM Plex Sans',sans-serif;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}

/* bubbles */
.bw{display:flex;margin-bottom:1rem;align-items:flex-start;gap:.6rem;}
.bw.user{flex-direction:row-reverse;}
.av{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:.65rem;font-weight:700;flex-shrink:0;
  font-family:'IBM Plex Mono',monospace;}
.av.user{background:var(--bubble-user);color:#fff;}
.av.ai{background:#e2e8f0;color:var(--muted);border:1px solid var(--border);}
.bb{max-width:75%;padding:.65rem .9rem;border-radius:10px;font-size:.86rem;line-height:1.6;}
.bb.user{background:var(--bubble-user);color:#fff;border-bottom-right-radius:2px;}
.bb.ai{background:var(--bubble-ai);border:1px solid var(--border);border-bottom-left-radius:2px;}

/* intent badges */
.ib{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:.65rem;
  font-weight:700;padding:.12rem .5rem;border-radius:4px;margin-bottom:.35rem;letter-spacing:.04em;}
.ib-sql{background:#dbeafe;color:#1d4ed8;}
.ib-python{background:#dcfce7;color:#15803d;}
.ib-vector{background:#fef3c7;color:#92400e;}
.ib-conv{background:#f3e8ff;color:#6b21a8;}
.ib-err{background:#fee2e2;color:#991b1b;}

/* log */
.lw{background:#f8f9fb;border:1px solid var(--border);border-left:3px solid var(--accent);
  border-radius:0 6px 6px 0;padding:.6rem .8rem;max-height:220px;overflow-y:auto;
  font-family:'IBM Plex Mono',monospace;font-size:.7rem;line-height:1.65;}
.ll-drop{color:var(--err);}.ll-ok{color:var(--ok);}.ll-warn{color:var(--warn);}.ll-dim{color:var(--muted);}

/* context pill */
.ctx-box{background:#fffbeb;border:1px solid #fde68a;border-radius:6px;
  padding:.4rem .7rem;font-size:.73rem;color:#78350f;margin:.3rem 0;}

.stButton>button{background:var(--accent);color:#fff;border:none;border-radius:6px;
  font-family:'IBM Plex Mono',monospace;font-weight:600;padding:.4rem 1.1rem;}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:6px;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 Chat settings")
    st.session_state.debug_mode = st.checkbox(
        "🐛 Debug mode",
        value=st.session_state.debug_mode,
        help="Shows SQL, BTS log, intent, and context injected",
    )
    reasoning_mode = st.checkbox("🧠 Reasoning mode", value=False,
        help="Ask LLM to explain its query derivation step by step")
    force_path = st.selectbox("Force path", ["auto", "sql", "python", "vector", "conversational"])
    top_k = st.slider("Vector top-K", 3, 20, 10)

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id}`")

    if st.button("🗑 Clear chat"):
        st.session_state.chat_messages = []
        try:
            requests.delete(
                f"{API_BASE}/history/{st.session_state.session_id}",
                timeout=5
            )
        except Exception:
            pass
        st.rerun()

    st.divider()

    # Active table info
    try:
        h = requests.get(f"{API_BASE}/health/", timeout=2).json()
        non_sys = [t for t in h.get("tables", []) if t != "chat_history"]
        if non_sys:
            st.markdown("**Active tables:**")
            for t in non_sys:
                n = h.get("row_counts", {}).get(t, "?")
                st.caption(f"📋 `{t}` — {n:,} rows" if isinstance(n, int) else f"📋 `{t}`")
        else:
            st.warning("No data loaded yet.\nGo to the **main page** to upload a file.")
    except Exception:
        st.caption("API offline")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _log_class(line: str) -> str:
    l = line.lower()
    if any(k in l for k in ("error", "failed", "❌")): return "ll-drop"
    if any(k in l for k in ("✅", "complete", "returned", "succeeded")): return "ll-ok"
    if any(k in l for k in ("⚠️", "warn", "retry", "0 rows")): return "ll-warn"
    return "ll-dim"

def _render_log(log: list, key_suffix: str = ""):
    html = "".join(f"<div class='{_log_class(l)}'>› {l}</div>" for l in log)
    st.markdown(f"<div class='lw'>{html}</div>", unsafe_allow_html=True)

def _intent_badge(intent: str) -> str:
    mapping = {
        "sql":            ('<span class="ib ib-sql">SQL</span>',     "🗄️"),
        "python":         ('<span class="ib ib-python">PYTHON</span>', "🐍"),
        "vector":         ('<span class="ib ib-vector">VECTOR</span>', "🔍"),
        "conversational": ('<span class="ib ib-conv">CHAT</span>',    "💬"),
    }
    badge, icon = mapping.get(intent, ('<span class="ib ib-err">?</span>', "❓"))
    return badge, icon

def _summary(qr: dict) -> str:
    if not qr.get("success"):
        return f"Error: {qr.get('error','')}"
    if qr.get("llm_answer"):
        words = qr["llm_answer"].split()
        return " ".join(words[:25]) + ("…" if len(words) > 25 else "")
    if qr.get("rows") is not None:
        return f"{qr.get('row_count', len(qr['rows'])):,} row(s) returned"
    if qr.get("chart_json"):
        return "Chart generated"
    if qr.get("vector_results"):
        return f"{len(qr['vector_results'])} similar result(s)"
    return "Done."


def _render_reasoning_panel(qr: dict, idx: int):
    """5-tab reasoning panel inside expander."""
    with st.expander("🔍 How did I get this?", expanded=False):
        tabs = st.tabs(["Result", "SQL / Code", "Reasoning", "Context used", "Execution log"])

        with tabs[0]:
            if qr.get("chart_json"):
                import plotly.io as pio
                fig = pio.from_json(qr["chart_json"])
                st.plotly_chart(fig, use_container_width=True, key=f"chart_exp_{idx}")
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
                st.info("No result data.")

        with tabs[1]:
            sql = qr.get("sql")
            code = qr.get("python_code")
            if sql and qr.get("intent") == "sql":
                st.code(sql, language="sql")
            elif code:
                st.code(code.replace("-- Python path\n", ""), language="python")
            else:
                st.caption("No SQL or code generated.")

        with tabs[2]:
            if qr.get("reasoning"):
                st.markdown(qr["reasoning"])
            else:
                st.caption("Enable **Reasoning mode** in the sidebar to capture chain-of-thought.")

        with tabs[3]:
            ctx = qr.get("context_used")
            if ctx:
                st.markdown(f"<div class='ctx-box'>{ctx}</div>", unsafe_allow_html=True)
            else:
                st.caption("No prior context was injected for this turn.")

        with tabs[4]:
            log = qr.get("bts_log", [])
            if log:
                _render_log(log, key_suffix=str(idx))
            else:
                st.caption("No execution log.")


def _render_message(msg: dict, idx: int):
    """Render one conversation turn."""
    role = msg["role"]

    if role == "user":
        st.markdown(
            f"<div class='bw user'>"
            f"<div class='av user'>YOU</div>"
            f"<div class='bb user'>{msg['text']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    # AI message
    qr = msg["result"]
    badge_html, _ = _intent_badge(qr.get("intent", "sql"))
    summary = _summary(qr)

    # Debug context tag
    ctx_tag = ""
    if st.session_state.debug_mode and qr.get("context_used"):
        ctx_tag = f"<div class='ctx-box'>📌 Context injected: {qr['context_used'][:80]}…</div>"

    st.markdown(
        f"<div class='bw ai'>"
        f"<div class='av ai'>AI</div>"
        f"<div class='bb ai'>{badge_html}<br>{ctx_tag}{summary}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Inline chart for Python/vector results (main view)
    if qr.get("chart_json") and qr.get("intent") == "python":
        import plotly.io as pio
        fig = pio.from_json(qr["chart_json"])
        st.plotly_chart(fig, use_container_width=True, key=f"chart_main_{idx}")

    # Inline dataframe for SQL results
    if qr.get("rows") and qr.get("intent") == "sql":
        df_show = pd.DataFrame(qr["rows"])
        st.dataframe(df_show.head(20), use_container_width=True)

    # Vector results inline
    if qr.get("vector_results") and qr.get("intent") == "vector":
        st.dataframe(pd.DataFrame(qr["vector_results"]).head(10), use_container_width=True)

    # Conversational — show full answer inline
    if qr.get("llm_answer") and qr.get("intent") == "conversational":
        st.markdown(qr["llm_answer"])

    # Error
    if not qr.get("success"):
        st.error(qr.get("error", "Unknown error"))

    # Always show reasoning panel
    _render_reasoning_panel(qr, idx)

    # Debug mode: always-visible BTS log
    if st.session_state.debug_mode and qr.get("bts_log"):
        with st.expander(f"🪵 Debug log (turn {idx})", expanded=True):
            _render_log(qr["bts_log"], key_suffix=f"debug_{idx}")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='margin:0;padding-top:1rem'>💬 Chat with your data</h2>"
    "<p style='color:#64748b;font-size:.85rem;margin-top:.2rem'>"
    "Ask anything in plain English — use the sidebar to adjust settings</p>",
    unsafe_allow_html=True,
)

# Check data is loaded
try:
    h = requests.get(f"{API_BASE}/health/", timeout=2).json()
    non_sys = [t for t in h.get("tables", []) if t != "chat_history"]
    data_ready = bool(non_sys)
except Exception:
    data_ready = False

if not data_ready:
    st.warning("No data loaded. Go to the **main page**, upload a file, and click **Run Pipeline**.")
    st.stop()

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.chat_messages:
    st.markdown(
        "<p style='text-align:center;color:#94a3b8;padding:3rem 0'>No messages yet — ask something below ↓</p>",
        unsafe_allow_html=True,
    )
else:
    for i, msg in enumerate(st.session_state.chat_messages):
        _render_message(msg, i)

st.divider()

# ── Input bar ─────────────────────────────────────────────────────────────────
col_q, col_send = st.columns([5, 1])
with col_q:
    user_input = st.text_input(
        "Ask a question",
        placeholder='e.g. "How many High priority tickets?" · "Plot tickets by department" · "Find tickets about login issues"',
        label_visibility="collapsed",
        key="chat_input",
    )
with col_send:
    send_btn = st.button("Send ▶", type="primary", use_container_width=True)

# Suggested questions
st.caption("Suggested: "
    "| How many tickets per priority? "
    "| Plot ticket volume over time "
    "| Find tickets about system failures "
    "| What did I ask earlier?")

# ── Handle submission ─────────────────────────────────────────────────────────
if send_btn and user_input.strip():
    query = user_input.strip()
    st.session_state.chat_messages.append({"role": "user", "text": query})

    with st.spinner("Thinking…"):
        try:
            resp = requests.post(
                f"{API_BASE}/query/",
                json={
                    "query":        query,
                    "session_id":   st.session_state.session_id,
                    "reasoning":    reasoning_mode,
                    "debug_mode":   st.session_state.debug_mode,
                    "force_intent": None if force_path == "auto" else force_path,
                    "top_k_vector": top_k,
                },
                timeout=90,
            )
            qr = resp.json()
        except Exception as e:
            qr = {"success": False, "intent": "error", "error": str(e), "bts_log": []}

    st.session_state.chat_messages.append({"role": "ai", "result": qr})
    st.rerun()