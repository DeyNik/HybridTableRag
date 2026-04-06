import streamlit as st
import sys, os
from pathlib import Path
import pandas as pd

# ── Project setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hybridtablerag.core.cleaner import read_file, clean_dataframe
from hybridtablerag.core.normalizer import Normalizer
from hybridtablerag.llm.factory import get_llm
from hybridtablerag.storage.store import DuckDBStore
from hybridtablerag.storage.vectors import VectorStore, get_embedding_provider
from hybridtablerag.reasoning.intent import IntentClassifier
from hybridtablerag.reasoning.sql import LLMSQLGenerator
from hybridtablerag.reasoning.orchestrator import QueryOrchestrator
from hybridtablerag.storage.schema import build_multi_table_schema_context, format_schema_for_prompt

st.set_page_config(page_title="HybridTableRAG Stepwise Interface", layout="wide")
st.title("HybridTableRAG Stepwise Full-Detail Interface")

# ── User Inputs
header_rows = st.number_input(
    "Header row count (0-indexed, e.g., 0=1 header, 1=2 headers)",
    min_value=0, max_value=5, value=0
)
uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])

# ── Session state for persistence
if "cleaned_sheets" not in st.session_state:
    st.session_state.cleaned_sheets = {}
if "plans" not in st.session_state:
    st.session_state.plans = {}
if "store" not in st.session_state:
    st.session_state.store = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "all_tables" not in st.session_state:
    st.session_state.all_tables = {}
if "llm" not in st.session_state:
    st.session_state.llm = None
if "logs" not in st.session_state:
    st.session_state.logs = {
        "cleaning": {},
        "normalization": {},
        "embedding": [],
        "query": []
    }

# ── Step 1: File Upload & Cleaning
if uploaded_file and st.button("Step 1: Load & Clean File"):
    st.subheader("Step 1: File Upload & Cleaning")
    sheets = read_file(uploaded_file, header_rows=list(range(header_rows + 1)))
    st.session_state.logs["cleaning"] = {}
    for name, df in sheets.items():
        cleaned, log = clean_dataframe(df, log=[])
        st.session_state.cleaned_sheets[name] = cleaned
        st.session_state.logs["cleaning"][name] = log

# Display all cleaning logs
st.subheader("Cleaning Logs")
for name, log in st.session_state.logs["cleaning"].items():
    st.markdown(f"**Sheet: {name}**")
    st.text("\n".join(log))
    st.dataframe(st.session_state.cleaned_sheets[name].head(5))

# ── Step 2: Normalization & DuckDB Registration
if st.session_state.cleaned_sheets and st.button("Step 2: Normalize & Register"):
    st.subheader("Step 2: Normalization & DuckDB Registration")
    DB_PATH = "data/hybridtablerag_streamlit.duckdb"
    os.makedirs(Path(DB_PATH).parent, exist_ok=True)
    store = DuckDBStore(db_path=DB_PATH)
    st.session_state.store = store
    st.success(f"DuckDB initialized at {DB_PATH}")

    if not st.session_state.llm:
        st.session_state.llm = get_llm()

    normalizer = Normalizer(llm=st.session_state.llm)
    st.session_state.logs["normalization"] = {}

    for name, df in st.session_state.cleaned_sheets.items():
        st.markdown(f"**Normalizing: {name}**")
        try:
            plan = normalizer.normalize(df, table_hint=name.replace(" ", "_").lower(), profile_hints={}, log=[])
            st.session_state.plans[name] = plan
            st.session_state.logs["normalization"][name] = plan.log if hasattr(plan, "log") else []

            store.register_normalization_plan(plan, [])
            st.success(f"Registered {plan.main_table_name} ({len(plan.bridge_tables)} bridge tables)")
        except Exception as e:
            st.session_state.logs["normalization"][name] = [str(e)]

# Display all normalization logs and table previews
st.subheader("Normalization Logs")
for name, log in st.session_state.logs["normalization"].items():
    st.markdown(f"**Table: {name}**")
    st.text("\n".join(log))
    plan = st.session_state.plans.get(name)
    if plan:
        st.dataframe(plan.main_df.head(5))
        for bt in plan.bridge_tables:
            st.markdown(f"Bridge Table: {bt.name}")
            st.dataframe(bt.df.head(5))

# ── Step 3: Vector Embedding
if st.session_state.plans and st.button("Step 3: Vector Embedding"):
    st.header("Step 3: Vector Embedding")
    provider = get_embedding_provider()
    vector_store = VectorStore(st.session_state.store.conn, provider)
    vector_store.setup()
    st.session_state.vector_store = vector_store
    st.success(f"VectorStore ready (dim: {provider.dimension})")

    bts_log = []

    all_tables = {}
    for plan in st.session_state.plans.values():
        all_tables[plan.main_table_name] = plan.main_df
        for bt in plan.bridge_tables:
            all_tables[bt.name] = bt.df
    st.session_state.all_tables = all_tables

    st.subheader("Tables to process:")
    st.write(list(all_tables.keys()))

    def is_semantic_column(store, table_name, col):
        try:
            if table_name not in store.list_tables():
                return False
            rows = store.conn.execute(
                f'SELECT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL LIMIT 50'
            ).fetchall()
            values = [str(r[0]) for r in rows if r[0] is not None]
            if not values:
                return False
            total_chars = sum(len(v) for v in values)
            alpha_chars = sum(sum(c.isalpha() for c in v) for v in values)
            alpha_ratio = alpha_chars / total_chars if total_chars else 0
            avg_token_len = sum(len(v.split()) for v in values) / len(values)
            distinct = store.conn.execute(f'SELECT COUNT(DISTINCT "{col}") FROM "{table_name}"').fetchone()[0]
            if alpha_ratio > 0.5 and avg_token_len >= 1:
                return True
            if distinct < 50 and alpha_ratio > 0.6:
                return True
            return False
        except Exception as e:
            bts_log.append(f"Semantic check failed for {table_name}.{col}: {e}")
            return False

    for table_name in all_tables.keys():
        if table_name not in st.session_state.store.list_tables():
            st.warning(f"Skipping {table_name}: table not registered in DuckDB")
            continue

        st.write(f"Processing table: {table_name}")
        schema = st.session_state.store.get_table_schema(table_name)
        pk_column = next((c["column_name"] for c in schema if c["column_name"].endswith("_id")), None)
        if not pk_column:
            st.warning(f"No PK found for {table_name}, generating surrogate PK")
            st.session_state.store.conn.execute(
                f'ALTER TABLE "{table_name}" ADD COLUMN row_id INTEGER GENERATED ALWAYS AS IDENTITY'
            )
            pk_column = "row_id"

        text_cols = [
            c["column_name"]
            for c in schema
            if c["data_type"] in ("VARCHAR", "TEXT") and c["column_name"] != pk_column and
            is_semantic_column(st.session_state.store, table_name, c["column_name"])
        ]

        if not text_cols:
            st.info(f"No semantic text columns found for {table_name}, skipping embedding")
            continue

        vector_store.embed_table(table_name=table_name, text_columns=text_cols, pk_column=pk_column, bts_log=bts_log)
        st.success(f"Embedded table {table_name} with columns: {text_cols}")

    st.session_state.logs["embedding"].extend(bts_log)
    with st.expander("Vector Embedding Logs", expanded=True):
        st.text("\n".join(st.session_state.logs["embedding"]) if st.session_state.logs["embedding"] else "No logs. All embeddings completed successfully.")
# ── Step 4: Query
if st.session_state.all_tables:
    st.subheader("Step 4: Query Interface")
    
    # Build schema context once
    schema_ctx = build_multi_table_schema_context(
        st.session_state.store.conn,
        list(st.session_state.all_tables.keys()),
        relationships=[], bts_log=[]
    )
    schema_summary = format_schema_for_prompt(schema_ctx)

    intent = IntentClassifier(st.session_state.llm)
    sql_gen = LLMSQLGenerator(st.session_state.llm)
    orchestrator = QueryOrchestrator(
        llm=st.session_state.llm,
        store=st.session_state.store,
        context_store=None,
        sql_generator=sql_gen,
        table_names=list(st.session_state.all_tables.keys()),
        relationships=[],
        vector_store=st.session_state.vector_store,
        default_table=list(st.session_state.all_tables.keys())[0],
    )

    # Persistent user query
    if "user_query" not in st.session_state:
        st.session_state.user_query = "show ticket count by status"

    st.session_state.user_query = st.text_input(
        "Enter your query here", 
        value=st.session_state.user_query, 
        key="query_input"
    )

    # Execute query on button click
    if st.button("Execute Query"):
        if st.session_state.user_query.strip():
            result = orchestrator.run(
                user_query=st.session_state.user_query,
                session_id="streamlit_session",
                debug_mode=True
            )
            st.session_state.logs["query"].append(result)

# Display all previous queries and results
if st.session_state.logs["query"]:
    st.subheader("Query History")
    for i, res in enumerate(st.session_state.logs["query"]):
        st.markdown(f"**Query {i+1}:** {st.session_state.user_query if i == len(st.session_state.logs['query'])-1 else ''}")
        st.write("Intent:", getattr(res, "intent", ""))
        st.write("SQL:", getattr(res, "sql", ""))
        if getattr(res, "dataframe", None) is not None:
            st.dataframe(res.dataframe)
        if getattr(res, "error", None):
            st.error(res.error)
        if getattr(res, "bts_log", None):
            st.text("\n".join(res.bts_log))