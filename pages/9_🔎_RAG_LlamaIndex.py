"""
RAG Chat powered by LlamaIndex + HuggingFace embeddings.
Free & open-source — no extra API keys needed.
"""
import streamlit as st
from src.ui_components import load_all_styles
from src.auth import require_auth

load_all_styles("assets")
require_auth()

st.markdown("""
<div class='app-header'>
    <h1>🔎 RAG Chat <span>— LlamaIndex</span></h1>
    <p>Vector search over your CSV · HuggingFace embeddings · LlamaIndex retrieval</p>
</div>""", unsafe_allow_html=True)

if st.session_state.get("df") is None:
    st.warning("Upload a file on the main page first.")
    st.stop()

df = st.session_state.clean_df

# ── Build / cache the index ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building vector index…")
def get_engine(_df, _hash):
    from src.rag_llamaindex import build_llamaindex_engine
    from src.ai_client import AIClient
    return build_llamaindex_engine(_df, AIClient())

import pandas as pd
file_hash = str(pd.util.hash_pandas_object(df).sum())

try:
    engine = get_engine(df, file_hash)
    st.success("✅ Vector index ready — powered by LlamaIndex + BGE-small embeddings")
except ImportError as e:
    st.error(str(e))
    st.code("pip install llama-index llama-index-embeddings-huggingface sentence-transformers")
    st.stop()

# ── Chat interface ─────────────────────────────────────────────────────────────
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []

for msg in st.session_state.rag_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask anything about your data…")
if question:
    st.session_state.rag_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context…"):
            from src.rag_llamaindex import llamaindex_query
            answer, sources = llamaindex_query(engine, question)

        st.markdown(answer)

        if sources:
            with st.expander(f"📎 {len(sources)} source chunk(s) retrieved"):
                for i, src in enumerate(sources, 1):
                    st.caption(f"Chunk {i}:")
                    st.code(src, language="text")

    st.session_state.rag_history.append({"role": "assistant", "content": answer})

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ How it works")
    st.markdown("""
1. CSV rows → LlamaIndex Documents
2. BGE-small embeddings (local HuggingFace)
3. ChromaDB vector store
4. Top-5 chunks retrieved per query
5. LLM synthesises the answer
    """)
    if st.button("🗑️ Clear chat"):
        st.session_state.rag_history = []
        st.rerun()
