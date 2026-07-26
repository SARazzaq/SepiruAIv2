"""
RAG Chat powered by LlamaIndex + HuggingFace embeddings.
Runs locally — requires: pip install llama-index llama-index-embeddings-huggingface sentence-transformers
"""
import streamlit as st
from src.ui_components import load_all_styles

load_all_styles("assets")

st.markdown("""
<div class='app-header'>
    <h1>🔎 RAG Chat <span>— LlamaIndex</span></h1>
    <p>Vector search over your CSV · HuggingFace embeddings · LlamaIndex retrieval</p>
</div>""", unsafe_allow_html=True)

# ── Dependency check ───────────────────────────────────────────────────────────
try:
    import llama_index  # noqa: F401
    _rag_available = True
except ImportError:
    _rag_available = False

if not _rag_available:
    st.warning("⚠️ LlamaIndex is not installed in this environment.")
    st.markdown("""
This feature requires heavy ML packages that exceed Streamlit Cloud's free-tier limits.

**Run it locally:**
```bash
pip install llama-index llama-index-embeddings-huggingface llama-index-llms-openai-like sentence-transformers
streamlit run app.py
```
Then navigate to this page in your local browser.
    """)
    st.stop()

if st.session_state.get("df") is None:
    st.warning("Upload a file on the main page first.")
    st.stop()

df = st.session_state.clean_df

@st.cache_resource(show_spinner="Building vector index…")
def get_engine(_df, _hash):
    from src.rag_llamaindex import build_llamaindex_engine
    from src.ai_client import AIClient
    return build_llamaindex_engine(_df, AIClient())

import pandas as pd
file_hash = str(pd.util.hash_pandas_object(df).sum())

try:
    engine = get_engine(df, file_hash)
    st.success("✅ Vector index ready — LlamaIndex + BGE-small embeddings")
except Exception as e:
    st.error(f"Index build failed: {e}")
    st.stop()

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
        with st.spinner("Retrieving context…"):
            from src.rag_llamaindex import llamaindex_query
            answer, sources = llamaindex_query(engine, question)
        st.markdown(answer)
        if sources:
            with st.expander(f"📎 {len(sources)} source chunk(s)"):
                for i, src in enumerate(sources, 1):
                    st.code(src, language="text")

    st.session_state.rag_history.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.markdown("### ℹ️ How it works")
    st.markdown("CSV rows → LlamaIndex Documents → BGE-small embeddings → ChromaDB → top-5 retrieval → LLM synthesis")
    if st.button("🗑️ Clear chat"):
        st.session_state.rag_history = []
        st.rerun()
