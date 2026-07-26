"""
DeepEval — evaluate RAG pipeline quality.
Requires: pip install deepeval
"""
import streamlit as st
import pandas as pd
from src.ui_components import load_all_styles

load_all_styles("assets")

st.markdown("""
<div class='app-header'>
    <h1>📊 DeepEval <span>RAG Evaluation</span></h1>
    <p>Measure answer quality · faithfulness · contextual recall</p>
</div>""", unsafe_allow_html=True)

# ── Dependency check ───────────────────────────────────────────────────────────
try:
    import deepeval
    st.caption(f"DeepEval {deepeval.__version__} ready")
    _deepeval_available = True
except ImportError:
    _deepeval_available = False

if not _deepeval_available:
    st.warning("⚠️ DeepEval is not installed in this environment.")
    st.markdown("""
**Run it locally:**
```bash
pip install deepeval
streamlit run app.py
```
DeepEval runs evaluation metrics locally — no API key needed when using `model="local"`.
    """)
    st.stop()

if st.session_state.get("df") is None:
    st.warning("Upload a file on the main page first.")
    st.stop()

df = st.session_state.clean_df

st.subheader("🧪 Evaluate a RAG response")

col1, col2 = st.columns(2)
with col1:
    question = st.text_area("Question", f"What is the average of {df.columns[0]}?", height=80)
with col2:
    expected = st.text_area("Expected answer (optional)", "", height=80)

answer = st.text_area("LLM Answer (to evaluate)", "", height=100)

num_cols = df.select_dtypes("number").columns[:2].tolist()
default_ctx = "\n".join([f"{col}: {df[col].describe().to_string()}" for col in num_cols])
contexts_raw = st.text_area("Retrieved contexts (one per line)", default_ctx, height=120)
contexts = [c.strip() for c in contexts_raw.split("\n") if c.strip()]

use_local = st.checkbox("Use local metrics (no LLM API)", value=True)

if st.button("🔍 Evaluate", type="primary"):
    if not answer.strip():
        st.warning("Enter an LLM answer to evaluate.")
        st.stop()

    from src.deepeval_runner import evaluate_rag_response, format_eval_report
    with st.spinner("Running DeepEval metrics…"):
        try:
            results = evaluate_rag_response(
                question=question,
                answer=answer,
                retrieved_contexts=contexts,
                expected_answer=expected or None,
                use_local=use_local,
            )
        except Exception as e:
            st.error(f"Evaluation error: {e}")
            st.stop()

    metric_cols = st.columns(len(results))
    for (metric, data), col in zip(results.items(), metric_cols):
        score = data.get("score")
        passed = data.get("passed")
        col.metric(
            metric.replace("_", " ").title(),
            f"{score:.3f}" if score is not None else "N/A",
            "✅ Pass" if passed else ("❌ Fail" if passed is False else "⚠️"),
        )

    with st.expander("📄 Full Report"):
        st.code(format_eval_report(results), language="text")

with st.sidebar:
    st.markdown("""### Metrics
**Answer Relevancy** — does the answer address the question?

**Faithfulness** — is every claim backed by context? (hallucination check)

**Contextual Recall** — does context contain enough to answer?
    """)
    st.markdown("[DeepEval Docs](https://docs.confident-ai.com)")
