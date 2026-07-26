"""
DeepEval — evaluate RAG pipeline quality with open-source metrics.
Apache 2.0 license. No LLM API needed for local metrics.
"""
import streamlit as st
import pandas as pd
from src.ui_components import load_all_styles
from src.auth import require_auth

load_all_styles("assets")
require_auth()

st.markdown("""
<div class='app-header'>
    <h1>📊 DeepEval <span>RAG Evaluation</span></h1>
    <p>Measure answer quality · faithfulness · contextual recall · open-source</p>
</div>""", unsafe_allow_html=True)

try:
    import deepeval
    st.caption(f"DeepEval {deepeval.__version__} ready")
except ImportError:
    st.error("DeepEval not installed.")
    st.code("pip install deepeval")
    st.stop()

if st.session_state.get("df") is None:
    st.warning("Upload a file on the main page first.")
    st.stop()

df = st.session_state.clean_df

# ── Check if RAG engine is available ─────────────────────────────────────────
rag_ready = False
engine = None

@st.cache_resource(show_spinner="Building RAG index for evaluation…")
def get_eval_engine(_df, _hash):
    from src.rag_llamaindex import build_llamaindex_engine
    from src.ai_client import AIClient
    return build_llamaindex_engine(_df, AIClient())

try:
    fhash = str(pd.util.hash_pandas_object(df).sum())
    engine = get_eval_engine(df, fhash)
    rag_ready = True
    st.success("✅ LlamaIndex RAG engine ready")
except ImportError:
    st.warning("LlamaIndex not installed — using manual input mode.")

# ── Evaluation mode ────────────────────────────────────────────────────────────
st.subheader("🧪 Evaluate a RAG response")

mode = st.radio("Mode", ["Auto (use RAG engine)", "Manual input"], horizontal=True,
                disabled=not rag_ready)

col1, col2 = st.columns(2)
with col1:
    question = st.text_area("Question", f"What is the average of {df.columns[0]}?", height=80)
with col2:
    expected = st.text_area("Expected answer (optional)", "", height=80,
                            help="Leave blank to skip contextual recall metric")

if mode == "Manual input" or not rag_ready:
    answer = st.text_area("LLM Answer (to evaluate)", "", height=100)
    contexts_raw = st.text_area(
        "Retrieved contexts (one per line)",
        "\n".join([f"{col}: {df[col].describe().to_string()}" for col in df.select_dtypes('number').columns[:2]]),
        height=150,
    )
    contexts = [c.strip() for c in contexts_raw.split("\n") if c.strip()]
else:
    answer = None
    contexts = None

use_local = st.checkbox("Use local metrics (no LLM API)", value=True,
                        help="Local metrics are fast and free. Disable for LLM-judged evaluation.")

if st.button("🔍 Evaluate", type="primary"):
    from src.deepeval_runner import evaluate_rag_response, format_eval_report
    from src.rag_llamaindex import llamaindex_query

    # Generate answer via RAG if in auto mode
    if mode == "Auto (use RAG engine)" and rag_ready and engine:
        with st.spinner("Querying RAG engine…"):
            answer, contexts = llamaindex_query(engine, question)
        st.markdown("**RAG Answer:**")
        st.info(answer)
        if contexts:
            with st.expander(f"📎 {len(contexts)} retrieved chunks"):
                for c in contexts:
                    st.caption(c[:300])

    if not answer:
        st.warning("Please provide an answer to evaluate.")
        st.stop()

    with st.spinner("Running DeepEval metrics…"):
        try:
            results = evaluate_rag_response(
                question=question,
                answer=answer,
                retrieved_contexts=contexts or [],
                expected_answer=expected or None,
                use_local=use_local,
            )
        except Exception as e:
            st.error(f"Evaluation error: {e}")
            st.stop()

    # Display results
    st.subheader("📋 Evaluation Results")

    metric_cols = st.columns(len(results))
    for (metric, data), col in zip(results.items(), metric_cols):
        score = data.get("score")
        passed = data.get("passed")
        label = metric.replace("_", " ").title()
        delta = "✅ Pass" if passed else ("❌ Fail" if passed is False else "⚠️ N/A")
        col.metric(label, f"{score:.3f}" if score is not None else "N/A", delta)

    # Detailed report
    with st.expander("📄 Full Report"):
        st.code(format_eval_report(results), language="text")

    # Batch eval CTA
    st.subheader("📦 Batch Evaluation")
    st.markdown("To evaluate multiple questions at once, use `deepeval_runner.batch_evaluate()` in your code.")
    st.code("""from src.deepeval_runner import batch_evaluate

test_cases = [
    {"question": "What is the max sales?", "answer": "...", "contexts": [...]},
    {"question": "Which region performs best?", "answer": "...", "contexts": [...]},
]

results = batch_evaluate(test_cases)""", language="python")

with st.sidebar:
    st.markdown("### 📐 Metrics explained")
    st.markdown("""
**Answer Relevancy**
Does the answer actually address the question?

**Faithfulness**
Is every claim in the answer supported by the retrieved context? (Hallucination check)

**Contextual Recall**
Does the retrieved context contain enough information to answer the question?
    """)
    st.markdown("[DeepEval Docs](https://docs.confident-ai.com)")
