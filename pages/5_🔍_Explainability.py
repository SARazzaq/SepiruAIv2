"""
Explainability Layer — AI explains ML model decisions in plain English.
Uses Groq (free) for explanations.
"""
import streamlit as st
import pandas as pd
import numpy as np
from src.ui_components import load_all_styles
from src.quota_guard import can_proceed, get_usage, reset_time_utc

load_all_styles("assets")

from src.auth import require_auth
require_auth()

from src.ai_client import AIClient

st.markdown("""
<div class='app-header'>
    <h1>🔍 Model <span>Explainability</span></h1>
    <p>Understand why your ML model made each prediction</p>
</div>""", unsafe_allow_html=True)

if not can_proceed():
    st.error(f"⏳ Daily quota reached. Resets in **{reset_time_utc()}**.")
    st.stop()

if st.session_state.get("df") is None:
    st.warning("Upload a file on the main page first.")
    st.stop()

if "ml_results" not in st.session_state:
    st.info("Train a model on the ML Training page first, then come back here.")
    st.stop()

results = st.session_state["ml_results"]
df      = st.session_state.clean_df
u       = get_usage()
st.caption(f"AI quota: {u['count']}/{u['limit']} used today · {u['remaining']} remaining")

# ── Model overview ────────────────────────────────────────────────────────────
st.subheader("🧠 Model Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Model", results["model_name"])
c2.metric("Task",  results["task"].capitalize())
if results["task"] == "classification":
    c3.metric("Accuracy", f"{results['accuracy']}%")
else:
    c3.metric("R²", results.get("r2", "N/A"))

# Feature importance chart
if results.get("fi_fig"):
    st.plotly_chart(results["fi_fig"], use_container_width=True)

# ── Feature importance table ──────────────────────────────────────────────────
feat_imp_str = "not available"
if results.get("feature_importances"):
    fi = results["feature_importances"]
    top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
    feat_imp_str = ", ".join(f"{k} ({v:.3f})" for k, v in top)
    fi_df = pd.DataFrame(top, columns=["Feature", "Importance"])
    st.subheader("📊 Feature Importance")
    st.dataframe(fi_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── AI Explanation ────────────────────────────────────────────────────────────
st.subheader("💬 Ask the Model to Explain Itself")

lang = st.selectbox("Response language", [
    "English","Hindi","Spanish","French","German",
    "Arabic","Chinese","Portuguese","Russian","Korean"
])

questions = [
    "Why did the model make these predictions?",
    "Which features matter most and why?",
    "Where is the model likely to be wrong?",
    "Explain this model to a non-technical manager.",
    "What data would improve this model?",
    "What are the limitations of this model?",
]
q_choice = st.selectbox("Choose a question or type your own",
                        ["Custom…"] + questions)
if q_choice == "Custom…":
    question = st.text_input("Your question", placeholder="Ask anything about the model…")
else:
    question = q_choice

if st.button("▶️ Explain", type="primary", use_container_width=True):
    if not question:
        st.warning("Please enter or select a question.")
    elif not can_proceed():
        st.error(f"Quota exhausted. Resets in {reset_time_utc()}.")
    else:
        with st.spinner("Generating explanation…"):
            try:
                ai = AIClient()
                if results["task"] == "classification":
                    perf = f"Accuracy: {results['accuracy']}%, CV: {results['cv_mean']}%±{results['cv_std']}%"
                else:
                    perf = f"MAE: {results.get('mae','N/A')}, RMSE: {results.get('rmse','N/A')}, R²: {results.get('r2','N/A')}"

                prompt = f"""ML Model: {results['model_name']}
Task: {results['task']}
Performance: {perf}
Top features by importance: {feat_imp_str}
Dataset columns: {list(df.columns)}
Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns

Question: {question}

Answer in {lang}. Be specific — use actual feature names and numbers.
Make it understandable for a business audience."""

                answer = ai.generate(
                    prompt,
                    system="You are an expert ML explainability specialist."
                )
                st.markdown(
                    f'<div class="insight-box">{answer}</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")

# ── Single-row prediction + explanation ───────────────────────────────────────
st.subheader("🔮 Predict & Explain a Single Row")
st.info("Fill in values to get a prediction with an AI explanation.")

model = results.get("model")
if model is None:
    st.warning("No trained model found in session. Please retrain on the ML Training page.")
    st.stop()

feature_cols = [c for c in df.columns if c != results.get("target_col", "")]
num_cols     = df[feature_cols].select_dtypes(include="number").columns.tolist()

if not num_cols:
    st.warning("No numeric feature columns found.")
    st.stop()

vals = {}
display_cols = num_cols[:8]
grid = st.columns(min(4, len(display_cols)))
for i, col in enumerate(display_cols):
    with grid[i % 4]:
        vals[col] = st.number_input(
            col,
            value=float(df[col].median()),
            key=f"exp_{col}"
        )

if st.button("Predict this row", use_container_width=True):
    try:
        from sklearn.preprocessing import StandardScaler
        row    = pd.DataFrame([vals])
        scaler = StandardScaler()
        scaler.fit(df[num_cols])
        row_scaled = scaler.transform(row[num_cols])
        pred = model.predict(row_scaled)[0]

        st.success(f"**Prediction: {pred}**")

        if can_proceed():
            with st.spinner("Generating explanation…"):
                try:
                    ai = AIClient()
                    exp_prompt = f"""Model: {results['model_name']}
Prediction: {pred}
Input values: {vals}
Top features: {feat_imp_str}

In 2-3 sentences, explain why this prediction makes sense given the input values.
Respond in {lang}."""
                    exp = ai.generate(exp_prompt)
                    st.markdown(
                        f'<div class="insight-box">{exp}</div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Explanation error: {e}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
