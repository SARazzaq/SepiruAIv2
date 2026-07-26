"""
ML Training Page — auto-detects classification vs regression
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

from src.ui_components import load_all_styles
load_all_styles("assets")

from src.ml_trainer import (
    detect_task, prepare_data,
    train_classification, train_regression
)

st.markdown("""
<div class='app-header'>
    <h1>🤖 <span>ML</span> Training</h1>
    <p>Auto-detect &nbsp;·&nbsp; Train &nbsp;·&nbsp; Evaluate &nbsp;—&nbsp; no code needed</p>
</div>
""", unsafe_allow_html=True)

# ── Load data from main app session ──────────────────────────────────────────
if st.session_state.get("df") is None:
    st.warning("⚠️ No data loaded. Please upload a file on the main page first.")
    st.stop()

df = st.session_state.clean_df

# ── Config ────────────────────────────────────────────────────────────────────
st.subheader("⚙️ Configuration")

c1, c2, c3 = st.columns(3)
with c1:
    target_col = st.selectbox("🎯 Target column (what to predict)", df.columns)
with c2:
    task = detect_task(df, target_col)
    st.metric("Detected Task", task.capitalize())
with c3:
    from src.ml_trainer import get_classification_models, get_regression_models
    if task == "classification":
        model_name = st.selectbox("🧠 Model", list(get_classification_models().keys()))
    else:
        model_name = st.selectbox("🧠 Model", list(get_regression_models().keys()))

    run_compare = st.checkbox("🏆 Compare ALL models (leaderboard)", value=False)

st.info(f"**{len(df)} rows** · Target: `{target_col}` · Task: **{task}**")

# ── Feature selection ─────────────────────────────────────────────────────────
with st.expander("🔧 Feature Selection (optional)", expanded=False):
    all_features = [c for c in df.columns if c != target_col]
    selected_features = st.multiselect(
        "Select features to use (leave empty = use all)",
        all_features
    )
    if selected_features:
        train_df = df[selected_features + [target_col]]
    else:
        train_df = df

# ── Train ─────────────────────────────────────────────────────────────────────
if st.button("▶️ Train Model", type="primary", use_container_width=True):
    with st.spinner("Training…"):
        try:
            X_train, X_test, y_train, y_test, feat_names, le = prepare_data(
                train_df, target_col
            )

            if run_compare:
                from src.ml_trainer import compare_all_models
                with st.spinner("Comparing all models… this may take a minute"):
                    leaderboard = compare_all_models(
                        X_train, X_test, y_train, y_test, task
                    )
                st.subheader("🏆 Model Leaderboard")
                st.dataframe(leaderboard, use_container_width=True,
                             hide_index=True)
                sort_col = "Accuracy %" if task == "classification" else "R²"
                if sort_col in leaderboard.columns:
                    best = leaderboard.iloc[0]["Model"]
                    st.success(f"🥇 Best model: **{best}**")

            if task == "classification":
                results = train_classification(
                    X_train, X_test, y_train, y_test,
                    feat_names, model_name, le
                )
            else:
                results = train_regression(
                    X_train, X_test, y_train, y_test,
                    feat_names, model_name
                )
            st.session_state["ml_results"] = results
        except Exception as e:
            st.error(f"Training error: {e}")
# ── Results ───────────────────────────────────────────────────────────────────
if "ml_results" in st.session_state:
    results = st.session_state["ml_results"]
    st.markdown("---")
    st.subheader("📊 Results")

    if results["task"] == "classification":
        m1, m2, m3 = st.columns(3)
        m1.metric("Model",    results["model_name"])
        m2.metric("Accuracy", f"{results['accuracy']}%")
        m3.metric("CV Score", f"{results['cv_mean']}% ± {results['cv_std']}%")
        report_df = pd.DataFrame(results["report"]).T.round(2)
        m3.metric("Classes", len(report_df) - 3)

        st.markdown("---")
        st.subheader("📋 Classification Report")
        st.dataframe(report_df, use_container_width=True)

        col_cm, col_fi = st.columns(2)
        with col_cm:
            st.plotly_chart(results["cm_fig"], use_container_width=True)
        with col_fi:
            if results["fi_fig"]:
                st.plotly_chart(results["fi_fig"], use_container_width=True)

    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", results["model_name"])
        m2.metric("MAE",   results["mae"])
        m3.metric("RMSE",  results["rmse"])
        m4.metric("R²",    results["r2"])
        st.metric("CV R²", f"{results['cv_mean']} ± {results['cv_std']}")

        col_ap, col_res = st.columns(2)
        with col_ap:
            st.plotly_chart(results["ap_fig"], use_container_width=True)
        with col_res:
            st.plotly_chart(results["res_fig"], use_container_width=True)

        if results["fi_fig"]:
            st.plotly_chart(results["fi_fig"], use_container_width=True)

# ── Predict on new data ───────────────────────────────────────────────────────
if "ml_results" in st.session_state:
    st.markdown("---")
    st.subheader("🔮 Predict on New Data")
    st.info("Upload a new CSV (same columns, no target) to get predictions.")
    pred_file = st.file_uploader("Upload prediction file", type=["csv","xlsx"])
    if pred_file:
        try:
            from src.utils import load_data
            pred_df = load_data(pred_file)
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            import numpy as np
            X_new = pred_df.copy()
            # Convert datetime to numeric
            for col in X_new.columns:
                if pd.api.types.is_datetime64_any_dtype(X_new[col]):
                    X_new[col] = X_new[col].astype(np.int64) // 10**9
            for col in X_new.select_dtypes(include="object").columns:
                le = LabelEncoder()
                X_new[col] = le.fit_transform(X_new[col].astype(str))
            X_new = X_new.select_dtypes(include="number")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_new)
            preds = st.session_state["ml_results"]["model"].predict(X_scaled)
            pred_df["Prediction"] = preds
            st.dataframe(pred_df, use_container_width=True)
            csv = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", csv,
                               file_name="predictions.csv", mime="text/csv",
                               use_container_width=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")