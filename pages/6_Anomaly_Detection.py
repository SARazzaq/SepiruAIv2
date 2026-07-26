"""
Anomaly Detection with Business Alerts — finds outliers and explains their business impact.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.ui_components import load_all_styles
from src.quota_guard import can_proceed, get_usage, reset_time_utc

load_all_styles("assets")

from src.auth import require_auth
require_auth()

from src.ai_client import AIClient

DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,15,26,0.6)",
    font=dict(family="DM Sans, sans-serif", color="#9090a8", size=11),
)

st.markdown("""
<div class='app-header'>
    <h1>⚠️ Anomaly <span>Detection</span></h1>
    <p>Detect unusual patterns and get AI business alerts</p>
</div>""", unsafe_allow_html=True)

if st.session_state.get("df") is None:
    st.warning("Upload a file on the main page first.")
    st.stop()

df = st.session_state.clean_df
num_cols = df.select_dtypes(include="number").columns.tolist()

if not num_cols:
    st.warning("No numeric columns found.")
    st.stop()

u = get_usage()
st.caption(f"AI quota: {u['count']}/{u['limit']} used · {u['remaining']} remaining")

st.subheader("⚙️ Detection Settings")
c1, c2, c3 = st.columns(3)
with c1:
    method = st.selectbox("Method", ["IQR (Statistical)", "Z-Score", "Isolation Forest"])
with c2:
    cols_sel = st.multiselect("Columns to analyze", num_cols, default=num_cols[:4])
with c3:
    lang = st.selectbox("Alert language", [
        "English","Hindi","Spanish","French","German","Arabic","Chinese","Portuguese"
    ])

if not cols_sel:
    st.info("Select at least one column.")
    st.stop()

if st.button("▶️ Detect Anomalies", type="primary", use_container_width=True):
    with st.spinner("Scanning for anomalies…"):
        anomaly_mask = pd.Series([False] * len(df), index=df.index)
        details = {}

        for col in cols_sel:
            s = df[col].dropna()
            if method == "IQR (Statistical)":
                Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
                IQR = Q3 - Q1
                mask = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)
                details[col] = {"method":"IQR","Q1":round(Q1,2),"Q3":round(Q3,2),"IQR":round(IQR,2)}
            elif method == "Z-Score":
                z = (df[col] - s.mean()) / s.std()
                mask = z.abs() > 3
                details[col] = {"method":"Z-Score","mean":round(s.mean(),2),"std":round(s.std(),2)}
            else:
                from sklearn.ensemble import IsolationForest
                iso = IsolationForest(contamination=0.05, random_state=42)
                preds = iso.fit_predict(df[[col]].fillna(df[col].median()))
                mask = pd.Series(preds == -1, index=df.index)
                details[col] = {"method":"Isolation Forest","contamination":"5%"}
            anomaly_mask |= mask

        anomaly_df = df[anomaly_mask].copy()
        normal_df  = df[~anomaly_mask].copy()
        n_anomalies = anomaly_mask.sum()
        pct = round(n_anomalies / len(df) * 100, 2)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows", f"{len(df):,}")
        m2.metric("Anomalies Found", f"{n_anomalies:,}")
        m3.metric("Anomaly Rate", f"{pct}%")
        m4.metric("Normal Rows", f"{len(normal_df):,}")

        st.session_state["anomaly_df"] = anomaly_df
        st.session_state["anomaly_details"] = details
        st.session_state["anomaly_pct"] = pct

        # Visualise
        if len(cols_sel) >= 2:
            plot_df = df[cols_sel[:2]].copy().reset_index(drop=True)
            plot_df["Type"] = anomaly_mask.reset_index(drop=True).map(
                {True: "Anomaly", False: "Normal"}
            )
            fig = px.scatter(plot_df, x=cols_sel[0], y=cols_sel[1],
                             color="Type",
                             color_discrete_map={"Anomaly":"#f43f5e","Normal":"rgba(201,168,76,.4)"},
                             title=f"Anomaly Map — {cols_sel[0]} vs {cols_sel[1]}", **DARK)
            st.plotly_chart(fig, use_container_width=True)

        if n_anomalies > 0:
            st.subheader("🚨 Anomalous Records")
            st.dataframe(anomaly_df.head(50), use_container_width=True, hide_index=True)

            csv = anomaly_df.to_csv(index=False).encode()
            st.download_button("📥 Download Anomalies CSV", csv,
                               "anomalies.csv", "text/csv", use_container_width=True)

        # AI Business Alert
        if can_proceed() and n_anomalies > 0:
            st.markdown("---")
            st.subheader("🤖 AI Business Alert")
            with st.spinner("Generating business alert…"):
                try:
                    ai = AIClient()
                    sample = anomaly_df[cols_sel].head(10).to_string()
                    prompt = f"""Dataset: {st.session_state.get('filename','data')}
Total rows: {len(df):,} | Anomalies: {n_anomalies} ({pct}%)
Detection method: {method}
Columns analyzed: {cols_sel}
Column stats: {details}
Sample anomalous records:
{sample}

Write a business alert in {lang} with:
1. SEVERITY level (Critical/High/Medium/Low)
2. What the anomalies mean in business terms
3. Potential causes
4. Immediate recommended actions
5. Monitoring suggestions

Be specific with numbers. Write for a business manager."""

                    alert = ai.generate(prompt, system="You are a senior business intelligence analyst.")
                    severity_color = "#f43f5e" if pct > 10 else ("#f59e0b" if pct > 5 else "#10b981")
                    st.markdown(f"""
                    <div class="insight-box" style="border-left-color:{severity_color};">
                    {alert}
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"AI alert error: {e}")
        elif n_anomalies == 0:
            st.success("✅ No anomalies detected. Your data looks clean!")
