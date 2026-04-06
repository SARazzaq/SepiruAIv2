"""
Deep Data Profiling Page
"""

import streamlit as st
import pandas as pd

from src.ui_components import load_all_styles
load_all_styles("assets")

from src.profiler import full_profile

st.markdown("""
<div class='app-header'>
    <h1>📊 Data <span>Profiling</span></h1>
    <p>Skewness &nbsp;·&nbsp; Kurtosis &nbsp;·&nbsp; Correlations &nbsp;·&nbsp; Deep statistical analysis</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.get("df") is None:
    st.warning("⚠️ No data loaded. Please upload a file on the main page first.")
    st.stop()

df = st.session_state.clean_df

if st.button("▶️ Generate Full Profile", type="primary", use_container_width=True):
    with st.spinner("Profiling dataset…"):
        profile = full_profile(df)
        st.session_state["profile"] = profile

if "profile" not in st.session_state:
    st.info("👆 Click the button above to generate the full profile.")
    st.stop()

profile = st.session_state["profile"]
ov = profile["overview"]

# ── Overview cards ────────────────────────────────────────────────────────────
st.subheader("📋 Overview")
cols = st.columns(4)
for col, (label, val) in zip(cols, [
    ("Rows",         f"{ov['rows']:,}"),
    ("Columns",      str(ov["cols"])),
    ("Missing %",    f"{ov['missing_pct']}%"),
    ("Memory",       f"{ov['memory_mb']} MB"),
]):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{label}</h3><h2>{val}</h2>
        </div>""", unsafe_allow_html=True)

st.markdown("")
cols2 = st.columns(4)
for col, (label, val) in zip(cols2, [
    ("Duplicates",     f"{ov['duplicate_rows']:,}"),
    ("Numeric cols",   str(ov["numeric_cols"])),
    ("Categorical",    str(ov["cat_cols"])),
    ("Datetime cols",  str(ov["datetime_cols"])),
]):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{label}</h3><h2>{val}</h2>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Column stats ──────────────────────────────────────────────────────────────
st.subheader("📐 Column Statistics")
st.dataframe(profile["col_stats"], use_container_width=True, hide_index=True)

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
if profile["dist_fig"]:
    st.subheader("📊 Distribution Grid")
    st.plotly_chart(profile["dist_fig"], use_container_width=True)

if profile["skew_fig"]:
    st.subheader("📉 Skewness Analysis")
    st.plotly_chart(profile["skew_fig"], use_container_width=True)
    st.info("Values > 1 or < -1 indicate high skewness — consider log transformation.")

if profile["corr_fig"]:
    st.subheader("🔗 Correlation Heatmap")
    st.plotly_chart(profile["corr_fig"], use_container_width=True)

if profile["strong_corr"] is not None:
    st.subheader("⚡ Strong Correlations (|r| > 0.5)")
    st.dataframe(profile["strong_corr"], use_container_width=True, hide_index=True)

if profile["missing_fig"]:
    st.subheader("❓ Missing Values Pattern")
    st.plotly_chart(profile["missing_fig"], use_container_width=True)
else:
    st.success("✅ No missing values in this dataset!")