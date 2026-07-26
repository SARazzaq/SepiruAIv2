"""
AI Business Report Generator — powered by Gemini 1.5 Flash (free).
Generates executive-ready PDF-style reports from CSV data.
"""
import streamlit as st
import pandas as pd
import io
from src.ui_components import load_all_styles
from src.quota_guard import can_proceed, get_usage, reset_time_utc

load_all_styles("assets")

from src.auth import require_auth
require_auth()

from src.ai_client import AIClient
from src.data_analyzer import DataAnalyzer

st.markdown("""
<div class='app-header'>
    <h1>📋 Business <span>Report</span></h1>
    <p>AI-generated executive reports from your data</p>
</div>""", unsafe_allow_html=True)

if not can_proceed():
    st.error(f"⏳ Daily AI quota reached. Resets in **{reset_time_utc()}**. Come back then!")
    st.stop()

if st.session_state.get("df") is None:
    st.warning("Upload a file on the main page first.")
    st.stop()

df = st.session_state.clean_df
analyzer = DataAnalyzer(df)

u = get_usage()
st.caption(f"AI quota: {u['count']}/{u['limit']} used today · {u['remaining']} remaining")

st.subheader("⚙️ Report Configuration")
c1, c2, c3 = st.columns(3)
with c1:
    report_type = st.selectbox("Report Type", [
        "Executive Summary", "Sales Performance", "Financial Analysis",
        "Operations Review", "Marketing Insights", "Risk Assessment"
    ])
with c2:
    audience = st.selectbox("Audience", ["CEO/Board", "Management", "Analysts", "General"])
with c3:
    language = st.selectbox("Language", [
        "English", "Hindi", "Spanish", "French", "German",
        "Arabic", "Chinese", "Japanese", "Portuguese"
    ])

tone = st.radio("Tone", ["Professional", "Concise", "Detailed"], horizontal=True)

if st.button("▶️ Generate Report", type="primary", use_container_width=True):
    if not can_proceed():
        st.error(f"Quota exhausted. Resets in {reset_time_utc()}.")
        st.stop()

    with st.spinner("Generating your business report…"):
        try:
            ai = AIClient()
            summary = analyzer.get_data_summary()
            stats = df.describe().round(2).to_string()

            prompt = f"""You are a senior business analyst. Generate a {report_type} in {language}.

Dataset: {st.session_state.get('filename','data')}
Rows: {df.shape[0]:,} | Columns: {df.shape[1]}
Summary: {summary}
Statistics: {stats}

Write a {tone.lower()} {report_type} for {audience}.
Structure: Executive Summary → Key Findings (3-5 bullet points) → Data Insights → Recommendations → Conclusion.
Use specific numbers from the data. Write in {language}. Be impactful and actionable."""

            report = ai.generate(prompt, system=f"You are an expert business analyst writing for {audience}.")

            st.markdown("---")
            st.subheader("📄 Generated Report")
            st.markdown(f"""
            <div class="insight-box" style="white-space:pre-wrap;line-height:1.9;">
            {report}
            </div>""", unsafe_allow_html=True)

            # Download as text
            st.download_button(
                "📥 Download Report",
                data=report.encode("utf-8"),
                file_name=f"sepiru_report_{report_type.lower().replace(' ','_')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error: {e}")
