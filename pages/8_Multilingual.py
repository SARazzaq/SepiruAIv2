"""
Multi-language Support — full app experience in any language.
Translate data insights, column names, and chat in your native language.
"""
import streamlit as st
import pandas as pd
from src.ui_components import load_all_styles
from src.quota_guard import can_proceed, get_usage, reset_time_utc

load_all_styles("assets")

from src.auth import require_auth
require_auth()

from src.ai_client import AIClient
from src.smart_context import extract_relevant_context

LANGUAGES = {
    "English": "en", "Hindi": "hi", "Spanish": "es",
    "French": "fr", "German": "de", "Arabic": "ar",
    "Chinese (Simplified)": "zh", "Japanese": "ja",
    "Portuguese": "pt", "Russian": "ru", "Korean": "ko",
    "Italian": "it", "Dutch": "nl", "Turkish": "tr",
}

st.markdown("""
<div class='app-header'>
    <h1>🌍 Multi-language <span>AI</span></h1>
    <p>Chat with your data in any language</p>
</div>""", unsafe_allow_html=True)

if st.session_state.get("df") is None:
    st.warning("Upload a file on the main page first.")
    st.stop()

df = st.session_state.clean_df
u = get_usage()
st.caption(f"AI quota: {u['count']}/{u['limit']} used · {u['remaining']} remaining")

c1, c2 = st.columns([1, 2])
with c1:
    lang = st.selectbox("🌐 Your Language", list(LANGUAGES.keys()))
    mode = st.radio("Mode", ["Chat with Data", "Translate Insights", "Summarise Dataset"], label_visibility="visible")

with c2:
    if mode == "Chat with Data":
        st.subheader(f"💬 Chat in {lang}")

        if "ml_chat" not in st.session_state:
            st.session_state.ml_chat = []

        for msg in st.session_state.ml_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input(f"Ask anything about your data in {lang}…")

        if user_input:
            if not can_proceed():
                st.error(f"Quota exhausted. Resets in {reset_time_utc()}.")
                st.stop()

            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.ml_chat.append({"role":"user","content":user_input})

            context = extract_relevant_context(df, user_input)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                response = ""
                try:
                    ai = AIClient()
                    prompt = f"""Data context:
{context}

User question (in {lang}): {user_input}

Answer in {lang}. Be specific with numbers and column names.
If the question is not in English, still answer in {lang}."""
                    for chunk in ai.generate_stream(prompt,
                        system=f"You are a data analyst. Always respond in {lang}."):
                        response += chunk
                        placeholder.markdown(response + "▌")
                    placeholder.markdown(response)
                except Exception as e:
                    response = f"Error: {e}"
                    placeholder.markdown(response)

            st.session_state.ml_chat.append({"role":"assistant","content":response})

        if st.session_state.ml_chat:
            if st.button("🗑️ Clear", key="clear_ml"):
                st.session_state.ml_chat = []
                st.rerun()

    elif mode == "Translate Insights":
        st.subheader(f"🔄 Translate Data Insights to {lang}")
        text_to_translate = st.text_area(
            "Paste any insight or analysis text to translate",
            height=150,
            placeholder="Paste English text here…"
        )
        if st.button("Translate", type="primary", use_container_width=True) and text_to_translate:
            if not can_proceed():
                st.error(f"Quota exhausted. Resets in {reset_time_utc()}.")
                st.stop()
            with st.spinner(f"Translating to {lang}…"):
                try:
                    ai = AIClient()
                    result = ai.generate(
                        f"Translate this to {lang}. Keep all numbers and technical terms accurate:\n\n{text_to_translate}",
                        system=f"You are a professional translator. Translate accurately to {lang}."
                    )
                    st.markdown(f'<div class="insight-box">{result}</div>', unsafe_allow_html=True)
                    st.download_button("📥 Download Translation",
                                       result.encode("utf-8"),
                                       f"translation_{lang.lower()}.txt",
                                       use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

    else:  # Summarise Dataset
        st.subheader(f"📊 Dataset Summary in {lang}")
        detail = st.radio("Detail level", ["Brief", "Standard", "Detailed"], horizontal=True)

        if st.button("Generate Summary", type="primary", use_container_width=True):
            if not can_proceed():
                st.error(f"Quota exhausted. Resets in {reset_time_utc()}.")
                st.stop()
            with st.spinner(f"Generating {detail.lower()} summary in {lang}…"):
                try:
                    ai = AIClient()
                    stats = df.describe().round(2).to_string()
                    prompt = f"""Dataset: {st.session_state.get('filename','data')}
Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
Columns: {list(df.columns)}
Statistics:
{stats}

Write a {detail.lower()} summary of this dataset in {lang}.
Include: what the data is about, key statistics, notable patterns, data quality observations.
Write entirely in {lang}."""
                    summary = ai.generate(prompt,
                        system=f"You are a data analyst. Write in {lang}.")
                    st.markdown(f'<div class="insight-box">{summary}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
