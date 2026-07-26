"""
Natural Language to SQL on CSV — ask questions, get SQL, run it on your data.
"""
import streamlit as st
import pandas as pd
import re
from src.ui_components import load_all_styles
from src.quota_guard import can_proceed, get_usage, reset_time_utc

load_all_styles("assets")

from src.auth import require_auth
require_auth()

from src.ai_client import AIClient

st.markdown("""
<div class='app-header'>
    <h1>🗄️ Natural Language <span>to SQL</span></h1>
    <p>Ask questions in plain English — get SQL — run it on your data</p>
</div>""", unsafe_allow_html=True)

if st.session_state.get("df") is None:
    st.warning("Upload a file on the main page first.")
    st.stop()

df = st.session_state.clean_df
filename = st.session_state.get("filename", "data.csv")
table_name = re.sub(r"[^a-zA-Z0-9_]", "_", filename.replace(".csv","").replace(".xlsx",""))

u = get_usage()
st.caption(f"AI quota: {u['count']}/{u['limit']} used · {u['remaining']} remaining")

# Schema display
with st.expander("📋 Table Schema", expanded=False):
    schema_rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
        schema_rows.append({"Column": col, "Type": dtype, "Sample": sample})
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

st.subheader("💬 Ask in Natural Language")

lang = st.selectbox("Question language", [
    "English","Hindi","Spanish","French","German","Arabic","Chinese","Portuguese"
], key="nl_lang")

# Suggested questions
suggestions = [
    f"Show top 10 rows by {df.columns[0]}",
    f"Count rows grouped by {df.columns[0]}",
    f"Find rows where {df.columns[0]} is maximum",
    "Show average of all numeric columns",
    "Find duplicate rows",
]
if len(df.columns) > 1:
    suggestions.append(f"Show rows where {df.columns[1]} is above average")

with st.expander("💡 Suggested questions"):
    for s in suggestions:
        if st.button(s, key=f"nl_{s}"):
            st.session_state["nl_question"] = s

question = st.text_input(
    "Your question",
    value=st.session_state.get("nl_question", ""),
    placeholder=f"e.g. Show top 5 rows by {df.columns[0]}",
    key="nl_q_input"
)

if st.button("▶️ Generate & Run SQL", type="primary", use_container_width=True) and question:
    if not can_proceed():
        st.error(f"Quota exhausted. Resets in {reset_time_utc()}.")
        st.stop()

    with st.spinner("Generating SQL…"):
        try:
            ai = AIClient()
            schema = ", ".join([f"{c} ({df[c].dtype})" for c in df.columns])
            prompt = f"""Table name: {table_name}
Schema: {schema}
Sample data (first 3 rows):
{df.head(3).to_string()}

User question (in {lang}): {question}

Generate a valid SQLite SQL query to answer this question.
Return ONLY the SQL query, nothing else. No explanation, no markdown, no backticks.
The query must work with SQLite syntax."""

            sql = ai.generate(prompt, system="You are a SQL expert. Return only valid SQLite SQL.")
            # Clean up any markdown
            sql = re.sub(r"```sql|```", "", sql).strip()

            st.subheader("📝 Generated SQL")
            st.code(sql, language="sql")

            # Execute using pandasql / duckdb
            st.subheader("📊 Results")
            try:
                import duckdb
                result = duckdb.execute(
                    sql.replace(table_name, "df")
                ).df()
                st.dataframe(result, use_container_width=True, hide_index=True)
                st.success(f"✅ {len(result):,} rows returned")

                csv = result.to_csv(index=False).encode()
                st.download_button("📥 Download Results", csv,
                                   "query_results.csv", "text/csv",
                                   use_container_width=True)
            except Exception as exec_err:
                # Fallback: try pandasql
                try:
                    from pandasql import sqldf
                    env = {table_name: df}
                    result = sqldf(sql, env)
                    st.dataframe(result, use_container_width=True, hide_index=True)
                    st.success(f"✅ {len(result):,} rows returned")
                except Exception:
                    st.error(f"SQL execution error: {exec_err}")
                    st.info("The SQL was generated but couldn't execute. Try rephrasing your question.")

        except Exception as e:
            st.error(f"Error: {e}")

# History
if "nl_history" not in st.session_state:
    st.session_state.nl_history = []

if question and st.session_state.get("nl_q_input"):
    if not st.session_state.nl_history or st.session_state.nl_history[-1] != question:
        st.session_state.nl_history.append(question)

if st.session_state.nl_history:
    with st.expander("🕐 Query History"):
        for i, q in enumerate(reversed(st.session_state.nl_history[-10:])):
            st.markdown(f"`{i+1}.` {q}")
