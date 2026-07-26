"""
AutoGen Multi-Agent Analysis — two AI agents collaborate on your data.
Microsoft AutoGen, MIT license. 100% free & open-source.
"""
import streamlit as st
import pandas as pd
from src.ui_components import load_all_styles
from src.auth import require_auth
from src.ai_client import AIClient
from src.smart_context import extract_relevant_context

load_all_styles("assets")
require_auth()

st.markdown("""
<div class='app-header'>
    <h1>🤝 AutoGen <span>Multi-Agent</span></h1>
    <p>DataAnalyst + Critic agents collaborate to answer your questions</p>
</div>""", unsafe_allow_html=True)

if st.session_state.get("df") is None:
    st.warning("Upload a file on the main page first.")
    st.stop()

df = st.session_state.clean_df

try:
    import autogen
    st.caption(f"AutoGen {autogen.__version__} ready")
except ImportError:
    st.error("AutoGen not installed.")
    st.code("pip install pyautogen")
    st.stop()

# ── Question input ─────────────────────────────────────────────────────────────
st.subheader("💬 Pose a question to the agent team")

suggested = [
    f"What are the key drivers of {df.columns[-1]} in this dataset?",
    "Identify the top 3 insights and any anomalies in this data.",
    "What business recommendations can you make from this data?",
    f"Is there a strong correlation between {df.columns[0]} and {df.columns[-1]}?",
]

col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_area("Your question", suggested[0], height=80)
with col2:
    st.markdown("**Quick picks:**")
    for s in suggested[1:]:
        if st.button(s[:45] + "…", key=s):
            question = s

max_rounds = st.slider("Max agent rounds", 2, 6, 3)

if st.button("🚀 Start Multi-Agent Analysis", type="primary"):
    ai = AIClient()

    with st.spinner("Extracting data context…"):
        context = extract_relevant_context(df, question)

    st.info(f"Context extracted: {len(context)} chars → sending to agents")

    with st.spinner("Agents collaborating… (this may take 30-60s)"):
        try:
            from src.autogen_agents import run_autogen_analysis
            result = run_autogen_analysis(context, question, ai, max_rounds=max_rounds)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"AutoGen error: {e}")
            st.stop()

    st.subheader("🧠 Agent Conversation")

    # Parse and display each agent's turn
    for block in result.split("\n\n"):
        if block.strip():
            if block.startswith("[DataAnalyst]"):
                with st.chat_message("assistant", avatar="🔬"):
                    st.markdown(block.replace("[DataAnalyst]: ", "**DataAnalyst:**\n"))
            elif block.startswith("[Critic]"):
                with st.chat_message("assistant", avatar="🔍"):
                    st.markdown(block.replace("[Critic]: ", "**Critic:**\n"))
            else:
                st.markdown(block)

    # Extract final answer
    if "FINAL_ANSWER:" in result:
        final = result.split("FINAL_ANSWER:")[-1].strip()
        st.success(f"✅ **Final Answer:** {final}")

with st.sidebar:
    st.markdown("### 🤖 Agent Roles")
    st.markdown("""
**DataAnalyst**
- Interprets data context
- Identifies patterns & insights
- Forms structured conclusions

**Critic**
- Reviews analyst's response
- Checks accuracy & completeness
- Approves or requests revisions
    """)
    st.markdown("### ⚙️ Provider support")
    st.markdown("Works with: Groq · OpenAI · Ollama · vLLM")
