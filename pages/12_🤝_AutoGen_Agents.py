"""
AutoGen Multi-Agent Analysis — two AI agents collaborate on your data.
Requires: pip install pyautogen
"""
import streamlit as st
import pandas as pd
from src.ui_components import load_all_styles
from src.ai_client import AIClient
from src.smart_context import extract_relevant_context

load_all_styles("assets")

st.markdown("""
<div class='app-header'>
    <h1>🤝 AutoGen <span>Multi-Agent</span></h1>
    <p>DataAnalyst + Critic agents collaborate to answer your questions</p>
</div>""", unsafe_allow_html=True)

# ── Dependency check ───────────────────────────────────────────────────────────
try:
    import autogen
    st.caption(f"AutoGen {autogen.__version__} ready")
    _autogen_available = True
except ImportError:
    _autogen_available = False

if not _autogen_available:
    st.warning("⚠️ AutoGen is not installed in this environment.")
    st.markdown("""
**Run it locally:**
```bash
pip install pyautogen
streamlit run app.py
```
AutoGen needs to call an LLM API multiple times per conversation, so it works best with Groq (free) or Ollama (local).
    """)
    st.stop()

if st.session_state.get("df") is None:
    st.warning("Upload a file on the main page first.")
    st.stop()

df = st.session_state.clean_df

st.subheader("💬 Pose a question to the agent team")

suggested = [
    f"What are the key drivers of {df.columns[-1]} in this dataset?",
    "Identify the top 3 insights and any anomalies in this data.",
    "What business recommendations can you make from this data?",
]

question = st.text_area("Your question", suggested[0], height=80)
for s in suggested[1:]:
    if st.button(s[:60] + "…", key=s):
        question = s

max_rounds = st.slider("Max agent rounds", 2, 6, 3)

if st.button("🚀 Start Multi-Agent Analysis", type="primary"):
    ai = AIClient()
    with st.spinner("Extracting data context…"):
        context = extract_relevant_context(df, question)

    with st.spinner("Agents collaborating… (30–60s)"):
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

    if "FINAL_ANSWER:" in result:
        final = result.split("FINAL_ANSWER:")[-1].strip()
        st.success(f"✅ **Final Answer:** {final}")

with st.sidebar:
    st.markdown("### 🤖 Agents\n**DataAnalyst** — interprets data, finds patterns\n\n**Critic** — reviews accuracy, approves or corrects")
    st.markdown("### Providers\nGroq · OpenAI · Ollama · vLLM")
