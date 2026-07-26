"""
LangFuse Observability — trace every LLM call with latency & token info.
Open-source, MIT license. Self-host or use cloud free tier.
"""
import streamlit as st
import time
from src.ui_components import load_all_styles
from src.auth import require_auth
from src.ai_client import AIClient
from src.langfuse_tracer import LangFuseTracer

load_all_styles("assets")
require_auth()

st.markdown("""
<div class='app-header'>
    <h1>📡 LangFuse <span>Tracing</span></h1>
    <p>Observe every LLM call — latency · tokens · inputs · outputs</p>
</div>""", unsafe_allow_html=True)

# ── Config panel ───────────────────────────────────────────────────────────────
import os
with st.expander("⚙️ LangFuse Configuration", expanded=not bool(os.getenv("LANGFUSE_SECRET_KEY"))):
    st.markdown("""
**Self-host (free, no account needed):**
```bash
docker run -d -p 3000:3000 langfuse/langfuse:latest
```
Then set `LANGFUSE_HOST=http://localhost:3000` in your `.env`.

**Cloud free tier:** Get keys at [cloud.langfuse.com](https://cloud.langfuse.com) (free, no CC needed).
    """)
    c1, c2, c3 = st.columns(3)
    with c1:
        pk = st.text_input("Public Key", os.getenv("LANGFUSE_PUBLIC_KEY", ""), type="password")
    with c2:
        sk = st.text_input("Secret Key", os.getenv("LANGFUSE_SECRET_KEY", ""), type="password")
    with c3:
        host = st.text_input("Host", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))

    if st.button("💾 Apply to session"):
        if pk: os.environ["LANGFUSE_PUBLIC_KEY"] = pk
        if sk: os.environ["LANGFUSE_SECRET_KEY"] = sk
        if host: os.environ["LANGFUSE_HOST"] = host
        st.success("Config applied.")

# ── Tracer init ────────────────────────────────────────────────────────────────
try:
    tracer = LangFuseTracer()
    if tracer.enabled:
        st.success(f"✅ LangFuse connected · Session: `{tracer._session_id[:8]}…`")
    else:
        st.warning("⚠️ LangFuse not configured — running in no-op mode (calls still work, just not traced).")
except Exception as e:
    st.error(f"LangFuse init error: {e}")
    tracer = None

# ── Test prompt ────────────────────────────────────────────────────────────────
st.subheader("🧪 Send a Traced LLM Call")

if st.session_state.get("df") is not None:
    df = st.session_state.clean_df
    default_prompt = f"Summarise this dataset with {len(df)} rows and {len(df.columns)} columns: {', '.join(df.columns[:6])}"
else:
    default_prompt = "What are best practices for data analysis?"

prompt = st.text_area("Prompt", default_prompt, height=100)
trace_name = st.text_input("Trace name", "manual-test")

if st.button("▶️ Send & Trace", type="primary"):
    if tracer is None:
        st.error("Tracer not initialised.")
    else:
        ai = AIClient()
        with st.spinner("Generating…"):
            t0 = time.time()
            response = tracer.wrap_generate(ai, prompt, name=trace_name)
            latency = int((time.time() - t0) * 1000)

        st.markdown("**Response:**")
        st.info(response)
        st.caption(f"⏱ Latency: {latency}ms · Provider: {ai.provider} · Model: {ai.model}")

        if tracer.enabled:
            stats = tracer.get_session_stats()
            st.success(f"✅ Trace logged to LangFuse · Session: `{stats['session_id'][:8]}…`")
            st.markdown(f"View at: [{stats['host']}]({stats['host']})")
        else:
            st.info("Trace not sent (LangFuse not configured). Configure above to enable tracing.")

# ── Trace history in session ───────────────────────────────────────────────────
st.subheader("📋 Live Trace Log (this session)")
if "lf_trace_log" not in st.session_state:
    st.session_state.lf_trace_log = []

if tracer and tracer.enabled:
    st.caption("Traces are also being sent to your LangFuse dashboard.")
else:
    st.caption("Showing local log only — connect LangFuse to persist traces.")

with st.sidebar:
    st.markdown("### ℹ️ What gets traced")
    st.markdown("""
- Input prompt (truncated)
- Output response
- Model & provider name
- Latency (ms)
- Session grouping
    """)
    st.markdown("### 🔗 Resources")
    st.markdown("[LangFuse Docs](https://langfuse.com/docs) · [GitHub](https://github.com/langfuse/langfuse)")
