"""
LangFuse Observability — trace every LLM call.
Cloud free tier works on Streamlit Cloud. Self-host works locally.
"""
import streamlit as st
import time
import os
from src.ui_components import load_all_styles
from src.ai_client import AIClient

load_all_styles("assets")

st.markdown("""
<div class='app-header'>
    <h1>📡 LangFuse <span>Tracing</span></h1>
    <p>Observe every LLM call — latency · inputs · outputs</p>
</div>""", unsafe_allow_html=True)

# ── Dependency check ───────────────────────────────────────────────────────────
try:
    from src.langfuse_tracer import LangFuseTracer
    _lf_available = True
except ImportError:
    _lf_available = False

if not _lf_available:
    st.warning("⚠️ LangFuse is not installed in this environment.")
    st.code("pip install langfuse")
    st.stop()

# ── Config ─────────────────────────────────────────────────────────────────────
with st.expander("⚙️ LangFuse Configuration", expanded=not bool(os.getenv("LANGFUSE_SECRET_KEY"))):
    st.markdown("""
**Cloud free tier** (works on Streamlit Cloud — no CC needed):
Get keys at [cloud.langfuse.com](https://cloud.langfuse.com)

**Self-host** (local only):
```bash
docker run -d -p 3000:3000 langfuse/langfuse:latest
```
    """)
    c1, c2, c3 = st.columns(3)
    with c1:
        pk = st.text_input("Public Key", os.getenv("LANGFUSE_PUBLIC_KEY", ""), type="password")
    with c2:
        sk = st.text_input("Secret Key", os.getenv("LANGFUSE_SECRET_KEY", ""), type="password")
    with c3:
        host = st.text_input("Host", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    if st.button("💾 Apply"):
        if pk: os.environ["LANGFUSE_PUBLIC_KEY"] = pk
        if sk: os.environ["LANGFUSE_SECRET_KEY"] = sk
        if host: os.environ["LANGFUSE_HOST"] = host
        st.success("Applied.")

tracer = LangFuseTracer()
if tracer.enabled:
    st.success(f"✅ LangFuse connected · Session `{tracer._session_id[:8]}…`")
else:
    st.info("LangFuse not configured — calls work but won't be traced. Add keys above to enable.")

# ── Test prompt ────────────────────────────────────────────────────────────────
st.subheader("🧪 Send a Traced LLM Call")

if st.session_state.get("df") is not None:
    df = st.session_state.clean_df
    default_prompt = f"Summarise this dataset: {len(df)} rows, columns: {', '.join(df.columns[:6])}"
else:
    default_prompt = "What are best practices for data analysis?"

prompt = st.text_area("Prompt", default_prompt, height=100)
trace_name = st.text_input("Trace name", "manual-test")

if st.button("▶️ Send & Trace", type="primary"):
    ai = AIClient()
    with st.spinner("Generating…"):
        t0 = time.time()
        response = tracer.wrap_generate(ai, prompt, name=trace_name)
        latency = int((time.time() - t0) * 1000)
    st.info(response)
    st.caption(f"⏱ {latency}ms · {ai.provider} · {ai.model}")
    if tracer.enabled:
        stats = tracer.get_session_stats()
        st.success(f"✅ Traced · View at [{stats['host']}]({stats['host']})")

with st.sidebar:
    st.markdown("### What gets traced\n- Prompt & response\n- Model & provider\n- Latency\n- Session grouping")
    st.markdown("[LangFuse GitHub](https://github.com/langfuse/langfuse)")
