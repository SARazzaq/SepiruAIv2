"""
vLLM Local Inference — run open-source LLMs at high throughput.
vLLM runs on your own machine (GPU). This page is a client/playground.
"""
import streamlit as st
import requests
import os
from src.ui_components import load_all_styles

load_all_styles("assets")

st.markdown("""
<div class='app-header'>
    <h1>⚡ vLLM <span>Local Inference</span></h1>
    <p>High-throughput local LLM serving · OpenAI-compatible · Apache 2.0</p>
</div>""", unsafe_allow_html=True)

st.info("vLLM runs on **your own machine** (GPU recommended). This page connects to it as a client. No install needed here.")

with st.expander("🚀 How to start vLLM on your machine", expanded=True):
    st.code("""# Install vLLM (free, Apache 2.0)
pip install vllm

# Start server (GPU)
python -m vllm.entrypoints.openai.api_server \\
    --model mistralai/Mistral-7B-Instruct-v0.2 --port 8000

# CPU-only (slower)
python -m vllm.entrypoints.openai.api_server \\
    --model facebook/opt-125m --device cpu --port 8000""", language="bash")

    st.markdown("Then in `.env` or Streamlit secrets:")
    st.code("AI_PROVIDER=vllm\nVLLM_HOST=http://your-machine-ip:8000\nVLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2")

# ── Connection ─────────────────────────────────────────────────────────────────
st.subheader("🔌 Connection")
vllm_host = st.text_input("vLLM Host URL", os.getenv("VLLM_HOST", "http://localhost:8000"))

if st.button("Test Connection"):
    try:
        r = requests.get(f"{vllm_host}/v1/models", timeout=5)
        if r.status_code == 200:
            models = r.json().get("data", [])
            st.success(f"✅ Connected — {len(models)} model(s) loaded:")
            for m in models:
                st.code(m["id"])
        else:
            st.error(f"Server responded {r.status_code}")
    except Exception as e:
        st.warning(f"Not reachable: {e}")

# ── Playground ─────────────────────────────────────────────────────────────────
st.subheader("🧪 Inference Playground")
vllm_model = st.text_input("Model", os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"))

if st.session_state.get("df") is not None:
    df = st.session_state.clean_df
    default_q = f"This dataset has {len(df)} rows, columns: {', '.join(df.columns[:6])}. What analysis would you suggest?"
else:
    default_q = "Explain the difference between supervised and unsupervised learning."

user_prompt = st.text_area("Prompt", default_q, height=100)
c1, c2 = st.columns(2)
with c1: max_tokens = st.slider("Max tokens", 100, 2000, 500)
with c2: temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.05)

if st.button("⚡ Generate", type="primary"):
    try:
        payload = {
            "model": vllm_model,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        with st.spinner("Generating…"):
            r = requests.post(f"{vllm_host}/v1/chat/completions", json=payload, timeout=120)
        if r.status_code == 200:
            st.info(r.json()["choices"][0]["message"]["content"])
            usage = r.json().get("usage", {})
            if usage:
                c1, c2, c3 = st.columns(3)
                c1.metric("Prompt tokens", usage.get("prompt_tokens", "?"))
                c2.metric("Completion tokens", usage.get("completion_tokens", "?"))
                c3.metric("Total tokens", usage.get("total_tokens", "?"))
        else:
            st.error(f"vLLM error {r.status_code}: {r.text}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to vLLM. Start the server first (see guide above).")
    except Exception as e:
        st.error(f"Error: {e}")

with st.sidebar:
    st.markdown("### Why vLLM?\n- PagedAttention — 24x throughput\n- OpenAI-compatible API\n- 50+ open models\n- Apache 2.0 — free forever")
    st.markdown("[vLLM GitHub](https://github.com/vllm-project/vllm)")
