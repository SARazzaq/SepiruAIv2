"""
vLLM Local Inference — run open-source LLMs at high throughput.
Apache 2.0 license. Free forever.

vLLM exposes an OpenAI-compatible API, so it's a drop-in replacement.
"""
import streamlit as st
import requests
import os
from src.ui_components import load_all_styles
from src.auth import require_auth

load_all_styles("assets")
require_auth()

st.markdown("""
<div class='app-header'>
    <h1>⚡ vLLM <span>Local Inference</span></h1>
    <p>High-throughput local LLM serving · OpenAI-compatible API · Apache 2.0</p>
</div>""", unsafe_allow_html=True)

# ── Setup guide ────────────────────────────────────────────────────────────────
with st.expander("🚀 How to start vLLM (one command)", expanded=True):
    st.markdown("**Requirements:** Python 3.9+, CUDA GPU (or CPU with `--device cpu`)")
    st.code("""# Install vLLM (free, Apache 2.0)
pip install vllm

# Start server with any HuggingFace model
python -m vllm.entrypoints.openai.api_server \\
    --model mistralai/Mistral-7B-Instruct-v0.2 \\
    --port 8000

# CPU-only mode (slower but works without GPU)
python -m vllm.entrypoints.openai.api_server \\
    --model facebook/opt-125m \\
    --device cpu --port 8000""", language="bash")

    st.markdown("Then set in your `.env`:")
    st.code("AI_PROVIDER=vllm\nVLLM_HOST=http://localhost:8000\nVLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2")

# ── Connection test ────────────────────────────────────────────────────────────
st.subheader("🔌 Connection")

vllm_host = st.text_input("vLLM Host", os.getenv("VLLM_HOST", "http://localhost:8000"))
if st.button("Test Connection"):
    try:
        r = requests.get(f"{vllm_host}/v1/models", timeout=5)
        if r.status_code == 200:
            models = r.json().get("data", [])
            st.success(f"✅ Connected! {len(models)} model(s) loaded:")
            for m in models:
                st.code(m["id"])
        else:
            st.error(f"Server responded with {r.status_code}")
    except Exception as e:
        st.warning(f"Not reachable: {e}\nStart vLLM using the command above.")

# ── Inference playground ───────────────────────────────────────────────────────
st.subheader("🧪 Inference Playground")

vllm_model = st.text_input("Model", os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"))
system_msg = st.text_input("System message", "You are a helpful data analyst assistant.")

if st.session_state.get("df") is not None:
    df = st.session_state.clean_df
    default_q = f"This dataset has {len(df)} rows and columns: {', '.join(df.columns[:6])}. What analysis would you suggest?"
else:
    default_q = "Explain the difference between supervised and unsupervised learning."

user_prompt = st.text_area("User prompt", default_q, height=100)
max_tokens = st.slider("Max tokens", 100, 2000, 500)
temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.05)

if st.button("⚡ Generate", type="primary"):
    try:
        payload = {
            "model": vllm_model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        with st.spinner("Generating…"):
            r = requests.post(f"{vllm_host}/v1/chat/completions", json=payload, timeout=120)
        if r.status_code == 200:
            result = r.json()["choices"][0]["message"]["content"]
            st.markdown("**Response:**")
            st.info(result)

            usage = r.json().get("usage", {})
            if usage:
                c1, c2, c3 = st.columns(3)
                c1.metric("Prompt tokens", usage.get("prompt_tokens", "?"))
                c2.metric("Completion tokens", usage.get("completion_tokens", "?"))
                c3.metric("Total tokens", usage.get("total_tokens", "?"))
        else:
            st.error(f"vLLM error {r.status_code}: {r.text}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to vLLM. Start the server first (see setup guide above).")
    except Exception as e:
        st.error(f"Error: {e}")

# ── Use as main provider ───────────────────────────────────────────────────────
st.subheader("🔄 Use vLLM as main provider")
st.markdown("""
To route all app AI calls through vLLM, update your `.env`:
```env
AI_PROVIDER=vllm
VLLM_HOST=http://localhost:8000
VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```
Then restart the app. All pages (chat, ML insights, reports) will use your local vLLM server.
""")

with st.sidebar:
    st.markdown("### 🔥 Why vLLM?")
    st.markdown("""
- **PagedAttention** — 24x higher throughput vs naive inference
- OpenAI-compatible REST API
- Supports 50+ open models (Mistral, LLaMA, Qwen, Gemma…)
- Continuous batching
- Apache 2.0 — completely free
    """)
    st.markdown("[vLLM GitHub](https://github.com/vllm-project/vllm)")
