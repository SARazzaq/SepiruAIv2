"""
PyTorch Deep Learning — train a tabular MLP on your CSV data.
Runs locally — requires: pip install torch
"""
import streamlit as st
import plotly.graph_objects as go
from src.ui_components import load_all_styles

load_all_styles("assets")

st.markdown("""
<div class='app-header'>
    <h1>🧠 PyTorch <span>Deep Learning</span></h1>
    <p>Train a neural network on your tabular data — GPU-accelerated when available</p>
</div>""", unsafe_allow_html=True)

# ── Dependency check ───────────────────────────────────────────────────────────
try:
    import torch
    _torch_available = True
    device_label = "GPU ✅" if torch.cuda.is_available() else "CPU"
    st.caption(f"PyTorch {torch.__version__} · Device: {device_label}")
except ImportError:
    _torch_available = False

if not _torch_available:
    st.warning("⚠️ PyTorch is not installed in this environment.")
    st.markdown("""
This feature requires PyTorch which exceeds Streamlit Cloud's free-tier resource limits.

**Run it locally:**
```bash
# CPU (lightweight)
pip install torch --index-url https://download.pytorch.org/whl/cpu
streamlit run app.py
```
    """)
    st.stop()

if st.session_state.get("df") is None:
    st.warning("Upload a file on the main page first.")
    st.stop()

import numpy as np
import pandas as pd
df = st.session_state.clean_df

st.subheader("⚙️ Configuration")
c1, c2, c3 = st.columns(3)
with c1:
    target_col = st.selectbox("🎯 Target column", df.columns)
with c2:
    task = st.selectbox("Task", ["classification", "regression"])
with c3:
    epochs = st.slider("Epochs", 5, 100, 30, step=5)

c4, c5 = st.columns(2)
with c4:
    lr = st.select_slider("Learning rate", [0.0001, 0.001, 0.005, 0.01, 0.05], value=0.001)
with c5:
    hidden_str = st.text_input("Hidden layers (comma-sep)", "128,64,32")

hidden = [int(x.strip()) for x in hidden_str.split(",") if x.strip().isdigit()]

if st.button("🚀 Train Neural Network", type="primary"):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from src.pytorch_models import train_pytorch_tabular, pytorch_score

    feature_cols = [c for c in df.columns if c != target_col]
    X_raw = df[feature_cols].copy()
    for col in X_raw.select_dtypes(include=["object", "category"]).columns:
        X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))
    X = StandardScaler().fit_transform(X_raw.fillna(0).values.astype(float))
    y_raw = df[target_col].values
    if task == "classification":
        y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(float)
    else:
        y = y_raw.astype(float)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    prog = st.progress(0, text="Training…")

    def progress_cb(epoch, loss):
        prog.progress(int(epoch / epochs * 100), text=f"Epoch {epoch}/{epochs} — loss: {loss:.4f}")

    with st.spinner("Training PyTorch model…"):
        model, train_losses, val_losses = train_pytorch_tabular(
            X_train, y_train, X_val, y_val,
            task=task, epochs=epochs, lr=lr, hidden=hidden,
            progress_cb=progress_cb,
        )

    prog.progress(100, text="Done!")
    scores = pytorch_score(model, X_val, y_val, task)
    for k, v in scores.items():
        st.metric(k.upper(), f"{v:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, name="Train Loss", line=dict(color="#c9a84c")))
    fig.add_trace(go.Scatter(y=val_losses, name="Val Loss", line=dict(color="#6366f1", dash="dash")))
    fig.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss",
                      template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,26,0.6)")
    st.plotly_chart(fig, use_container_width=True)
    st.session_state["pytorch_model"] = model
    st.success("✅ Model trained and saved to session.")

with st.sidebar:
    st.markdown("### Architecture\n- MLP + BatchNorm + Dropout\n- Adam + CosineAnnealing\n- Auto GPU/CPU")
