"""
Vision AI Page — analyze image datasets with Ollama vision models
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.ui_components import load_all_styles
load_all_styles("assets")

from src.vision_analyzer import (
    get_available_vision_models, analyze_image_stream,
    scan_image_folder, batch_analyze, VISION_MODELS, IMAGE_EXTENSIONS
)

st.markdown("""
<div class='app-header'>
    <h1>🖼️ Vision <span>AI</span></h1>
    <p>Analyze image datasets &nbsp;·&nbsp; Free local Ollama vision models</p>
</div>
""", unsafe_allow_html=True)

# ── Model setup ───────────────────────────────────────────────────────────────
st.subheader("🧠 Vision Model Setup")

available = get_available_vision_models()

if not available:
    st.warning("No vision models found locally.")
    st.markdown("Pull one of these free models:")
    for model, desc in VISION_MODELS.items():
        st.code(f"ollama pull {model}")
        st.caption(desc)
    st.stop()

col_m, col_i = st.columns(2)
with col_m:
    selected_model = st.selectbox("🤖 Vision Model", available)
with col_i:
    st.metric("Model loaded", selected_model)

st.markdown("---")

# ── Mode selection ────────────────────────────────────────────────────────────
mode = st.radio(
    "Mode",
    ["📁 Folder Analysis", "🖼️ Single Image Chat"],
    horizontal=True
)

# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — FOLDER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if mode == "📁 Folder Analysis":
    st.subheader("📁 Image Folder Analysis")

    folder_path = st.text_input(
        "📂 Enter full folder path",
        placeholder="e.g. C:\\Users\\student\\Desktop\\images"
    )

    if folder_path:
        images = scan_image_folder(folder_path)

        if not images:
            st.error("No images found in that folder. "
                     f"Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
        else:
            st.success(f"✅ Found {len(images)} images")

            import pandas as pd
            img_df = pd.DataFrame(images)[["filename","ext","size_kb"]]
            st.dataframe(img_df, use_container_width=True, hide_index=True)

            # Preview grid
            st.subheader("🖼️ Image Preview")
            preview_n = min(6, len(images))
            cols = st.columns(3)
            for i, img in enumerate(images[:preview_n]):
                with cols[i % 3]:
                    try:
                        st.image(img["path"], caption=img["filename"],
                                 use_container_width=True)
                    except Exception:
                        st.write(img["filename"])

            st.markdown("---")

            # Analysis options
            st.subheader("🤖 Batch AI Analysis")

            analysis_type = st.selectbox("Analysis Type", [
                "General description",
                "Classify the image content",
                "Detect objects present",
                "Describe colors and composition",
                "Identify any text in the image",
                "Custom prompt",
            ])

            prompts = {
                "General description":         "Describe this image in detail.",
                "Classify the image content":  "What category does this image belong to? Give a single label.",
                "Detect objects present":       "List all objects visible in this image.",
                "Describe colors and composition": "Describe the colors, lighting, and composition of this image.",
                "Identify any text in the image": "Extract and list all text visible in this image.",
                "Custom prompt": "",
            }

            if analysis_type == "Custom prompt":
                prompt = st.text_area("Enter your prompt:", height=80)
            else:
                prompt = prompts[analysis_type]
                st.info(f"Prompt: *{prompt}*")

            max_images = st.slider(
                "Max images to analyze", 1, len(images), min(5, len(images))
            )

            if st.button("▶️ Analyze Images", type="primary",
                         use_container_width=True):
                if not prompt:
                    st.warning("Please enter a prompt.")
                else:
                    paths = [img["path"] for img in images[:max_images]]
                    results = []
                    progress = st.progress(0)
                    status   = st.empty()

                    for i, path in enumerate(paths):
                        fname = Path(path).name
                        status.text(f"Analyzing {fname}… ({i+1}/{len(paths)})")
                        result = ""
                        for chunk in analyze_image_stream(path, prompt, selected_model):
                            result += chunk
                        results.append({"Image": fname, "Analysis": result})
                        progress.progress((i+1) / len(paths))

                    progress.empty()
                    status.empty()

                    result_df = pd.DataFrame(results)
                    st.success(f"✅ Analyzed {len(results)} images!")
                    st.dataframe(result_df, use_container_width=True,
                                 hide_index=True)

                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Results CSV", csv,
                        file_name="vision_analysis.csv", mime="text/csv",
                        use_container_width=True
                    )

# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — SINGLE IMAGE CHAT
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.subheader("🖼️ Chat with a Single Image")

    uploaded_img = st.file_uploader(
        "Upload an image",
        type=["jpg","jpeg","png","bmp","webp"]
    )

    if uploaded_img:
        import tempfile
        import os

        # Save to temp file
        suffix = Path(uploaded_img.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_img.read())
            tmp_path = tmp.name

        col_img, col_chat = st.columns([1, 2])

        with col_img:
            st.image(tmp_path, caption=uploaded_img.name,
                     use_container_width=True)
            st.caption(f"Size: {os.path.getsize(tmp_path)//1024} KB")

        with col_chat:
            # Chat history for this image
            if "vision_chat" not in st.session_state:
                st.session_state.vision_chat = []

            for msg in st.session_state.vision_chat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            vision_input = st.chat_input("Ask anything about this image…")

            if vision_input:
                with st.chat_message("user"):
                    st.markdown(vision_input)
                st.session_state.vision_chat.append(
                    {"role": "user", "content": vision_input}
                )

                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    response    = ""
                    for chunk in analyze_image_stream(
                        tmp_path, vision_input, selected_model
                    ):
                        response += chunk
                        placeholder.markdown(response + "▌")
                    placeholder.markdown(response)

                st.session_state.vision_chat.append(
                    {"role": "assistant", "content": response}
                )

            if st.session_state.vision_chat:
                if st.button("🗑️ Clear Chat", key="clear_vision"):
                    st.session_state.vision_chat = []
                    st.rerun()

            with st.expander("💡 Suggested questions"):
                for s in [
                    "What is in this image?",
                    "Describe the colors and mood.",
                    "What objects can you identify?",
                    "Is there any text in this image?",
                    "What category does this image belong to?",
                    "Describe this image for someone who cannot see it.",
                ]:
                    if st.button(s, key=f"vis_{s}"):
                        st.session_state.vision_chat.append(
                            {"role": "user", "content": s}
                        )
                        st.rerun()