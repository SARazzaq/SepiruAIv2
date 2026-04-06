"""
CSV AI Analyst — Main Application
Run: streamlit run app.py
"""
import plotly.graph_objects as go
from src.animations import aurora_background, show_lottie, apex_motion_engine
from src.ui_components import load_all_styles, typing_indicator, render_section, render_insight, render_metric_cards, upload_cta, status_pill
from src.auth import require_auth
from src.smart_context import extract_relevant_context
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import io
import os
from dotenv import load_dotenv


from src.ai_client     import AIClient
from src.data_analyzer import DataAnalyzer
from src.data_cleaner  import DataCleaner
from src.forecaster    import Forecaster
from src.utils         import load_data, build_system_prompt, build_analysis_prompt

load_dotenv()

# ── Load secrets (Streamlit Cloud injects via st.secrets, local uses .env) ───
def _secret(key: str, fallback: str = "") -> str:
    """Read from st.secrets first (cloud), then env vars (local)."""
    try:
        return st.secrets.get(key, os.getenv(key, fallback))
    except Exception:
        return os.getenv(key, fallback)

# Seed env from secrets so AIClient picks them up via os.getenv
for _k in ["AI_PROVIDER","GROQ_API_KEY","GROQ_MODEL",
           "OPENAI_API_KEY","OPENAI_MODEL","ANTHROPIC_API_KEY","ANTHROPIC_MODEL"]:
    _v = _secret(_k)
    if _v:
        os.environ[_k] = _v

# Default to groq on cloud (no OLLAMA_HOST available)
if not os.environ.get("AI_PROVIDER"):
    os.environ["AI_PROVIDER"] = "groq"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSV AI Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth gate — must come right after set_page_config ────────────────────────
require_auth()

# ── Load all styles ───────────────────────────────────────────────────────────
load_all_styles("assets")

# ── Particle background ───────────────────────────────────────────────────────
#components.html(particle_background(), height=0)
components.html(aurora_background(), height=1, scrolling=False)
components.html(apex_motion_engine(), height=1, scrolling=False)


# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "df": None, "clean_df": None, "filename": None,
    "df2": None, "filename2": None,
    "ai_ready": False, "chat_history": [],
    "analysis_history": [], "compare_chat_history": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
    <h1>CSV <span>AI</span> Analyst</h1>
    <p>Upload &nbsp;·&nbsp; Clean &nbsp;·&nbsp; Visualise &nbsp;·&nbsp; Chat &nbsp;·&nbsp; Forecast &nbsp;·&nbsp; Compare</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Provider fixed to groq — no dropdown needed
    provider = "groq"
    os.environ["AI_PROVIDER"] = provider

    st.divider()
    try:
        ai = AIClient()
        ok, msg = ai.check_connection()
        if ok:
            st.markdown(status_pill(msg, online=True), unsafe_allow_html=True)
            st.session_state.ai_ready = True
        else:
            st.error(f"🔴 {msg}")
            st.session_state.ai_ready = False
    except Exception as e:
        st.error(str(e))
        st.session_state.ai_ready = False
        ai = None

    if ai and st.session_state.ai_ready:
        models = ai.get_available_models()
        if models:
            ai.model = st.selectbox("🤖 Model", models)
            st.session_state.selected_model = ai.model
            chat_model = st.selectbox(
                "💬 Chat Model",
                models,
                index=0,
                help="Used only in the Chat tab"
            )
        else:
            chat_model = ai.model
    else:
        chat_model = None

    st.divider()
    st.markdown("### 🎛️ Generation Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    max_tokens  = st.slider("Max Tokens", 500, 4000, 2000, 100)
    if ai:
        ai.temperature = temperature
        ai.max_tokens  = max_tokens

    st.divider()
    st.markdown("### 📂 Upload Data")
    uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded:
        if uploaded.name != st.session_state.get("filename"):
            df = load_data(uploaded)
            if df is not None:
                st.session_state.df       = df
                st.session_state.clean_df = df.copy()
                st.session_state.filename = uploaded.name
                st.session_state.chat_history = []
        st.success(f"✅ {st.session_state.filename}")
        st.info(f"**{st.session_state.df.shape[0]:,} rows · {st.session_state.df.shape[1]} cols**")

# ── Main ──────────────────────────────────────────────────────────────────────
if st.session_state.df is None:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        show_lottie("upload", height=180)

        st.markdown(upload_cta(
            title="Drop your data here",
            subtitle="Supports .csv and .xlsx — any size",
            features=["Overview","Visualise","Clean","Chat","Forecast","Compare"]
        ), unsafe_allow_html=True)

        center_file = st.file_uploader(
            "Drop your CSV or Excel file here",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
            key="center_upload"
        )
        if center_file:
            df = load_data(center_file)
            if df is not None:
                st.session_state.df       = df
                st.session_state.clean_df = df.copy()
                st.session_state.filename = center_file.name
                st.session_state.chat_history = []
                st.rerun()

        st.markdown("---")
        st.markdown("### 🚀 Features")
        for feat in [
            "📋 Data overview & quality report",
            "📈 Auto visualisations + custom chart builder",
            "🧹 Data cleaning — missing values, duplicates, outliers",
            "💬 Chat with your data (multi-turn AI Q&A)",
            "🔮 Forecasting — predict future values with confidence bands",
            "⚠️ Outlier detection & raw data explorer",
            "🆚 Compare two datasets side by side",
        ]:
            st.markdown(f"- {feat}")

else:
    df       = st.session_state.clean_df
    analyzer = DataAnalyzer(df)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Overview",
        "📈 Visualisations",
        "🧹 Data Cleaning",
        "💬 Chat with Data",
        "🔮 Forecasting",
        "⚠️ Outliers & Explore",
        "🆚 Compare Files",
    ])

    # ── TAB 1 — OVERVIEW ──────────────────────────────────────────────────────
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, val) in zip([c1, c2, c3, c4], [
            ("Rows",          f"{df.shape[0]:,}"),
            ("Columns",       str(df.shape[1])),
            ("Numeric cols",  str(len(analyzer.info["numeric_cols"]))),
            ("Missing cells", f"{sum(analyzer.info['missing'].values()):,}"),
        ]):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>{label}</h3><h2>{val}</h2>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🧹 Data Quality Report")
        st.dataframe(analyzer.get_quality_report(), use_container_width=True, hide_index=True)

        if analyzer.info["numeric_cols"]:
            st.subheader("📐 Statistical Summary")
            st.dataframe(df[analyzer.info["numeric_cols"]].describe().round(3),
                         use_container_width=True)

        if analyzer.info["duplicate_rows"]:
            st.warning(f"⚠️ {analyzer.info['duplicate_rows']} duplicate rows detected.")

    # ── TAB 2 — VISUALISATIONS ────────────────────────────────────────────────
    with tab2:
        num_cols = analyzer.info["numeric_cols"]
        cat_cols = analyzer.info["cat_cols"]

        DARK = dict(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,26,0.6)",
            font=dict(family="DM Sans, sans-serif", color="#9090a8", size=11),
            title_font=dict(family="DM Serif Display, serif", color="#e8e6e0", size=15),
        )

        st.subheader("🌐 3D Visualisations")

        if len(num_cols) >= 3:
            v1, v2, v3, v4 = st.columns(4)
            with v1: x3 = st.selectbox("X axis", num_cols, key="x3")
            with v2: y3 = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="y3")
            with v3: z3 = st.selectbox("Z axis", num_cols, index=min(2, len(num_cols)-1), key="z3")
            with v4:
                c3d = st.selectbox("Color by", ["None"] + list(df.columns), key="c3d")
                c3d = None if c3d == "None" else c3d

            chart3d = st.selectbox("3D Chart Type", [
                "3D Scatter", "3D Line", "3D Surface", "3D Bar (Mesh)"
            ], key="chart3d")

            if st.button("Generate 3D Chart", type="primary", use_container_width=True):
                if chart3d == "3D Scatter":
                    fig = px.scatter_3d(
                        df, x=x3, y=y3, z=z3, color=c3d,
                        color_continuous_scale="YlOrBr", opacity=0.85,
                        title=f"3D Scatter — {x3} × {y3} × {z3}",
                    )
                    fig.update_traces(marker=dict(size=4))

                elif chart3d == "3D Line":
                    fig = px.line_3d(
                        df.sort_values(x3), x=x3, y=y3, z=z3, color=c3d,
                        title=f"3D Line — {x3} × {y3} × {z3}",
                    )

                elif chart3d == "3D Surface":
                    import numpy as np
                    from scipy.interpolate import griddata
                    xi = np.linspace(df[x3].min(), df[x3].max(), 40)
                    yi = np.linspace(df[y3].min(), df[y3].max(), 40)
                    xi, yi = np.meshgrid(xi, yi)
                    zi = griddata((df[x3].values, df[y3].values), df[z3].values,
                                  (xi, yi), method="linear")
                    fig = go.Figure(data=[go.Surface(
                        x=xi, y=yi, z=zi, colorscale="YlOrBr", opacity=0.9,
                        contours=dict(z=dict(show=True, usecolormap=True,
                                             highlightcolor="#c9a84c", project_z=True))
                    )])
                    fig.update_layout(title=f"3D Surface — {x3} × {y3} × {z3}")

                elif chart3d == "3D Bar (Mesh)":
                    fig = go.Figure(data=[go.Mesh3d(
                        x=df[x3], y=df[y3], z=df[z3],
                        color="#c9a84c", opacity=0.6, alphahull=5,
                    )])
                    fig.update_layout(title=f"3D Mesh — {x3} × {y3} × {z3}")

                fig.update_layout(
                    height=600,
                    scene=dict(
                        xaxis=dict(backgroundcolor="rgba(15,15,26,0.8)",
                                   gridcolor="rgba(201,168,76,0.1)", title=x3),
                        yaxis=dict(backgroundcolor="rgba(15,15,26,0.8)",
                                   gridcolor="rgba(201,168,76,0.1)", title=y3),
                        zaxis=dict(backgroundcolor="rgba(15,15,26,0.8)",
                                   gridcolor="rgba(201,168,76,0.1)", title=z3),
                        bgcolor="rgba(10,10,15,0.9)",
                    ),
                    **DARK
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 3 numeric columns for 3D charts.")

        st.divider()

        if len(num_cols) >= 2:
            st.subheader("🔮 3D Distribution (Histogram Surface)")
            dh1, dh2 = st.columns(2)
            with dh1: hx = st.selectbox("X column", num_cols, key="hx")
            with dh2: hy = st.selectbox("Y column", num_cols, index=min(1, len(num_cols)-1), key="hy")

            if st.button("Generate Distribution Surface", use_container_width=True):
                import numpy as np
                x_data = df[hx].dropna().values
                y_data = df[hy].dropna().values
                min_len = min(len(x_data), len(y_data))
                x_data, y_data = x_data[:min_len], y_data[:min_len]
                hist, xedges, yedges = np.histogram2d(x_data, y_data, bins=20)
                xpos = (xedges[:-1] + xedges[1:]) / 2
                ypos = (yedges[:-1] + yedges[1:]) / 2
                xpos, ypos = np.meshgrid(xpos, ypos)
                fig = go.Figure(data=[go.Surface(
                    x=xpos, y=ypos, z=hist.T, colorscale="YlOrBr", opacity=0.88,
                    contours=dict(z=dict(show=True, usecolormap=True,
                                         highlightcolor="#c9a84c", project_z=False))
                )])
                fig.update_layout(
                    title=f"3D Distribution Surface — {hx} × {hy}", height=550,
                    scene=dict(
                        xaxis_title=hx, yaxis_title=hy, zaxis_title="Frequency",
                        xaxis=dict(backgroundcolor="rgba(15,15,26,0.8)",
                                   gridcolor="rgba(201,168,76,0.1)"),
                        yaxis=dict(backgroundcolor="rgba(15,15,26,0.8)",
                                   gridcolor="rgba(201,168,76,0.1)"),
                        zaxis=dict(backgroundcolor="rgba(15,15,26,0.8)",
                                   gridcolor="rgba(201,168,76,0.1)"),
                        bgcolor="rgba(10,10,15,0.9)",
                    ),
                    **DARK
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        if len(num_cols) >= 3:
            st.subheader("🌍 3D Correlation Network")
            if st.button("Generate Correlation Network", use_container_width=True):
                import numpy as np
                corr = df[num_cols].corr()
                n = len(num_cols)
                angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                xs, ys = np.cos(angles), np.sin(angles)
                zs = np.random.uniform(-0.3, 0.3, n)
                edge_x, edge_y, edge_z = [], [], []
                for i in range(n):
                    for j in range(i+1, n):
                        if abs(corr.iloc[i, j]) > 0.3:
                            edge_x += [xs[i], xs[j], None]
                            edge_y += [ys[i], ys[j], None]
                            edge_z += [zs[i], zs[j], None]
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z, mode="lines",
                    line=dict(color="rgba(201,168,76,0.3)", width=2), name="Correlations"
                ))
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="markers+text",
                    marker=dict(size=10, color="#c9a84c", line=dict(color="#fff", width=1)),
                    text=num_cols, textposition="top center",
                    textfont=dict(color="#e8e6e0", size=11), name="Columns"
                ))
                fig.update_layout(
                    title="3D Correlation Network", height=550,
                    scene=dict(
                        bgcolor="rgba(10,10,15,0.9)",
                        xaxis=dict(showticklabels=False, showgrid=False,
                                   backgroundcolor="rgba(15,15,26,0.8)"),
                        yaxis=dict(showticklabels=False, showgrid=False,
                                   backgroundcolor="rgba(15,15,26,0.8)"),
                        zaxis=dict(showticklabels=False, showgrid=False,
                                   backgroundcolor="rgba(15,15,26,0.8)"),
                    ),
                    **DARK
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("📊 Standard Charts")
        charts = analyzer.generate_visualizations()
        if charts:
            for title, fig in charts:
                with st.expander(title, expanded=False):
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("🎨 Custom Chart Builder")
        c1, c2, c3, c4 = st.columns(4)
        with c1: x_col   = st.selectbox("X axis", df.columns, key="cx")
        with c2: y_col   = st.selectbox("Y axis", df.columns, key="cy")
        with c3: chart_t = st.selectbox("Type", ["Scatter","Line","Bar","Box","Violin","Histogram","Area"])
        with c4:
            color_col = st.selectbox("Color by", ["None"] + list(df.columns))
            color_col = None if color_col == "None" else color_col
        if st.button("Generate Chart", use_container_width=True):
            st.plotly_chart(
                analyzer.create_custom_chart(x_col, y_col, chart_t, color_col),
                use_container_width=True
            )

    # ── TAB 3 — CLEANING ──────────────────────────────────────────────────────
    with tab3:
        st.subheader("🧹 Data Cleaning Tools")
        cleaner = DataCleaner(st.session_state.clean_df)
        issues  = cleaner.get_issues()

        ci1, ci2, ci3, ci4 = st.columns(4)
        ci1.metric("Missing columns",  len(issues["missing"]))
        ci2.metric("Duplicate rows",   issues["duplicates"])
        ci3.metric("Constant columns", len(issues["constant_cols"]))
        ci4.metric("Outlier columns",  len(issues["outlier_cols"]))

        st.markdown("---")
        col_ops, col_prev = st.columns([1, 2])

        with col_ops:
            st.markdown("#### 🔧 Operations")
            do_dedup      = st.checkbox("Remove duplicates",
                                        value=issues["duplicates"] > 0,
                                        disabled=issues["duplicates"] == 0)
            do_drop_const = st.checkbox("Drop constant columns",
                                        value=len(issues["constant_cols"]) > 0,
                                        disabled=len(issues["constant_cols"]) == 0)
            do_fix_names  = st.checkbox("Standardise column names", value=True)

            st.markdown("**Fill missing numeric:**")
            num_strategy = st.radio("Strategy", ["median","mean","zero","skip"], horizontal=True)
            st.markdown("**Fill missing categorical:**")
            cat_strategy = st.radio("Strategy ", ["mode","unknown","skip"], horizontal=True)

            do_cap = st.checkbox("Cap outliers (Winsorization)",
                                 disabled=len(issues["outlier_cols"]) == 0)

            st.markdown("---")
            run_clean = st.button("▶️ Apply Cleaning", type="primary", use_container_width=True)
            reset_btn = st.button("↩️ Reset to Original", use_container_width=True)

        with col_prev:
            st.markdown("#### 👁️ Issues Found")
            if issues["missing"]:
                st.dataframe(
                    pd.DataFrame({"Column": list(issues["missing"].keys()),
                                  "Missing values": list(issues["missing"].values())}),
                    hide_index=True, use_container_width=True
                )
            else:
                st.success("✅ No missing values!")
            if issues["outlier_cols"]:
                st.warning(f"Outlier columns: {', '.join(issues['outlier_cols'])}")
            if issues["constant_cols"]:
                st.warning(f"Constant columns: {', '.join(issues['constant_cols'])}")

        if run_clean:
            c = DataCleaner(st.session_state.clean_df)
            if do_dedup:                c.drop_duplicates()
            if do_drop_const:           c.drop_constant_columns()
            if do_fix_names:            c.fix_column_names()
            if num_strategy != "skip":  c.fill_missing_numeric(num_strategy)
            if cat_strategy != "skip":  c.fill_missing_categorical(cat_strategy)
            if do_cap:                  c.cap_outliers()

            st.session_state.clean_df = c.get_cleaned_df()
            summary = c.get_change_summary()

            st.success("✅ Cleaning complete!")
            for entry in summary["log"]:
                st.markdown(f"- {entry}")

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Rows",    summary["rows_after"],
                       delta=summary["rows_after"]  - summary["rows_before"])
            rc2.metric("Columns", summary["cols_after"],
                       delta=summary["cols_after"]  - summary["cols_before"])
            rc3.metric("Missing", summary["cells_missing_after"],
                       delta=summary["cells_missing_after"] - summary["cells_missing_before"])

            csv = st.session_state.clean_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Cleaned CSV", csv,
                               file_name=f"cleaned_{st.session_state.filename}",
                               mime="text/csv", use_container_width=True)
            st.rerun()

        if reset_btn:
            st.session_state.clean_df = st.session_state.df.copy()
            st.success("↩️ Reset to original.")
            st.rerun()

    # ── TAB 4 — CHAT ──────────────────────────────────────────────────────────
    with tab4:
        st.subheader("💬 Chat with Your Data")

        if not st.session_state.ai_ready or ai is None:
            st.error("AI not connected — check sidebar configuration.")
        else:
            c_l, c_r = st.columns([1, 6])
            with c_l:
                show_lottie("robot", height=80)

            SYSTEM = (
                "You are an expert data analyst. "
                "Answer questions using ONLY the data context provided. "
                "Always cite specific numbers, column names, and row values. "
                "If the answer is not in the context, say so honestly."
            )

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_input = st.chat_input("Ask anything about your data…")

            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input}
                )

                context = extract_relevant_context(df, user_input)
                recent  = st.session_state.chat_history[-10:]
                history = ""
                for m in recent[:-1]:
                    role = "User" if m["role"] == "user" else "Assistant"
                    history += f"\n{role}: {m['content']}"

                full_prompt = f"""Here is the relevant data extracted from the dataset:

{context}

Conversation so far:{history}

User: {user_input}

Answer using the data above. Be specific with numbers and values."""

                ai.model = chat_model
                with st.chat_message("assistant"):
                    st.markdown(typing_indicator(), unsafe_allow_html=True)
                    placeholder   = st.empty()
                    full_response = ""
                    for chunk in ai.generate_stream(full_prompt, system=SYSTEM):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": full_response}
                )
                ai.model = st.session_state.get("selected_model", ai.model)

            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()

            with st.expander("💡 Suggested questions"):
                for s in [
                    "What are the top 3 insights from this data?",
                    "Which column has the highest variance?",
                    "Which category performs best?",
                    "Are there any trends over time?",
                    "Summarise this data for a business executive.",
                    "What data quality issues should I fix first?",
                ]:
                    if st.button(s, key=f"sug_{s}"):
                        st.session_state.chat_history.append(
                            {"role": "user", "content": s}
                        )
                        st.rerun()

    # ── TAB 5 — FORECASTING ───────────────────────────────────────────────────
    with tab5:
        st.subheader("🔮 Time-Series Forecasting")
        dt_cols  = analyzer.info["dt_cols"]
        num_cols = analyzer.info["numeric_cols"]

        if not dt_cols:
            st.warning("⚠️ No datetime column detected.")
            st.info("Make sure your date column uses a standard format like YYYY-MM-DD.")
        elif not num_cols:
            st.warning("No numeric columns found to forecast.")
        else:
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1: date_col  = st.selectbox("📅 Date column", dt_cols)
            with fc2: value_col = st.selectbox("📈 Value to forecast", num_cols)
            with fc3: method    = st.selectbox("🧮 Method", Forecaster.METHODS)
            with fc4: periods   = st.slider("Periods ahead", 7, 365, 30)

            extra = {}
            if method == "Exponential Smoothing":
                extra["alpha"] = st.slider("Smoothing α", 0.05, 0.95, 0.3, 0.05,
                                           help="Higher = more weight on recent data")
            elif method == "Moving Average":
                extra["window"] = st.slider("Window size", 3, 30, 7)

            if st.button("▶️ Run Forecast", type="primary", use_container_width=True):
                lottie_placeholder = st.empty()
                with lottie_placeholder:
                    show_lottie("chart", height=120)

                with st.spinner("Running forecast…"):
                    try:
                        fc = Forecaster(df, date_col, value_col)
                        forecast_df = fc.forecast(method, periods, **extra)
                        lottie_placeholder.empty()

                        st.plotly_chart(fc.plot(forecast_df, method), use_container_width=True)

                        metrics = fc.in_sample_metrics(method)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("MAE",    metrics["MAE"])
                        m2.metric("RMSE",   metrics["RMSE"])
                        m3.metric("MAPE %", f"{metrics['MAPE %']}%")

                        with st.expander("📋 Forecast Table"):
                            display_df = forecast_df.copy()
                            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
                            display_df.columns = ["Date","Forecast","Lower 95%","Upper 95%"]
                            st.dataframe(display_df.round(2), use_container_width=True,
                                         hide_index=True)
                            csv = display_df.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 Download Forecast", csv,
                                               file_name="forecast.csv", mime="text/csv")

                        if st.session_state.ai_ready and ai:
                            st.markdown("---")
                            st.markdown("#### 🤖 AI Interpretation")
                            prompt = f"""
Forecast method: {method}
Column forecasted: {value_col}
Periods ahead: {periods}
Accuracy — MAE: {metrics['MAE']}, RMSE: {metrics['RMSE']}, MAPE: {metrics['MAPE %']}%
First 5 predicted values: {forecast_df['forecast'].head().round(2).tolist()}

In 3-4 sentences, interpret these results for a non-technical business audience.
Are the metrics good? What should they watch out for?
"""
                            st.markdown(typing_indicator(), unsafe_allow_html=True)
                            with st.spinner("AI interpreting…"):
                                interp = ai.generate(prompt, system=build_system_prompt())
                                st.markdown(
                                    f"<div class='insight-box'>{interp}</div>",
                                    unsafe_allow_html=True
                                )
                    except Exception as e:
                        lottie_placeholder.empty()
                        st.error(f"Forecasting error: {e}")
                        st.info("Ensure the date column has valid dates and the value column has numbers.")

    # ── TAB 6 — OUTLIERS & EXPLORE ────────────────────────────────────────────
    with tab6:
        col_out, col_exp = st.columns(2)

        with col_out:
            st.subheader("⚠️ Outlier Detection")
            if analyzer.info["numeric_cols"]:
                st.dataframe(analyzer.detect_outliers(), use_container_width=True, hide_index=True)
                sel = st.selectbox("Visualise:", analyzer.info["numeric_cols"])
                fig = px.box(df, y=sel, points="outliers",
                             title=f"Outlier plot · {sel}",
                             color_discrete_sequence=["#c9a84c"])
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,15,26,0.6)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns for outlier analysis.")

        with col_exp:
            st.subheader("🔍 Raw Data Explorer")
            search = st.text_input("Filter:", placeholder="Search all columns…")
            view = df
            if search:
                mask = df.astype(str).apply(
                    lambda x: x.str.contains(search, case=False, na=False)
                ).any(axis=1)
                view = df[mask]
                st.info(f"{len(view):,} rows match")
            st.dataframe(view, use_container_width=True, height=380)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("📥 CSV", df.to_csv(index=False).encode(),
                                   file_name=f"export_{st.session_state.filename}",
                                   mime="text/csv", use_container_width=True)
            with dl2:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                    df.to_excel(w, index=False, sheet_name="Data")
                st.download_button("📥 Excel", buf.getvalue(),
                                   file_name=f"export_{st.session_state.filename}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

    # ── TAB 7 — COMPARE FILES ─────────────────────────────────────────────────
    with tab7:
        st.subheader("🆚 Compare Two Datasets")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("#### 📄 File 1 (current)")
            st.success(f"✅ {st.session_state.filename}")
            st.info(f"{df.shape[0]:,} rows · {df.shape[1]} cols")

        with col_f2:
            st.markdown("#### 📄 File 2 (upload to compare)")
            uploaded2 = st.file_uploader(
                "Upload second file",
                type=["csv", "xlsx", "xls"],
                key="compare_upload",
                label_visibility="collapsed"
            )
            if uploaded2:
                df2 = load_data(uploaded2)
                if df2 is not None:
                    st.session_state.df2       = df2
                    st.session_state.filename2 = uploaded2.name
                    st.success(f"✅ {uploaded2.name}")
                    st.info(f"{df2.shape[0]:,} rows · {df2.shape[1]} cols")

        if st.session_state.df2 is None:
            st.info("👆 Upload a second file above to start comparing.")
        else:
            df2       = st.session_state.df2
            analyzer2 = DataAnalyzer(df2)

            st.markdown("---")
            st.subheader("📐 Shape Comparison")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("File 1 Rows", f"{df.shape[0]:,}")
            sc2.metric("File 2 Rows", f"{df2.shape[0]:,}", delta=df2.shape[0] - df.shape[0])
            sc3.metric("File 1 Cols", df.shape[1])
            sc4.metric("File 2 Cols", df2.shape[1], delta=df2.shape[1] - df.shape[1])

            st.markdown("---")
            st.subheader("🗂️ Column Comparison")
            cols1 = set(df.columns)
            cols2 = set(df2.columns)
            common    = cols1 & cols2
            only_in_1 = cols1 - cols2
            only_in_2 = cols2 - cols1

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.markdown("**✅ Common columns**")
                st.write(sorted(common) if common else "None")
            with cc2:
                st.markdown(f"**🔵 Only in {st.session_state.filename}**")
                st.write(sorted(only_in_1) if only_in_1 else "None")
            with cc3:
                st.markdown(f"**🟠 Only in {st.session_state.filename2}**")
                st.write(sorted(only_in_2) if only_in_2 else "None")

            st.markdown("---")

            common_num = [c for c in common
                          if c in analyzer.info["numeric_cols"]
                          and c in analyzer2.info["numeric_cols"]]

            if common_num:
                st.subheader("📊 Stats Comparison (common numeric columns)")
                stats1 = df[common_num].describe().round(2)
                stats2 = df2[common_num].describe().round(2)

                tab_s1, tab_s2, tab_s3 = st.tabs([
                    f"📄 {st.session_state.filename}",
                    f"📄 {st.session_state.filename2}",
                    "📉 Difference"
                ])
                with tab_s1:
                    st.dataframe(stats1, use_container_width=True)
                with tab_s2:
                    st.dataframe(stats2, use_container_width=True)
                with tab_s3:
                    diff = (stats2 - stats1).round(2)
                    st.dataframe(diff.style.applymap(
                        lambda v: "color: green" if v > 0 else ("color: red" if v < 0 else "")
                    ), use_container_width=True)

                st.markdown("---")
                st.subheader("📈 Distribution Comparison")
                sel_col = st.selectbox("Select column to compare", common_num)

                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df[sel_col].dropna(),
                    name=st.session_state.filename, opacity=0.75, marker_color="#c9a84c"))
                fig.add_trace(go.Histogram(x=df2[sel_col].dropna(),
                    name=st.session_state.filename2, opacity=0.75, marker_color="#6366f1"))
                fig.update_layout(
                    barmode="overlay", title=f"Distribution of '{sel_col}'",
                    xaxis_title=sel_col, yaxis_title="Count", height=400,
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,15,26,0.6)",
                    font=dict(family="DM Sans, sans-serif", color="#9090a8"),
                )
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Box(y=df[sel_col].dropna(),
                    name=st.session_state.filename, marker_color="#c9a84c"))
                fig2.add_trace(go.Box(y=df2[sel_col].dropna(),
                    name=st.session_state.filename2, marker_color="#6366f1"))
                fig2.update_layout(
                    title=f"Box Plot · '{sel_col}'", height=400,
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,15,26,0.6)",
                    font=dict(family="DM Sans, sans-serif", color="#9090a8"),
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")
            st.subheader("🧹 Missing Values Comparison")
            miss1 = df.isnull().sum().reset_index()
            miss1.columns = ["Column", f"Missing ({st.session_state.filename})"]
            miss2 = df2.isnull().sum().reset_index()
            miss2.columns = ["Column", f"Missing ({st.session_state.filename2})"]
            miss_merged = miss1.merge(miss2, on="Column", how="outer").fillna(0)
            st.dataframe(miss_merged, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("💬 Chat with Both Datasets")

            if not st.session_state.ai_ready or ai is None:
                st.error("AI not connected — check sidebar.")
            else:
                if "compare_chat_history" not in st.session_state:
                    st.session_state.compare_chat_history = []

                COMPARE_SYSTEM = (
                    "You are an expert data analyst comparing two datasets. "
                    "Always reference specific numbers, column names, and "
                    "which file (File 1 or File 2) you are referring to. "
                    "Be precise and data-driven."
                )

                for msg in st.session_state.compare_chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                compare_input = st.chat_input(
                    "Ask anything about both datasets…",
                    key="compare_chat_input"
                )

                if compare_input:
                    with st.chat_message("user"):
                        st.markdown(compare_input)
                    st.session_state.compare_chat_history.append(
                        {"role": "user", "content": compare_input}
                    )

                    context1 = extract_relevant_context(df,  compare_input)
                    context2 = extract_relevant_context(df2, compare_input)

                    recent  = st.session_state.compare_chat_history[-10:]
                    history = ""
                    for m in recent[:-1]:
                        role = "User" if m["role"] == "user" else "Assistant"
                        history += f"\n{role}: {m['content']}"

                    full_prompt = f"""You are comparing two datasets.

FILE 1 — {st.session_state.filename}:
{context1}

FILE 2 — {st.session_state.filename2}:
{context2}

Conversation so far:{history}

User: {compare_input}

Answer using data from both files. Always mention which file you are referring to."""

                    ai.model = chat_model
                    with st.chat_message("assistant"):
                        st.markdown(typing_indicator(), unsafe_allow_html=True)
                        placeholder   = st.empty()
                        full_response = ""
                        for chunk in ai.generate_stream(full_prompt, system=COMPARE_SYSTEM):
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)

                    st.session_state.compare_chat_history.append(
                        {"role": "assistant", "content": full_response}
                    )
                    ai.model = st.session_state.get("selected_model", ai.model)

                if st.session_state.compare_chat_history:
                    if st.button("🗑️ Clear Comparison Chat", key="clear_compare"):
                        st.session_state.compare_chat_history = []
                        st.rerun()

                with st.expander("💡 Suggested comparison questions"):
                    for s in [
                        "Which file has better data quality?",
                        "What are the biggest differences between the two files?",
                        "Which file has higher average values?",
                        "Are there any columns that behave differently across files?",
                        "Which file would you recommend for analysis and why?",
                        "Summarise both files in one paragraph each.",
                    ]:
                        if st.button(s, key=f"cmp_{s}"):
                            st.session_state.compare_chat_history.append(
                                {"role": "user", "content": s}
                            )
                            st.rerun()

            st.markdown("---")
            st.subheader("🤖 AI Comparison Summary")
            if st.session_state.ai_ready and ai:
                if st.button("▶️ Generate AI Comparison Report",
                             type="primary", use_container_width=True):
                    prompt = f"""
You are comparing two datasets.

FILE 1: {st.session_state.filename}
{analyzer.get_data_summary()}

FILE 2: {st.session_state.filename2}
{analyzer2.get_data_summary()}

Common columns: {sorted(common)}
Only in File 1: {sorted(only_in_1)}
Only in File 2: {sorted(only_in_2)}

Write a clear comparison report covering:
1. Key differences in size and structure
2. Statistical differences in common numeric columns
3. Data quality differences
4. Which file appears to be higher quality and why
5. Any notable patterns unique to each file
"""
                    st.markdown(typing_indicator(), unsafe_allow_html=True)
                    with st.spinner("AI generating comparison report…"):
                        full = ""
                        result_box = st.empty()
                        for chunk in ai.generate_stream(prompt, system=build_system_prompt()):
                            full += chunk
                            result_box.markdown(
                                f"<div class='insight-box'>{full}▌</div>",
                                unsafe_allow_html=True
                            )
                        result_box.markdown(
                            f"<div class='insight-box'>{full}</div>",
                            unsafe_allow_html=True
                        )
            else:
                st.info("Connect AI in the sidebar to generate a comparison report.")
