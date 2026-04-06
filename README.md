<div align="center">

<br/>

# ✦ SEPIRU AI

### *Your data has a story. We make it speak.*

<br/>

[![Live Demo](https://img.shields.io/badge/▶_Live_Demo-Streamlit_Cloud-c9a84c?style=for-the-badge&logoColor=white)](https://share.streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-6366f1?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-10b981?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Powered by Groq](https://img.shields.io/badge/Powered_by-Groq_AI-f59e0b?style=for-the-badge)](https://groq.com)

<br/>

> **Drop a CSV. Ask anything. Get intelligence.**
> No dashboards to configure. No SQL to write. No data science degree required.
> Just upload — and watch your data come alive.

<br/>

---

</div>

## ⚡ What is Sepiru AI?

Sepiru AI is a **zero-friction data intelligence platform** built for people who want answers, not workflows.

Upload any CSV or Excel file and instantly get — AI-powered chat with your data, automated visualisations, ML model training, time-series forecasting, deep statistical profiling, and side-by-side dataset comparison. All wrapped in a **cinematic dark UI** with 3D animations that feels nothing like any data tool you've used before.

It runs on **Groq's free API** — meaning the AI is blazing fast, and it costs you nothing.

<br/>

---

## ✦ Features that hit different

| | Feature | What it does |
|---|---|---|
| 💬 | **Chat with your data** | Ask questions in plain English. Get precise, data-backed answers. Multi-turn memory. |
| 📈 | **Auto Visualisations** | 3D scatter, surface, correlation networks, custom chart builder — generated instantly. |
| 🧹 | **Smart Data Cleaning** | Detects missing values, duplicates, outliers, constant columns. One-click fix. |
| 🔮 | **Forecasting** | Predict future values with confidence bands. 6 methods. AI interprets the results. |
| 🤖 | **ML Training** | Auto-detects classification vs regression. Trains 10+ models. Leaderboard comparison. |
| 📊 | **Deep Profiling** | Skewness, kurtosis, correlation heatmaps, distribution grids — the full statistical picture. |
| 🆚 | **Compare Datasets** | Upload two files. Get a side-by-side breakdown with AI commentary. |
| 🖼️ | **Vision AI** | Analyze image folders with local Ollama vision models. |

<br/>

---

## 🎨 UI that makes people stop and stare

Sepiru AI isn't just functional — it's **visually unlike anything else** in the data space.

- **6-layer immersive 3D background** — rotating wireframe geometry, volumetric light beams, aurora depth, neural particle mesh, floating data orbs, and a perspective grid floor
- **Spring-physics cursor** with 4 states (hover / input / drag / click)
- **Cinematic page transitions** — curtain wipe on every load
- **3D card tilt** with dynamic lighting that follows your mouse
- **Magnetic buttons** that pull toward your cursor
- **Animated number counters** on every metric
- **Scroll-reveal animations** with staggered entrance choreography
- **Playfair Display + DM Sans** typography — editorial, elegant, not another tech font

<br/>

---

## 🚀 Deploy in 5 minutes (free)

**1. Fork this repo**

**2. Get a free Groq API key**
→ [console.groq.com](https://console.groq.com) · Sign up · API Keys · Create

**3. Deploy on Streamlit Cloud**
→ [share.streamlit.io](https://share.streamlit.io) · New app · select this repo · `app.py`

**4. Add your secret**
→ App Settings → Secrets → paste:

```toml
AI_PROVIDER = "groq"
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL   = "llama-3.1-70b-versatile"
```

**5. Done.** Your live app is ready to share.

<br/>

---

## 🖥️ Run locally

```bash
git clone https://github.com/yourusername/sepiru-ai
cd sepiru-ai

pip install -r requirements.txt

# Add your key to .env
echo 'AI_PROVIDER=groq' >> .env
echo 'GROQ_API_KEY=gsk_your_key_here' >> .env

streamlit run app.py
```

<br/>

---

## 🧠 AI Providers supported

| Provider | Cost | Speed | Best for |
|---|---|---|---|
| **Groq** *(recommended)* | Free tier | ⚡ Fastest | Cloud deployment |
| **Ollama** | Free, local | Fast | Privacy, offline |
| **OpenAI** | Paid | Fast | GPT-4o quality |
| **Anthropic** | Paid | Fast | Claude quality |

<br/>

---

## 🛠️ Tech stack

```
Streamlit      — UI framework
Groq / Ollama  — LLM inference
Plotly         — Interactive 3D charts
scikit-learn   — ML training
XGBoost / LightGBM / CatBoost — Boosting models
Pandas / NumPy — Data processing
Canvas API     — 3D background animations (pure JS, no Three.js)
```

<br/>

---

## 📁 Project structure

```
sepiru-ai/
├── app.py                  ← Main application
├── pages/
│   ├── 1_🤖_ML_Training.py
│   ├── 2_📊_Data_Profiling.py
│   └── 3_🖼️_Vision_AI.py
├── src/
│   ├── ai_client.py        ← Multi-provider AI
│   ├── animations.py       ← 3D background engine
│   ├── ui_components.py    ← Premium component library
│   ├── data_analyzer.py
│   ├── data_cleaner.py
│   ├── forecaster.py
│   ├── ml_trainer.py
│   └── ...
├── assets/
│   ├── style.css           ← Design tokens + layout
│   ├── components.css      ← Component library
│   └── animations.css      ← Keyframes + choreography
└── sample_data/
    └── sample_sales.csv    ← Try it instantly
```

<br/>

---

## 🌟 Why Sepiru AI?

Most data tools make you feel like you're filling out a form.

Sepiru AI makes you feel like you're in a **mission control room**.

The same data. A completely different experience.

<br/>

---

<div align="center">

**Built with obsession. Deployed with one click.**

*If this made you stop scrolling — give it a ⭐*

<br/>

[![Star on GitHub](https://img.shields.io/github/stars/yourusername/sepiru-ai?style=social)](https://github.com/yourusername/sepiru-ai)

</div>
