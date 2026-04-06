"""
Utility helpers for CSV AI Analyst.
"""

import pandas as pd
import streamlit as st
from typing import Optional


def load_data(file) -> Optional[pd.DataFrame]:
    """Load CSV or Excel file into a DataFrame."""
    try:
        name = file.name.lower()
        if name.endswith(".csv"):
            # Try common encodings
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    file.seek(0)
                    return pd.read_csv(file, encoding=enc)
                except UnicodeDecodeError:
                    continue
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file)
        else:
            st.error("Unsupported format — please upload .csv or .xlsx")
            return None
    except Exception as e:
        st.error(f"Could not load file: {e}")
        return None


def build_system_prompt() -> str:
    return (
        "You are an expert data scientist and business analyst. "
        "You analyse structured datasets and provide clear, actionable insights. "
        "Always back up observations with specific numbers from the data. "
        "Format your response with markdown headings and bullet points for readability."
    )


def build_analysis_prompt(data_summary: str, question: str) -> str:
    return f"""Here is a summary of the dataset I am working with:

{data_summary}

---

Please answer the following question thoroughly:

**{question}**

Structure your answer with:
1. Direct answer / key finding
2. Supporting evidence (use numbers from the summary)
3. Patterns or anomalies worth noting
4. Recommended next steps or actions
"""


def format_number(n: float) -> str:
    if n >= 1e9:  return f"{n/1e9:.2f}B"
    if n >= 1e6:  return f"{n/1e6:.2f}M"
    if n >= 1e3:  return f"{n/1e3:.1f}K"
    return str(round(n, 2))


def build_chat_system_prompt() -> str:
    return (
        "You are an expert data analyst assistant. "
        "You are given a dataset summary and must answer the user's questions "
        "with precise, data-driven insights. "
        "Always reference specific column names, numbers, and statistics from the data. "
        "Be concise, clear, and actionable."
    )