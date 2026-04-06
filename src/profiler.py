"""
Deep data profiling — richer than basic describe().
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,15,26,0.6)",
    font=dict(family="DM Sans, sans-serif", color="#9090a8", size=11),
    title_font=dict(family="DM Serif Display, serif", color="#e8e6e0", size=15),
)


def full_profile(df: pd.DataFrame) -> dict:
    """Generate complete profile of the dataset."""
    profile = {}

    # ── Overview ──────────────────────────────────────────────────
    profile["overview"] = {
        "rows": len(df),
        "cols": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / df.size * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "numeric_cols": len(df.select_dtypes(include="number").columns),
        "cat_cols": len(df.select_dtypes(include="object").columns),
        "datetime_cols": len(df.select_dtypes(include="datetime64").columns),
    }

    # ── Per-column stats ──────────────────────────────────────────
    col_stats = []
    for col in df.columns:
        s = df[col]
        stat = {
            "Column": col,
            "Type": str(s.dtype),
            "Missing": int(s.isnull().sum()),
            "Missing %": round(s.isnull().mean() * 100, 1),
            "Unique": int(s.nunique()),
            "Unique %": round(s.nunique() / len(df) * 100, 1),
        }
        if pd.api.types.is_numeric_dtype(s):
            stat.update({
                "Mean":     round(float(s.mean()), 3),
                "Std":      round(float(s.std()), 3),
                "Min":      round(float(s.min()), 3),
                "Max":      round(float(s.max()), 3),
                "Skewness": round(float(s.skew()), 3),
                "Kurtosis": round(float(s.kurt()), 3),
                "Zeros":    int((s == 0).sum()),
                "Negative": int((s < 0).sum()),
            })
        else:
            top = s.value_counts().index[0] if s.nunique() > 0 else "N/A"
            stat.update({
                "Top Value": str(top),
                "Top Freq":  int(s.value_counts().iloc[0]) if s.nunique() > 0 else 0,
            })
        col_stats.append(stat)

    profile["col_stats"] = pd.DataFrame(col_stats)

    # ── Missing heatmap ───────────────────────────────────────────
    miss = df.isnull().astype(int)
    if miss.sum().sum() > 0:
        fig = px.imshow(
            miss.T,
            color_continuous_scale=[[0, "#0f0f1a"], [1, "#c9a84c"]],
            title="Missing Values Heatmap",
            aspect="auto",
        )
        fig.update_layout(**DARK, coloraxis_showscale=False)
        profile["missing_fig"] = fig
    else:
        profile["missing_fig"] = None

    # ── Correlation ───────────────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().round(2)
        fig = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale="YlOrBr",
            title="Correlation Heatmap"
        )
        fig.update_layout(**DARK)
        profile["corr_fig"] = fig

        # Strong correlations table
        strong = []
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                val = corr.iloc[i, j]
                if abs(val) > 0.5:
                    strong.append({
                        "Column A": num_cols[i],
                        "Column B": num_cols[j],
                        "Correlation": val,
                        "Strength": "Strong" if abs(val) > 0.7 else "Moderate"
                    })
        profile["strong_corr"] = pd.DataFrame(strong) if strong else None
    else:
        profile["corr_fig"] = None
        profile["strong_corr"] = None

    # ── Distribution grid ─────────────────────────────────────────
    if len(num_cols) > 0:
        n = min(6, len(num_cols))
        cols_per_row = min(3, n)
        rows = (n + cols_per_row - 1) // cols_per_row
        fig = make_subplots(rows=rows, cols=cols_per_row,
                            subplot_titles=list(num_cols[:n]))
        for idx, col in enumerate(num_cols[:n]):
            r = idx // cols_per_row + 1
            c = idx %  cols_per_row + 1
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(),
                    marker_color="#c9a84c",
                    opacity=0.8, name=col
                ),
                row=r, col=c
            )
        fig.update_layout(
            height=280 * rows, showlegend=False,
            title_text="Distribution Grid",
            **DARK
        )
        profile["dist_fig"] = fig
    else:
        profile["dist_fig"] = None

    # ── Skewness chart ────────────────────────────────────────────
    if len(num_cols) > 0:
        skew_df = pd.DataFrame({
            "Column": num_cols,
            "Skewness": [df[c].skew() for c in num_cols]
        }).sort_values("Skewness")
        fig = px.bar(
            skew_df, x="Skewness", y="Column",
            orientation="h",
            color="Skewness",
            color_continuous_scale="RdYlGn",
            title="Skewness by Column"
        )
        fig.add_vline(x=0, line_dash="dash",
                      line_color="rgba(201,168,76,0.5)")
        fig.update_layout(**DARK, coloraxis_showscale=False)
        profile["skew_fig"] = fig
    else:
        profile["skew_fig"] = None

    return profile