"""
Smart context extractor — pulls relevant data from DataFrame
based on the user's question using pandas, no embeddings needed.
"""

import pandas as pd
import numpy as np
from typing import Optional


def extract_relevant_context(df: pd.DataFrame, question: str) -> str:
    """
    Analyse the question and extract the most relevant
    data from the DataFrame to answer it accurately.
    """
    question_lower = question.lower()
    context_parts  = []

    # ── 1. Always include schema ──────────────────────────────────────────
    schema = "DATASET SCHEMA:\n"
    for col in df.columns:
        dtype   = str(df[col].dtype)
        n_miss  = df[col].isnull().sum()
        n_uniq  = df[col].nunique()
        schema += f"- {col} ({dtype}) | missing: {n_miss} | unique: {n_uniq}\n"
    context_parts.append(schema)

    # ── 2. Detect which columns are relevant to the question ──────────────
    relevant_cols = []
    for col in df.columns:
        if col.lower() in question_lower:
            relevant_cols.append(col)

    # If no columns matched, use all
    if not relevant_cols:
        relevant_cols = list(df.columns)

    num_cols = df[relevant_cols].select_dtypes(include="number").columns.tolist()
    cat_cols = df[relevant_cols].select_dtypes(include=["object","category"]).columns.tolist()

    # ── 3. Stats for numeric columns ──────────────────────────────────────
    if num_cols:
        stats = df[num_cols].describe().round(2)
        context_parts.append(f"NUMERIC STATS for {num_cols}:\n{stats.to_string()}")

    # ── 4. Value counts for categorical columns ───────────────────────────
    if cat_cols:
        for col in cat_cols[:3]:
            vc = df[col].value_counts().head(10)
            context_parts.append(f"TOP VALUES in '{col}':\n{vc.to_string()}")

    # ── 5. Question-specific extractions ─────────────────────────────────

    # Min / least / lowest / cheapest / smallest
    if any(w in question_lower for w in ["min","least","lowest","cheapest","smallest","minimum"]):
        for col in num_cols[:3]:
            idx     = df[col].idxmin()
            min_val = df[col].min()
            row     = df.loc[idx].to_dict()
            context_parts.append(
                f"MINIMUM of '{col}': {min_val}\nFull row: {row}"
            )

    # Max / most / highest / expensive / largest / best
    if any(w in question_lower for w in ["max","most","highest","expensive","largest","best","top","maximum"]):
        for col in num_cols[:3]:
            idx     = df[col].idxmax()
            max_val = df[col].max()
            row     = df.loc[idx].to_dict()
            context_parts.append(
                f"MAXIMUM of '{col}': {max_val}\nFull row: {row}"
            )

    # Average / mean
    if any(w in question_lower for w in ["average","mean","avg"]):
        for col in num_cols[:3]:
            context_parts.append(
                f"MEAN of '{col}': {df[col].mean():.2f}"
            )

    # Count / how many
    if any(w in question_lower for w in ["count","how many","number of","total"]):
        context_parts.append(f"TOTAL ROWS: {len(df)}")
        for col in cat_cols[:2]:
            vc = df[col].value_counts()
            context_parts.append(f"COUNT by '{col}':\n{vc.to_string()}")

    # Trend / over time / by date
    if any(w in question_lower for w in ["trend","over time","by date","monthly","daily","yearly"]):
        dt_cols = df.select_dtypes(include="datetime64").columns.tolist()
        if dt_cols and num_cols:
            dt_col  = dt_cols[0]
            val_col = num_cols[0]
            trend   = (df.groupby(dt_col)[val_col]
                       .sum()
                       .reset_index()
                       .tail(10))
            context_parts.append(
                f"TREND of '{val_col}' over '{dt_col}' (last 10):\n{trend.to_string(index=False)}"
            )

    # Correlation
    if any(w in question_lower for w in ["correlation","correlate","related","relationship"]):
        if len(num_cols) >= 2:
            corr = df[num_cols].corr().round(2)
            context_parts.append(f"CORRELATION MATRIX:\n{corr.to_string()}")

    # Group by / by category / per
    if any(w in question_lower for w in ["by","per","group","each","category","region","type"]):
        for cat in cat_cols[:2]:
            for num in num_cols[:2]:
                grp = df.groupby(cat)[num].agg(["mean","sum","count"]).round(2)
                context_parts.append(
                    f"'{num}' grouped by '{cat}':\n{grp.to_string()}"
                )

    # Distribution / spread / outlier
    if any(w in question_lower for w in ["distribution","spread","outlier","skew"]):
        for col in num_cols[:3]:
            q1  = df[col].quantile(0.25)
            q3  = df[col].quantile(0.75)
            iqr = q3 - q1
            n_out = int(((df[col] < q1-1.5*iqr) | (df[col] > q3+1.5*iqr)).sum())
            context_parts.append(
                f"DISTRIBUTION of '{col}': "
                f"min={df[col].min():.2f}, Q1={q1:.2f}, "
                f"median={df[col].median():.2f}, Q3={q3:.2f}, "
                f"max={df[col].max():.2f}, outliers={n_out}"
            )

    # Sample rows — always include a few
    context_parts.append(
        f"SAMPLE ROWS (first 5):\n{df.head(5).to_string(index=False)}"
    )

    # If question mentions a specific value, filter rows
    for word in question_lower.split():
        if len(word) > 3:
            for col in cat_cols:
                matches = df[df[col].astype(str).str.lower().str.contains(word, na=False)]
                if 0 < len(matches) <= 20:
                    context_parts.append(
                        f"ROWS WHERE '{col}' contains '{word}':\n"
                        f"{matches.to_string(index=False)}"
                    )

    return "\n\n".join(context_parts)