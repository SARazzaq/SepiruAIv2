"""
Advanced data analysis utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PALETTE = px.colors.qualitative.Bold


class DataAnalyzer:
    """Analyse a DataFrame and produce stats, quality reports and charts."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.info = self._build_info()

    # ── Meta ─────────────────────────────────────────────────────────────────

    def _build_info(self) -> Dict[str, Any]:
        num_cols  = list(self.df.select_dtypes(include=["int64", "float64"]).columns)
        cat_cols  = list(self.df.select_dtypes(include=["object", "category"]).columns)
        dt_cols   = list(self.df.select_dtypes(include=["datetime64"]).columns)

        # Try to parse string columns as datetime
        for c in cat_cols[:]:
            try:
                self.df[c] = pd.to_datetime(self.df[c])
                dt_cols.append(c)
                cat_cols.remove(c)
            except Exception:
                pass

        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing": self.df.isnull().sum().to_dict(),
            "missing_pct": (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict(),
            "numeric_cols": num_cols,
            "cat_cols": cat_cols,
            "dt_cols": dt_cols,
            "duplicate_rows": int(self.df.duplicated().sum()),
        }

    # ── Text summaries ────────────────────────────────────────────────────────

    def get_data_summary(self) -> str:
        rows, cols = self.info["shape"]
        missing_total = sum(self.info["missing"].values())
        completeness = 100 - (missing_total / (rows * cols) * 100) if rows * cols else 100

        lines = [
            "=== DATASET OVERVIEW ===",
            f"Rows        : {rows:,}",
            f"Columns     : {cols}",
            f"Memory      : {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            f"Completeness: {completeness:.1f}%",
            f"Duplicates  : {self.info['duplicate_rows']:,}",
            "",
            "=== COLUMN BREAKDOWN ===",
            f"Numeric     : {len(self.info['numeric_cols'])} → {self.info['numeric_cols']}",
            f"Categorical  : {len(self.info['cat_cols'])} → {self.info['cat_cols']}",
            f"Datetime    : {len(self.info['dt_cols'])} → {self.info['dt_cols']}",
            "",
            "=== NUMERIC STATS ===",
        ]

        if self.info["numeric_cols"]:
            desc = self.df[self.info["numeric_cols"]].describe().round(2)
            lines.append(desc.to_string())
        else:
            lines.append("No numeric columns.")

        if self.info["cat_cols"]:
            lines += ["", "=== CATEGORICAL COUNTS ==="]
            for c in self.info["cat_cols"][:5]:
                top = self.df[c].value_counts().head(5).to_dict()
                lines.append(f"{c}: {top}")

        return "\n".join(lines)

    def get_quality_report(self) -> pd.DataFrame:
        rows = []
        for col in self.df.columns:
            miss_pct = self.info["missing_pct"][col]
            unique_n = self.df[col].nunique()
            unique_pct = round(unique_n / len(self.df) * 100, 1) if len(self.df) else 0
            sample = str(self.df[col].dropna().head(3).tolist())[:60]
            status = "✅" if miss_pct == 0 else ("⚠️" if miss_pct < 20 else "❌")
            rows.append({
                "Column": col,
                "Type": str(self.df[col].dtype),
                "Missing %": f"{miss_pct:.1f}%",
                "Unique": f"{unique_n} ({unique_pct}%)",
                "Status": status,
                "Sample Values": sample,
            })
        return pd.DataFrame(rows)

    # ── Visualisations ────────────────────────────────────────────────────────

    def generate_visualizations(self) -> List[Tuple[str, go.Figure]]:
        figs = []
        num_cols = self.info["numeric_cols"][:6]
        cat_cols = self.info["cat_cols"]
        dt_cols  = self.info["dt_cols"]

        # 1. Distributions
        if num_cols:
            n = len(num_cols)
            cols_per_row = min(3, n)
            rows_needed  = (n + cols_per_row - 1) // cols_per_row
            fig = make_subplots(rows=rows_needed, cols=cols_per_row,
                                subplot_titles=num_cols)
            for idx, col in enumerate(num_cols):
                r = idx // cols_per_row + 1
                c = idx %  cols_per_row + 1
                fig.add_trace(
                    go.Histogram(x=self.df[col].dropna(), name=col,
                                 marker_color=PALETTE[idx % len(PALETTE)]),
                    row=r, col=c)
            fig.update_layout(height=280 * rows_needed, showlegend=False,
                              title_text="Numeric Distributions")
            figs.append(("📊 Distributions", fig))

        # 2. Correlation heatmap
        if len(num_cols) >= 2:
            corr = self.df[num_cols].corr().round(2)
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Correlation Matrix")
            figs.append(("🔗 Correlations", fig))

        # 3. Top categorical values
        for cat_col in cat_cols[:2]:
            vc = self.df[cat_col].value_counts().head(10).reset_index()
            vc.columns = [cat_col, "count"]
            fig = px.bar(vc, x="count", y=cat_col, orientation="h",
                         color="count", color_continuous_scale="Blues",
                         title=f"Top values · {cat_col}")
            fig.update_layout(yaxis=dict(autorange="reversed"),
                              coloraxis_showscale=False)
            figs.append((f"🏷️ {cat_col}", fig))

        # 4. Time-series if datetime present
        if dt_cols and num_cols:
            dt_col  = dt_cols[0]
            val_col = num_cols[0]
            ts = (self.df[[dt_col, val_col]]
                  .dropna()
                  .sort_values(dt_col)
                  .set_index(dt_col)
                  .resample("D")[val_col].sum()
                  .reset_index())
            if len(ts) > 1:
                fig = px.line(ts, x=dt_col, y=val_col,
                              title=f"{val_col} over Time",
                              color_discrete_sequence=["#636EFA"])
                fig.update_traces(line_width=2)
                figs.append(("📅 Time Series", fig))

        # 5. Box plots for numeric × first category
        if num_cols and cat_cols:
            n_col  = num_cols[0]
            c_col  = cat_cols[0]
            top_cats = self.df[c_col].value_counts().head(8).index
            sub = self.df[self.df[c_col].isin(top_cats)]
            fig = px.box(sub, x=c_col, y=n_col, color=c_col,
                         title=f"{n_col} by {c_col}",
                         color_discrete_sequence=PALETTE)
            fig.update_layout(showlegend=False)
            figs.append((f"📦 Box · {n_col} × {c_col}", fig))

        return figs

    def create_custom_chart(self, x_col: str, y_col: str,
                            chart_type: str,
                            color_col: Optional[str] = None) -> go.Figure:
        kwargs = dict(data_frame=self.df, x=x_col, y=y_col,
                      color=color_col if color_col else None,
                      color_discrete_sequence=PALETTE)
        dispatch = {
            "Scatter":   px.scatter,
            "Line":      px.line,
            "Bar":       px.bar,
            "Box":       px.box,
            "Violin":    px.violin,
            "Histogram": px.histogram,
            "Area":      px.area,
        }
        fn = dispatch.get(chart_type, px.scatter)
        if chart_type == "Histogram":
            kwargs.pop("y", None)
        return fn(**{k: v for k, v in kwargs.items() if v is not None
                     or k not in ("color",)})

    # ── Outlier detection ─────────────────────────────────────────────────────

    def detect_outliers(self) -> pd.DataFrame:
        rows = []
        for col in self.info["numeric_cols"]:
            s = self.df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            rows.append({
                "Column": col,
                "Q1": round(float(q1), 2),
                "Median": round(float(s.median()), 2),
                "Q3": round(float(q3), 2),
                "IQR": round(float(iqr), 2),
                "Outliers": n_out,
                "Outlier %": f"{n_out/len(s)*100:.1f}%",
            })
        return pd.DataFrame(rows)
