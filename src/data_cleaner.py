"""
Data cleaning utilities — fix missing values, duplicates, types, outliers.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class DataCleaner:
    """Apply cleaning operations to a DataFrame and track changes."""

    def __init__(self, df: pd.DataFrame):
        self.original = df.copy()
        self.df = df.copy()
        self.log: list[str] = []

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_issues(self) -> Dict[str, Any]:
        issues = {}
        # Missing values
        miss = self.df.isnull().sum()
        issues["missing"] = miss[miss > 0].to_dict()
        # Duplicates
        issues["duplicates"] = int(self.df.duplicated().sum())
        # Constant columns
        issues["constant_cols"] = [c for c in self.df.columns if self.df[c].nunique() <= 1]
        # Numeric cols with outliers (IQR)
        outlier_cols = []
        for col in self.df.select_dtypes(include="number").columns:
            s = self.df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum() > 0:
                outlier_cols.append(col)
        issues["outlier_cols"] = outlier_cols
        return issues

    # ── Operations ────────────────────────────────────────────────────────────

    def drop_duplicates(self) -> int:
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = before - len(self.df)
        if removed:
            self.log.append(f"✅ Removed {removed} duplicate rows")
        return removed

    def fill_missing_numeric(self, strategy: str = "median") -> Dict[str, int]:
        filled = {}
        for col in self.df.select_dtypes(include="number").columns:
            n = self.df[col].isnull().sum()
            if n == 0:
                continue
            if strategy == "mean":
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == "median":
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == "zero":
                self.df[col] = self.df[col].fillna(0)
            filled[col] = int(n)
        if filled:
            self.log.append(
                f"✅ Filled missing numeric values ({strategy}) in: {', '.join(filled.keys())}"
            )
        return filled

    def fill_missing_categorical(self, strategy: str = "mode") -> Dict[str, int]:
        filled = {}
        for col in self.df.select_dtypes(include=["object", "category"]).columns:
            n = self.df[col].isnull().sum()
            if n == 0:
                continue
            if strategy == "mode":
                mode_val = self.df[col].mode()
                if len(mode_val):
                    self.df[col] = self.df[col].fillna(mode_val[0])
            elif strategy == "unknown":
                self.df[col] = self.df[col].fillna("Unknown")
            filled[col] = int(n)
        if filled:
            self.log.append(
                f"✅ Filled missing categorical values ({strategy}) in: {', '.join(filled.keys())}"
            )
        return filled

    def drop_missing_rows(self, threshold: float = 0.5) -> int:
        """Drop rows where more than `threshold` fraction of values are missing."""
        before = len(self.df)
        self.df = self.df[self.df.isnull().mean(axis=1) < threshold].reset_index(drop=True)
        removed = before - len(self.df)
        if removed:
            self.log.append(f"✅ Dropped {removed} rows with >{int(threshold*100)}% missing values")
        return removed

    def drop_constant_columns(self) -> list[str]:
        cols = [c for c in self.df.columns if self.df[c].nunique() <= 1]
        if cols:
            self.df = self.df.drop(columns=cols)
            self.log.append(f"✅ Dropped constant columns: {', '.join(cols)}")
        return cols

    def fix_column_names(self) -> None:
        """Lowercase, strip, replace spaces with underscores."""
        renamed = {c: c.strip().lower().replace(" ", "_") for c in self.df.columns}
        self.df = self.df.rename(columns=renamed)
        self.log.append("✅ Standardised column names (lowercase + underscores)")

    def cap_outliers(self, method: str = "iqr") -> Dict[str, int]:
        """Cap outliers at IQR fences (Winsorization)."""
        capped = {}
        for col in self.df.select_dtypes(include="number").columns:
            s = self.df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = int(((self.df[col] < lo) | (self.df[col] > hi)).sum())
            if n:
                self.df[col] = self.df[col].clip(lower=lo, upper=hi)
                capped[col] = n
        if capped:
            self.log.append(
                f"✅ Capped outliers in: {', '.join(capped.keys())}"
            )
        return capped

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_cleaned_df(self) -> pd.DataFrame:
        return self.df

    def get_change_summary(self) -> Dict[str, Any]:
        return {
            "rows_before": len(self.original),
            "rows_after":  len(self.df),
            "cols_before": len(self.original.columns),
            "cols_after":  len(self.df.columns),
            "cells_missing_before": int(self.original.isnull().sum().sum()),
            "cells_missing_after":  int(self.df.isnull().sum().sum()),
            "log": self.log,
        }
