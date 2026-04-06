"""
Time-series forecasting using simple but effective methods.
No heavy dependencies — uses only pandas/numpy + optional scipy.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


# ── Helpers ───────────────────────────────────────────────────────────────────

def _exponential_smoothing(series: np.ndarray, alpha: float) -> np.ndarray:
    result = np.zeros_like(series, dtype=float)
    result[0] = series[0]
    for t in range(1, len(series)):
        result[t] = alpha * series[t] + (1 - alpha) * result[t - 1]
    return result


def _linear_trend(x: np.ndarray, y: np.ndarray):
    """Returns (slope, intercept) via least squares."""
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


# ── Main forecaster ───────────────────────────────────────────────────────────

class Forecaster:
    """
    Fits a simple additive model:  trend + seasonality + smoothing
    Works on any numeric column with a datetime index.
    """

    METHODS = ["Linear Trend", "Exponential Smoothing", "Moving Average"]

    def __init__(self, df: pd.DataFrame, date_col: str, value_col: str):
        self.raw = df[[date_col, value_col]].dropna().copy()
        self.raw[date_col] = pd.to_datetime(self.raw[date_col])
        self.raw = self.raw.sort_values(date_col).reset_index(drop=True)
        self.date_col  = date_col
        self.value_col = value_col

        # Infer frequency
        self.dates  = self.raw[date_col]
        self.values = self.raw[value_col].astype(float).values
        self.n      = len(self.values)

    # ── Forecast methods ──────────────────────────────────────────────────────

    def forecast_linear(self, periods: int) -> pd.DataFrame:
        x = np.arange(self.n)
        slope, intercept = _linear_trend(x, self.values)
        future_x  = np.arange(self.n, self.n + periods)
        pred_hist = slope * x + intercept
        pred_fut  = slope * future_x + intercept

        # Residual std for confidence interval
        residuals = self.values - pred_hist
        std = residuals.std()

        future_dates = self._future_dates(periods)
        return pd.DataFrame({
            "date":  future_dates,
            "forecast": pred_fut,
            "lower":    pred_fut - 1.96 * std,
            "upper":    pred_fut + 1.96 * std,
        })

    def forecast_exp_smoothing(self, periods: int, alpha: float = 0.3) -> pd.DataFrame:
        smoothed   = _exponential_smoothing(self.values, alpha)
        last_val   = smoothed[-1]
        # Simple flat forecast with widening CI
        std        = np.std(self.values - smoothed)
        future_dates = self._future_dates(periods)
        t = np.arange(1, periods + 1)
        return pd.DataFrame({
            "date":     future_dates,
            "forecast": np.full(periods, last_val),
            "lower":    last_val - 1.96 * std * np.sqrt(t),
            "upper":    last_val + 1.96 * std * np.sqrt(t),
        })

    def forecast_moving_average(self, periods: int, window: int = 7) -> pd.DataFrame:
        w = min(window, self.n)
        ma = np.convolve(self.values, np.ones(w) / w, mode="valid")
        last_ma = ma[-1]
        std = np.std(self.values[-w:])
        future_dates = self._future_dates(periods)
        t = np.arange(1, periods + 1)
        return pd.DataFrame({
            "date":     future_dates,
            "forecast": np.full(periods, last_ma),
            "lower":    last_ma - 1.96 * std * np.sqrt(t / w),
            "upper":    last_ma + 1.96 * std * np.sqrt(t / w),
        })

    # ── Unified entry point ───────────────────────────────────────────────────

    def forecast(self, method: str, periods: int, **kwargs) -> pd.DataFrame:
        if method == "Linear Trend":
            return self.forecast_linear(periods)
        elif method == "Exponential Smoothing":
            return self.forecast_exp_smoothing(periods, **kwargs)
        elif method == "Moving Average":
            return self.forecast_moving_average(periods, **kwargs)
        else:
            return self.forecast_linear(periods)

    # ── Plot ──────────────────────────────────────────────────────────────────

    def plot(self, forecast_df: pd.DataFrame, method: str) -> go.Figure:
        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=self.dates, y=self.values,
            mode="lines", name="Historical",
            line=dict(color="#6366f1", width=2),
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["forecast"],
            mode="lines", name="Forecast",
            line=dict(color="#f59e0b", width=2, dash="dash"),
        ))

        # Confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(245,158,11,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence",
        ))

        fig.update_layout(
            title=f"{self.value_col} Forecast — {method}",
            xaxis_title="Date",
            yaxis_title=self.value_col,
            hovermode="x unified",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    # ── Accuracy metrics ──────────────────────────────────────────────────────

    def in_sample_metrics(self, method: str) -> dict:
        """Quick in-sample MAE / RMSE using last 20% as pseudo test."""
        split = max(1, int(self.n * 0.8))
        train_vals = self.values[:split]
        test_vals  = self.values[split:]
        n_test     = len(test_vals)

        temp = Forecaster(
            pd.DataFrame({self.date_col: self.dates[:split],
                          self.value_col: train_vals}),
            self.date_col, self.value_col
        )
        pred_df = temp.forecast(method, n_test)
        preds   = pred_df["forecast"].values[:n_test]

        mae  = float(np.mean(np.abs(test_vals - preds)))
        rmse = float(np.sqrt(np.mean((test_vals - preds) ** 2)))
        mape = float(np.mean(np.abs((test_vals - preds) /
                                    np.where(test_vals == 0, 1, test_vals))) * 100)
        return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE %": round(mape, 2)}

    # ── Utility ───────────────────────────────────────────────────────────────

    def _future_dates(self, periods: int) -> pd.DatetimeIndex:
        last = self.dates.iloc[-1]
        # Infer step from median gap
        if self.n >= 2:
            gaps = self.dates.diff().dropna()
            step = gaps.median()
        else:
            step = pd.Timedelta(days=1)
        return pd.date_range(start=last + step, periods=periods, freq=step)
