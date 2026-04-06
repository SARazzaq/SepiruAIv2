"""
Auto ML trainer — detects classification vs regression and trains models.
Includes latest and most robust ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor,
    BaggingClassifier, BaggingRegressor,
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
import plotly.express as px
import plotly.graph_objects as go

DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,15,26,0.6)",
    font=dict(family="DM Sans, sans-serif", color="#9090a8", size=11),
    title_font=dict(family="DM Serif Display, serif", color="#e8e6e0", size=15),
)

# ── Model registries ──────────────────────────────────────────────────────────

def get_classification_models() -> Dict:
    models = {
        # ── Tree-based (most robust) ──────────────────────────────
        "🥇 Random Forest":         RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "🥇 Extra Trees":           ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "🥇 Gradient Boosting":     GradientBoostingClassifier(n_estimators=200, random_state=42),
        # ── Boosting (latest & best) ──────────────────────────────
        "⚡ XGBoost":               None,  # loaded dynamically
        "⚡ LightGBM":              None,
        "⚡ CatBoost":              None,
        # ── Linear ───────────────────────────────────────────────
        "📐 Logistic Regression":   LogisticRegression(max_iter=1000, random_state=42),
        "📐 SGD Classifier":        SGDClassifier(max_iter=1000, random_state=42),
        # ── Distance-based ───────────────────────────────────────
        "📍 KNN":                   KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        # ── Support Vector ────────────────────────────────────────
        "🔷 SVM (RBF kernel)":      SVC(kernel="rbf", probability=True, random_state=42),
        # ── Single tree ───────────────────────────────────────────
        "🌿 Decision Tree":         DecisionTreeClassifier(random_state=42),
    }

    # Try loading optional boosting libs
    try:
        from xgboost import XGBClassifier
        models["⚡ XGBoost"] = XGBClassifier(
            n_estimators=200, random_state=42,
            eval_metric="logloss", verbosity=0, n_jobs=-1
        )
    except ImportError:
        del models["⚡ XGBoost"]

    try:
        import lightgbm as lgb
        models["⚡ LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, random_state=42,
            verbose=-1, n_jobs=-1
        )
    except ImportError:
        del models["⚡ LightGBM"]

    try:
        from catboost import CatBoostClassifier
        models["⚡ CatBoost"] = CatBoostClassifier(
            iterations=200, random_seed=42, verbose=0
        )
    except ImportError:
        del models["⚡ CatBoost"]

    return {k: v for k, v in models.items() if v is not None}


def get_regression_models() -> Dict:
    models = {
        # ── Tree-based ────────────────────────────────────────────
        "🥇 Random Forest":         RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "🥇 Extra Trees":           ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "🥇 Gradient Boosting":     GradientBoostingRegressor(n_estimators=200, random_state=42),
        # ── Boosting ─────────────────────────────────────────────
        "⚡ XGBoost":               None,
        "⚡ LightGBM":              None,
        "⚡ CatBoost":              None,
        # ── Linear ───────────────────────────────────────────────
        "📐 Linear Regression":     LinearRegression(),
        "📐 Ridge Regression":      Ridge(alpha=1.0),
        "📐 Lasso Regression":      Lasso(alpha=0.1, max_iter=2000),
        "📐 ElasticNet":            ElasticNet(alpha=0.1, max_iter=2000),
        "📐 SGD Regressor":         SGDRegressor(max_iter=1000, random_state=42),
        # ── Distance ─────────────────────────────────────────────
        "📍 KNN Regressor":         KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        # ── Support Vector ────────────────────────────────────────
        "🔷 SVR (RBF kernel)":      SVR(kernel="rbf"),
        # ── Tree ─────────────────────────────────────────────────
        "🌿 Decision Tree":         DecisionTreeRegressor(random_state=42),
    }

    try:
        from xgboost import XGBRegressor
        models["⚡ XGBoost"] = XGBRegressor(
            n_estimators=200, random_state=42, verbosity=0, n_jobs=-1
        )
    except ImportError:
        del models["⚡ XGBoost"]

    try:
        import lightgbm as lgb
        models["⚡ LightGBM"] = lgb.LGBMRegressor(
            n_estimators=200, random_state=42,
            verbose=-1, n_jobs=-1
        )
    except ImportError:
        del models["⚡ LightGBM"]

    try:
        from catboost import CatBoostRegressor
        models["⚡ CatBoost"] = CatBoostRegressor(
            iterations=200, random_seed=42, verbose=0
        )
    except ImportError:
        del models["⚡ CatBoost"]

    return {k: v for k, v in models.items() if v is not None}


# ── Data prep ─────────────────────────────────────────────────────────────────

def detect_task(df: pd.DataFrame, target_col: str) -> str:
    col = df[target_col].dropna()
    if col.dtype == object or col.nunique() <= 20:
        return "classification"
    return "regression"


def prepare_data(df: pd.DataFrame, target_col: str) -> Tuple:
    data = df.copy().dropna()
    X = data.drop(columns=[target_col])
    y = data[target_col]

    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    le_target = None
    if y.dtype == object:
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))

    X = X.select_dtypes(include="number")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, list(X.columns), le_target


# ── Auto compare all models ───────────────────────────────────────────────────

def compare_all_models(X_train, X_test, y_train, y_test,
                        task: str) -> pd.DataFrame:
    """Train all models and return leaderboard."""
    models = (get_classification_models() if task == "classification"
              else get_regression_models())
    rows = []
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            if task == "classification":
                score = round(accuracy_score(y_test, preds) * 100, 2)
                rows.append({"Model": name, "Accuracy %": score})
            else:
                mae  = round(float(mean_absolute_error(y_test, preds)), 4)
                r2   = round(float(r2_score(y_test, preds)), 4)
                rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
                rows.append({"Model": name, "R²": r2, "MAE": mae, "RMSE": rmse})
        except Exception as e:
            rows.append({"Model": name, "Error": str(e)})

    df = pd.DataFrame(rows)
    sort_col = "Accuracy %" if task == "classification" else "R²"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)
    return df.reset_index(drop=True)


# ── Train single model ────────────────────────────────────────────────────────

def train_classification(X_train, X_test, y_train, y_test,
                          feature_names, model_name, le_target=None):
    models = get_classification_models()
    model  = models.get(model_name, models["🥇 Random Forest"])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc    = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    cm     = confusion_matrix(y_test, preds)
    labels = le_target.classes_.tolist() if le_target else sorted(set(y_test))

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    cm_fig = px.imshow(
        cm, text_auto=True,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        color_continuous_scale="YlOrBr",
        title="Confusion Matrix",
    )
    cm_fig.update_layout(**DARK)

    fi_fig = None
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True).tail(15)
        fi_fig = px.bar(
            fi, x="Importance", y="Feature", orientation="h",
            title="Feature Importance",
            color="Importance", color_continuous_scale="YlOrBr"
        )
        fi_fig.update_layout(**DARK, coloraxis_showscale=False)

    return {
        "task": "classification",
        "model_name": model_name,
        "accuracy": round(acc * 100, 2),
        "cv_mean": round(cv_scores.mean() * 100, 2),
        "cv_std":  round(cv_scores.std()  * 100, 2),
        "report":  report,
        "cm_fig":  cm_fig,
        "fi_fig":  fi_fig,
        "model":   model,
    }


def train_regression(X_train, X_test, y_train, y_test,
                      feature_names, model_name):
    models = get_regression_models()
    model  = models.get(model_name, models["🥇 Random Forest"])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae  = round(float(mean_absolute_error(y_test, preds)), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
    r2   = round(float(r2_score(y_test, preds)), 4)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                 scoring="r2")

    ap_fig = go.Figure()
    ap_fig.add_trace(go.Scatter(
        x=list(y_test), y=list(preds), mode="markers",
        marker=dict(color="#c9a84c", size=5, opacity=0.7),
        name="Predictions"
    ))
    ap_fig.add_trace(go.Scatter(
        x=[min(y_test), max(y_test)],
        y=[min(y_test), max(y_test)],
        mode="lines",
        line=dict(color="#6366f1", dash="dash"),
        name="Perfect fit"
    ))
    ap_fig.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Actual", yaxis_title="Predicted", **DARK
    )

    residuals = np.array(y_test) - preds
    res_fig = px.histogram(
        x=residuals, nbins=30,
        title="Residuals Distribution",
        color_discrete_sequence=["#c9a84c"]
    )
    res_fig.update_layout(**DARK)

    fi_fig = None
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True).tail(15)
        fi_fig = px.bar(
            fi, x="Importance", y="Feature", orientation="h",
            title="Feature Importance",
            color="Importance", color_continuous_scale="YlOrBr"
        )
        fi_fig.update_layout(**DARK, coloraxis_showscale=False)

    return {
        "task": "regression",
        "model_name": model_name,
        "mae": mae, "rmse": rmse, "r2": r2,
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std":  round(cv_scores.std(),  4),
        "ap_fig":  ap_fig,
        "res_fig": res_fig,
        "fi_fig":  fi_fig,
        "model":   model,
    }