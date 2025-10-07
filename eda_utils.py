# eda_utils.py
from __future__ import annotations
import io
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def basic_overview(df: pd.DataFrame) -> dict:
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    nunique = df.nunique(dropna=False).to_dict()
    missing = df.isna().sum().to_dict()
    dtypes = df.dtypes.astype(str).to_dict()
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "memory_mb": round(mem_mb, 2),
        "dtypes": dtypes,
        "nunique": nunique,
        "missing": missing,
    }

def split_columns(df: pd.DataFrame, max_cat_card: int = 30) -> Tuple[List[str], List[str]]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categoric = [c for c in df.columns if c not in numeric]
    # re-clasifica numéricas de baja cardinalidad como categóricas si conviene graficar
    for c in list(numeric):
        if df[c].dropna().nunique() <= max_cat_card and df[c].dtype != bool:
            # mantener numéricas como numéricas; la decisión de gráfico se hace luego
            pass
    return numeric, categoric

def plot_numeric_hist(df: pd.DataFrame, col: str):
    fig = px.histogram(df, x=col, nbins=40, marginal="box", opacity=0.8)
    fig.update_layout(title=f"Distribución de {col}")
    return fig

def plot_categoric_bar(df: pd.DataFrame, col: str, top: int = 20):
    vc = df[col].astype(str).fillna("NA").value_counts().head(top)
    fig = px.bar(x=vc.index, y=vc.values, labels={"x": col, "y": "count"})
    fig.update_layout(title=f"Top {top} categorías – {col}", xaxis_tickangle=-30)
    return fig

def corr_heatmap(df: pd.DataFrame, numeric_cols: List[str]):
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de correlación (numéricas)")
    return fig

def missing_bar(df: pd.DataFrame):
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return None
    fig = px.bar(x=miss.index, y=(miss.values * 100), labels={"x": "columna", "y": "% faltantes"})
    fig.update_layout(title="% de faltantes por columna", xaxis_tickangle=-30)
    return fig

def target_relationships(df: pd.DataFrame, target: str, numeric_cols: List[str], categoric_cols: List[str]):
    plots = []
    if target in numeric_cols:
        # target numérico: scatter para primeras numéricas
        for col in numeric_cols[:6]:
            if col == target: 
                continue
            fig = px.scatter(df, x=col, y=target, trendline="ols")
            fig.update_layout(title=f"{target} vs {col}")
            plots.append(fig)
    else:
        # target categórico/binario: distros por clase
        for col in numeric_cols[:6]:
            fig = px.histogram(df, x=col, color=target, barmode="overlay", nbins=40, opacity=0.6)
            fig.update_layout(title=f"Distribución de {col} por {target}")
            plots.append(fig)
        for col in categoric_cols[:6]:
            vc = (df.groupby(col)[target].value_counts(normalize=True)
                    .rename("prop").reset_index())
            fig = px.bar(vc, x=col, y="prop", color=target, barmode="group")
            fig.update_layout(title=f"Proporciones de {target} por {col}", xaxis_tickangle=-30)
            plots.append(fig)
    return plots

def dataframe_head_csv(df: pd.DataFrame, n: int = 50) -> bytes:
    return df.head(n).to_csv(index=False).encode("utf-8")

# --------- ydata-profiling helper (opcional) ----------
def render_ydata_profile_html(df: pd.DataFrame) -> str:
    """Devuelve HTML como string; úsese con st.components.v1.html."""
    from ydata_profiling import ProfileReport
    pr = ProfileReport(df, title="EDA Report", minimal=True, explorative=True)
    return pr.to_html()
