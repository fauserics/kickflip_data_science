import io
import json
import os
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    load_model, load_metadata, load_schema,
    validate_dataframe, pretty_metric_table
)

# ------------------------- Config básica -------------------------
st.set_page_config(
    page_title="Income Predictor – Demo lista para vender",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    "# 📈 Income Predictor\n"
    "**Prototipo escalable, listo para demos y ventas.**\n\n"
    "Entrena con datos públicos reales (OpenML *Adult*), compara modelos, "
    "elige el mejor y permite **scorear en tiempo real** por formulario o por archivo.\n"
)

with st.expander("ℹ️ Cómo funciona"):
    st.markdown(
        "- **Datos:** OpenML *Adult* (ingresos > USD 50k).\n"
        "- **Modelos:** Regresión Logística, Random Forest, XGBoost (selección automática por ROC AUC).\n"
        "- **Métricas:** ROC AUC (CV), precisión, F1 en hold-out.\n"
        "- **Uso:** formulario para un caso individual o upload de CSV para múltiples casos.\n"
        "- **Arquitectura:** Python + scikit-learn + Streamlit + GitHub Actions (re-entrenos opcionales).\n"
    )

# ------------------------- KPIs de cabecera -------------------------
meta_header = load_metadata()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Estado del modelo", "Cargado" if load_model() else "No encontrado")
with col2:
    st.metric("Mejor modelo", meta_header.get("best_model", "—"))
with col3:
    st.metric("Rows dataset", meta_header.get("rows", "—"))

st.divider()

# ------------------------- Tabs -------------------------
tab_overview, tab_train, tab_realtime, tab_batch, tab_monitor = st.tabs(
    ["📊 Overview", "🛠️ Entrenar & Comparar", "🧮 Scoring en tiempo real", "📂 Scoring por archivo", "📡 Monitor"]
)

# ========================= 1) OVERVIEW =========================
with tab_overview:
    meta = load_metadata()
    if meta:
        st.subheader("Resumen del experimento")
        cv_df = pretty_metric_table(meta.get("cv", []))
        st.dataframe(cv_df, use_container_width=True)
        holdout = meta.get("holdout", {})
        st.write("**Hold-out:**", holdout)

        # Gráfico simple
        if not cv_df.empty:
            fig = px.bar(
                cv_df, x="model", y="roc_auc",
                error_y="roc_auc_std",
                title="ROC AUC (CV) por modelo"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay metadatos aún. Ve a **Entrenar & Comparar** para generar el primer modelo.")

# ========================= 2) TRAIN =========================
with tab_train:
    st.subheader("Entrenar / Re-entrenar")
    st.caption("Esto descarga el dataset desde OpenML (requiere internet en el entorno donde se ejecute `train.py`).")
    run_train = st.button("🚀 Entrenar ahora", type="primary")

    if run_train:
        with st.spinner("Entrenando…"):
            # Ejecuta el script de entrenamiento
            exit_code = os.system("python train.py")
            # Pequeña espera para asegurar escritura de artefactos
            time.sleep(1.0)

        if exit_code == 0:
            # Limpia cualquier caché de Streamlit por si existiera
            try:
                st.cache_data.clear()
            except Exception:
                pass
            try:
                st.cache_resource.clear()
            except Exception:
                pass

            st.success("Entrenamiento finalizado. Refrescando métricas…")
            # Vuelve a ejecutar el script y recarga estado/artefactos
            st.rerun()
        else:
            st.error("Falló el entrenamiento. Revisá logs de consola / Actions.")

# ========================= 3) REAL-TIME SCORING =========================
with tab_realtime:
    st.subheader("Scoring individual")
    schema = load_schema()
    model = load_model()

    if not model or not schema:
        st.warning("Primero entrena un modelo en la pestaña **Entrenar & Comparar**.")
    else:
        with st.form("form_realtime"):
            inputs = {}
            for col in schema["columns"]:
                name = col["name"]
                if col["type"] == "numeric":
                    val = st.number_input(name, value=0.0, step=1.0, format="%.4f")
                else:
                    options = col.get("values") or []
                    if options:
                        val = st.selectbox(name, options=options, index=0 if options else None)
                    else:
                        val = st.text_input(name, "")
                inputs[name] = val

            submitted = st.form_submit_button("Scorear")

        if submitted:
            df = pd.DataFrame([inputs])
            df = validate_dataframe(df, schema)
            try:
                proba = float(model.predict_proba(df)[0, 1])
                st.success(f"Probabilidad de ingreso > 50k: **{proba:.3f}**")
                st.progress(proba)
            except Exception as e:
                st.error(f"No se pudo scorear el caso: {e}")

# ========================= 4) BATCH SCORING =========================
with tab_batch:
    st.subheader("Scoring por archivo (CSV)")
    st.caption("El CSV debe contener las columnas del esquema de entrada. Recibirás un archivo con una columna nueva: `score`.")
    schema = load_schema()
    model = load_model()

    if not model or not schema:
        st.warning("Primero entrena un modelo en la pestaña **Entrenar & Comparar**.")
    else:
        uploaded = st.file_uploader("Subí un CSV", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                df_val = validate_dataframe(df, schema)
                scores = model.predict_proba(df_val)[:, 1]
                out = df.copy()
                out["score"] = scores

                buf = io.BytesIO()
                out.to_csv(buf, index=False)
                st.download_button(
                    "⬇️ Descargar con scores",
                    data=buf.getvalue(),
                    file_name="scored.csv",
                    mime="text/csv"
                )
                st.success(f"Listo: {len(out)} filas scoreadas.")
            except Exception as e:
                st.error(f"No se pudo procesar el archivo: {e}")

# ========================= 5) MONITOR =========================
with tab_monitor:
    st.subheader("Información del modelo")
    meta = load_metadata()
    st.json(meta if meta else {"status": "No metadata yet"})
    st.caption("Tip: en producción podés registrar estos metadatos en un dashboard propio o DataDog/Prometheus.")
