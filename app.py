# app.py — Kickflip en Cloud Run con login Firebase y env vars (sin st.secrets)
# Basado en tu app original; mantiene tabs, entrenamiento, forecasting y EDA.

import io
import os
import time
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Opcional (para disparar GitHub Actions)
try:
    import requests
except Exception:
    requests = None

from auth import require_login, authorized_headers  # login Firebase (mod. aparte)
from utils import (
    load_model, load_metadata, load_schema,
    validate_dataframe, pretty_metric_table
)

# ========================= Config básica =========================
st.set_page_config(
    page_title="Income, Forecast & EDA – Demo lista para vender",
    page_icon="📈",
    layout="wide"
)

# ========================= Variables de entorno (REMPLAZAN st.secrets) =========================
def env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)

GH_OWNER         = env("GH_OWNER", "fauserics")
GH_REPO          = env("GH_REPO", "kickflip_data_science")
GH_REF           = env("GH_REF", "main")
GH_TOKEN         = env("GH_TOKEN")  # inyectar con Secret Manager si se usa
GH_WF_CLASSIFIER = env("GH_WF_CLASSIFIER", "retrain-classifier.yml")
GH_WF_FORECAST   = env("GH_WF_FORECAST", "retrain-forecast.yml")

# ========================= Barrera de login (antes de dibujar la app) =========================
uid, id_token, email = require_login("Kickflip")
st.success("Autenticado ✅")
st.write(f"UID: {uid}")
st.write(f"Email: {email}")

st.markdown(
    "# 📈 Income, 🔮 Forecast & 🔍 EDA – Demo\n"
    "**Prototipo escalable, listo para demos y ventas.**\n\n"
    "Entrena, compara y sirve un modelo tabular (OpenML *Adult*), scorea en tiempo real por formulario o archivo, "
    "pronostica series con SARIMAX y hace **EDA automático**."
)

with st.expander("ℹ️ Cómo funciona"):
    st.markdown(
        "- **Clasificador:** OpenML *Adult* (ingresos > USD 50k). Modelos: LogReg, RandomForest, XGBoost (selección por ROC AUC).\n"
        "- **Forecasting:** SARIMAX (statsmodels). Demo pre-entrenada + entreno al vuelo con CSV o datos manuales.\n"
        "- **EDA:** modo **Ligero (nativo)** con Plotly y **Rápido (ydata-profiling)** embebido.\n"
        "- **Arquitectura:** GitHub Actions entrena y commitea artefactos; Cloud Run sirve la UI y/o APIs.\n"
    )

# ========================= KPIs de cabecera =========================
meta_header = load_metadata()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Estado del modelo", "Cargado" if load_model() else "No encontrado")
with col2:
    st.metric("Mejor modelo", meta_header.get("best_model", "—"))
with col3:
    st.metric("Rows dataset", meta_header.get("rows", "—"))

st.divider()

# ========================= Tabs principales =========================
tab_overview, tab_train, tab_realtime, tab_batch, tab_monitor, tab_forecast, tab_eda = st.tabs(
    ["📊 Overview", "🛠️ Entrenar", "🧮 Scoring en tiempo real", "📂 Scoring por archivo", "📡 Monitor", "🔮 Forecasting", "🔍 EDA"]
)

# ========================= 1) OVERVIEW =========================
with tab_overview:
    meta = load_metadata()
    if meta:
        st.subheader("Resumen del experimento (Clasificador)")
        cv_df = pretty_metric_table(meta.get("cv", []))
        st.dataframe(cv_df, use_container_width=True)
        holdout = meta.get("holdout", {})
        st.write("**Hold-out:**", holdout)
        if not cv_df.empty:
            fig = px.bar(
                cv_df, x="model", y="roc_auc",
                error_y="roc_auc_std",
                title="ROC AUC (CV) por modelo"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay metadatos aún. Ve a **Entrenar** para generar el primer modelo.")

# ========================= 2) TRAIN =========================
with tab_train:
    st.subheader("Entrenar / Re-entrenar")

    subtab_actions, subtab_local = st.tabs(["🚀 GitHub Actions (recomendado)", "🧪 Local en el runtime (opcional)"])

    # ---------- A) Disparar GitHub Actions ----------
    with subtab_actions:
        st.caption("Dispara los workflows de GitHub. Cuando terminen y hagan push, la app usará los modelos nuevos.")

        def dispatch_workflow(owner: str, repo: str, workflow_file_or_name: str, ref: str, token: str):
            """
            POST /repos/{owner}/{repo}/actions/workflows/{workflow_id|file_name}/dispatches
            """
            if requests is None:
                raise RuntimeError("El paquete 'requests' no está instalado.")
            if not token:
                raise RuntimeError("Falta GH_TOKEN (inyectalo como secreto en Cloud Run).")
            workflow_id = workflow_file_or_name.split("/")[-1]
            url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches"
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
            return requests.post(url, headers=headers, json={"ref": ref}, timeout=30)

        missing = not all([GH_OWNER, GH_REPO]) or (requests is None)
        if missing:
            st.warning("Definí GH_OWNER/GH_REPO (env vars) y opcional GH_REF/GH_WF_*; instala 'requests' si falta.")

        colA, colB = st.columns(2)
        with colA:
            if st.button("🔁 Re-entrenar **Clasificador** (Adult)"):
                if missing:
                    st.error("Faltan env vars o el paquete 'requests'.")
                else:
                    with st.spinner("Lanzando workflow del clasificador…"):
                        r = dispatch_workflow(GH_OWNER, GH_REPO, GH_WF_CLASSIFIER, GH_REF, GH_TOKEN)
                    if r.status_code in (201, 204):
                        st.success("Workflow del clasificador disparado. Al terminar y pushear, la app usará el nuevo modelo.")
                    else:
                        st.error(f"Error {r.status_code}: {r.text}")

        with colB:
            if st.button("🔁 Re-entrenar **Forecasting**"):
                if missing:
                    st.error("Faltan env vars o el paquete 'requests'.")
                else:
                    with st.spinner("Lanzando workflow de forecasting…"):
                        r = dispatch_workflow(GH_OWNER, GH_REPO, GH_WF_FORECAST, GH_REF, GH_TOKEN)
                    if r.status_code in (201, 204):
                        st.success("Workflow de forecasting disparado. Al terminar y pushear, la app usará el nuevo modelo.")
                    else:
                        st.error(f"Error {r.status_code}: {r.text}")

        if GH_OWNER and GH_REPO:
            st.link_button("🧭 Ver runs en GitHub Actions", f"https://github.com/{GH_OWNER}/{GH_REPO}/actions", type="secondary")

    # ---------- B) Entrenamiento local (opcional; puede fallar en serverless) ----------
    with subtab_local:
        st.caption("Ejecuta `python train.py` en este entorno. En Cloud Run puede fallar por límites.")
        if st.button("▶️ Entrenar aquí (local)"):
            with st.spinner("Entrenando…"):
                exit_code = os.system("python train.py")
                time.sleep(1.0)
            if exit_code == 0:
                for clear_fn in (getattr(st, "cache_data", None), getattr(st, "cache_resource", None)):
                    try:
                        if clear_fn:
                            clear_fn.clear()
                    except Exception:
                        pass
                st.success("Entrenamiento finalizado. Refrescando métricas…")
                st.rerun()
            else:
                st.error("Falló el entrenamiento. Revisá logs (consola/Actions).")

# ========================= 3) REAL-TIME SCORING (Clasificador) =========================
with tab_realtime:
    st.subheader("Scoring individual (Clasificador)")
    schema = load_schema()
    model = load_model()

    if not model or not schema:
        st.warning("Primero entrená un modelo en la pestaña **Entrenar**.")
    else:
        with st.form("form_realtime"):
            inputs: Dict[str, Any] = {}
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
        st.warning("Primero entrená un modelo en la pestaña **Entrenar**.")
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
    st.subheader("Información del modelo (Clasificador)")
    meta = load_metadata()
    st.json(meta if meta else {"status": "No metadata yet"})
    st.caption("Tip: en producción podés registrar estos metadatos en un dashboard propio o DataDog/Prometheus.")

# ========================= 6) FORECASTING =========================
with tab_forecast:
    st.subheader("Forecasting en tiempo real")

    # --- Controles de entrenamiento dentro de Forecasting ---
    with st.container(border=True):
        st.markdown("### ⚙️ Entrenamiento del modelo de forecast (demo)")
        colA, colB = st.columns(2)

        # Botón A: GitHub Actions (recomendado)
        with colA:
            def dispatch_workflow(owner: str, repo: str, workflow_file_or_name: str, ref: str, token: str):
                if requests is None:
                    raise RuntimeError("El paquete 'requests' no está instalado.")
                if not token:
                    raise RuntimeError("Falta GH_TOKEN (inyectalo como secreto en Cloud Run).")
                workflow_id = workflow_file_or_name.split("/")[-1]
                url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches"
                headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
                return requests.post(url, headers=headers, json={"ref": ref}, timeout=30)

            disabled = not all([GH_OWNER, GH_REPO]) or (requests is None)
            if st.button("🔁 Re-entrenar forecasting en GitHub"):
                if disabled:
                    st.error("Faltan env vars o el paquete 'requests'.")
                else:
                    with st.spinner("Lanzando workflow de forecasting en GitHub Actions…"):
                        r = dispatch_workflow(GH_OWNER, GH_REPO, GH_WF_FORECAST, GH_REF, GH_TOKEN)
                    if r.status_code in (201, 204):
                        st.success("Workflow disparado. Al terminar y pushear, esta app tomará el nuevo modelo.")
                    else:
                        st.error(f"Error {r.status_code}: {r.text}")

        # Botón B: Entrenamiento local (opcional)
        with colB:
            st.caption("Opción local (puede fallar en entornos serverless).")
            if st.button("▶️ Entrenar aquí con forecast_train.py"):
                with st.spinner("Entrenando modelo SARIMAX…"):
                    code = os.system("python forecast_train.py")
                    time.sleep(1.0)
                if code == 0:
                    for clear_fn in (getattr(st, "cache_data", None), getattr(st, "cache_resource", None)):
                        try:
                            if clear_fn:
                                clear_fn.clear()
                        except Exception:
                            pass
                    st.success("Entrenamiento finalizado. Recargando…")
                    st.rerun()
                else:
                    st.error("Falló el entrenamiento local. Usá el botón de GitHub Actions.")

    # --- Utils de forecasting (sin pmdarima) ---
    from forecast_utils import (
        load_demo_forecast_model, load_demo_forecast_meta,
        parse_ts, fit_quick_sarimax, forecast_to_df
    )

    # Helper para construir la serie histórica desde statsmodels
    def endog_series(res, saved_index: Optional[pd.DatetimeIndex] = None, meta: Optional[dict] = None) -> pd.Series:
        endog = np.asarray(res.model.endog).ravel()

        def _same_len(idx, n) -> bool:
            try:
                return idx is not None and len(idx) == n
            except Exception:
                return False

        idx = saved_index if _same_len(saved_index, len(endog)) else None

        if idx is None:
            try:
                idx2 = res.model.data.row_labels
                if _same_len(idx2, len(endog)):
                    idx = idx2
            except Exception:
                idx = None

        if idx is None:
            freq = (meta or {}).get("freq", "MS")
            start = pd.to_datetime((meta or {}).get("train_start", "2000-01-01"))
            idx = pd.date_range(start=start, periods=len(endog), freq=freq)

        return pd.Series(endog, index=pd.DatetimeIndex(idx), name="y")

    # --- Sub-tabs de forecasting ---
    sub_demo, sub_upload, sub_manual = st.tabs(["📦 Demo pre-entrenada", "📤 Tu serie (CSV)", "✍️ Ingresar manualmente"])

    # --- A) DEMO PRE-ENTRENADA ---
    with sub_demo:
        fobj = load_demo_forecast_model()
        meta_f = load_demo_forecast_meta()
        if not fobj or not meta_f:
            st.warning("Aún no hay modelo de forecasting demo. Corré `retrain-forecast` en Actions o el botón de arriba.")
        else:
            res = fobj["model"]
            hist = endog_series(res, saved_index=fobj.get("endog_index"), meta=meta_f)

            steps = st.number_input("Horizonte (pasos)", 1, 120, 12)
            yhat = res.get_forecast(int(steps)).predicted_mean.values
            out = forecast_to_df(hist, yhat, int(steps))

            st.write("**Meta del modelo:**", meta_f)
            fig = go.Figure()
            fig.add_scatter(x=hist.index, y=hist.values, mode="lines", name="histórico")
            fig.add_scatter(x=out["ds"], y=out["yhat"], mode="lines", name="pronóstico")
            fig.update_layout(title="Pronóstico (demo)", xaxis_title="Fecha", yaxis_title="Valor")
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "⬇️ Descargar pronóstico (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                "forecast_demo.csv",
                "text/csv"
            )

    # --- B) SUBIR SERIE (CSV) ---
    with sub_upload:
        st.caption("Formato CSV con columnas: `ds,y` o `date,value` (fechas ISO o dd/mm/aaaa).")
        f = st.file_uploader("Subí tu serie temporal", type=["csv"])
        steps2 = st.number_input("Horizonte a pronosticar", 1, 120, 12, key="steps_upload")
        if f:
            try:
                df = pd.read_csv(f)
                s, freq = parse_ts(df)
                st.write(f"Frecuencia detectada: **{freq}** — Observaciones: **{len(s)}**")

                with st.spinner("Entrenando SARIMAX…"):
                    res2, order, seas, aic = fit_quick_sarimax(s)
                    yhat2 = res2.get_forecast(int(steps2)).predicted_mean.values
                out2 = forecast_to_df(s, yhat2, int(steps2))

                fig2 = go.Figure()
                fig2.add_scatter(x=s.index, y=s.values, mode="lines", name="histórico")
                fig2.add_scatter(x=out2["ds"], y=out2["yhat"], mode="lines", name="pronóstico")
                fig2.update_layout(
                    title=f"Pronóstico (orden={order}, seasonal={seas}, AIC={aic:.1f})",
                    xaxis_title="Fecha", yaxis_title="Valor"
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.download_button(
                    "⬇️ Descargar pronóstico (CSV)",
                    out2.to_csv(index=False).encode("utf-8"),
                    "forecast_uploaded.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"No se pudo procesar la serie: {e}")

    # --- C) INGRESAR MANUALMENTE ---
    with sub_manual:
        from datetime import date, timedelta
        st.caption("Cargá una serie corta a mano para probar el pronóstico al vuelo.")

        # Opción A: Editor de tabla con plantilla
        st.markdown("#### Opción A: Editor de tabla")
        today = pd.to_datetime(date.today())
        df_default = pd.DataFrame({
            "ds": [today + timedelta(days=i) for i in range(6)],
            "y": [100, 102, 101, 103, 104, 105],
        })
        df_edit = st.data_editor(
            df_default,
            num_rows="dynamic",
            use_container_width=True,
            key="ts_manual_editor",
            column_config={
                "ds": st.column_config.DateColumn("Fecha (ds)"),
                "y": st.column_config.NumberColumn("Valor (y)", step=0.1),
            }
        )

        # Opción B: Pegar texto CSV
        st.markdown("#### Opción B: Pegar texto (CSV rápido)")
        st.caption("Formato: 2 columnas `ds,y` separadas por coma. Ejemplo:\n\n```\n2024-01-01,100\n2024-01-02,101.5\n```")
        txt = st.text_area("Pegar aquí", value="", height=140, key="ts_manual_text")

        steps3 = st.number_input("Horizonte a pronosticar", 1, 120, 12, key="steps_manual")
        run_manual = st.button("🔮 Entrenar & pronosticar (manual)")

        if run_manual:
            try:
                # Elegir fuente: texto si está, si no el editor
                if txt.strip():
                    df_in = pd.read_csv(io.StringIO(txt.strip()), header=None, names=["ds", "y"])
                else:
                    df_in = df_edit.copy()

                if df_in.shape[1] < 2:
                    raise ValueError("Se necesitan dos columnas: fecha y valor (ds,y).")

                df_in = df_in.rename(columns={df_in.columns[0]: "ds", df_in.columns[1]: "y"})
                df_in["ds"] = pd.to_datetime(df_in["ds"], errors="coerce")
                df_in["y"] = pd.to_numeric(df_in["y"], errors="coerce")
                df_in = df_in.dropna().sort_values("ds")
                if len(df_in) < 6:
                    raise ValueError("Ingresá al menos 6 observaciones para entrenar.")

                s, freq = parse_ts(df_in)
                with st.spinner(f"Entrenando SARIMAX (freq detectada: {freq})…"):
                    res3, order, seas, aic = fit_quick_sarimax(s)
                    yhat3 = res3.get_forecast(int(steps3)).predicted_mean.values

                out3 = forecast_to_df(s, yhat3, int(steps3))

                fig3 = go.Figure()
                fig3.add_scatter(x=s.index, y=s.values, mode="lines", name="histórico")
                fig3.add_scatter(x=out3["ds"], y=out3["yhat"], mode="lines", name="pronóstico")
                fig3.update_layout(
                    title=f"Pronóstico manual (orden={order}, seasonal={seas}, AIC={aic:.1f})",
                    xaxis_title="Fecha", yaxis_title="Valor"
                )
                st.plotly_chart(fig3, use_container_width=True)

                st.download_button(
                    "⬇️ Descargar pronóstico (CSV)",
                    out3.to_csv(index=False).encode("utf-8"),
                    "forecast_manual.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"No se pudo entrenar/forecastear con los datos ingresados: {e}")

# ========================= 7) EDA =========================
with tab_eda:
    st.subheader("Análisis Exploratorio de Datos (EDA)")

    mode = st.radio("Modo de EDA", ["Ligero (nativo)", "Rápido (ydata-profiling)"], horizontal=True)
    f = st.file_uploader("Subí un dataset (CSV, Parquet)", type=["csv", "parquet"])

    if f is None:
        st.info("Subí un archivo para comenzar. Tip: soporta CSV y Parquet.")
    else:
        # Lectura del dataset
        try:
            if f.name.lower().endswith(".parquet"):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)
        except Exception as e:
            st.error(f"No pude leer el archivo: {e}")
            df = None

        if df is not None:
            st.success(f"Cargado: {df.shape[0]:,} filas × {df.shape[1]:,} columnas")
            st.dataframe(df.head(10), use_container_width=True)

            if mode.startswith("Rápido"):
                # Auto-EDA con ydata-profiling (opcional)
                try:
                    from eda_utils import render_ydata_profile_html
                    with st.spinner("Generando reporte…"):
                        html = render_ydata_profile_html(df)
                    import streamlit.components.v1 as components
                    components.html(html, height=900, scrolling=True)
                except Exception as e:
                    st.error(f"No pude generar el reporte de ydata-profiling: {e}")
                    st.info("Probá el modo 'Ligero (nativo)' para un análisis más rápido.")
            else:
                # EDA nativo (ligero) con Plotly
                from eda_utils import (
                    basic_overview, split_columns, plot_numeric_hist, plot_categoric_bar,
                    corr_heatmap, missing_bar, target_relationships, dataframe_head_csv
                )

                # Resumen
                info = basic_overview(df)
                c1, c2, c3 = st.columns(3)
                c1.metric("Filas", info["rows"])
                c2.metric("Columnas", info["cols"])
                c3.metric("Memoria (MB)", info["memory_mb"])
                with st.expander("Tipos, cardinalidad y faltantes"):
                    st.write("**Tipos:**", info["dtypes"])
                    st.write("**# Únicos:**", info["nunique"])
                    st.write("**Faltantes:**", info["missing"])

                # Distribuciones
                num_cols, cat_cols = split_columns(df)
                st.markdown("#### Distribuciones")
                if num_cols:
                    colnum = st.multiselect("Numéricas a graficar", num_cols, default=num_cols[:4], key="eda_num")
                    for col in colnum:
                        st.plotly_chart(plot_numeric_hist(df, col), use_container_width=True)
                if cat_cols:
                    colcat = st.multiselect("Categóricas a graficar", cat_cols, default=cat_cols[:4], key="eda_cat")
                    for col in colcat:
                        st.plotly_chart(plot_categoric_bar(df, col), use_container_width=True)

                # Correlaciones y faltantes
                st.markdown("#### Correlaciones y faltantes")
                fig_corr = corr_heatmap(df, num_cols)
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Se necesitan al menos 2 columnas numéricas para la correlación.")
                fig_miss = missing_bar(df)
                if fig_miss:
                    st.plotly_chart(fig_miss, use_container_width=True)

                # Relación con target (opcional)
                st.markdown("#### Análisis por variable objetivo (opcional)")
                target = st.selectbox("Seleccioná una columna objetivo (opcional)", ["—"] + df.columns.tolist())
                if target != "—":
                    plots = target_relationships(df, target, num_cols, cat_cols)
                    for fig in plots:
                        st.plotly_chart(fig, use_container_width=True)

                # Export rápido (muestra)
                st.download_button(
                    "⬇️ Descargar muestra (CSV, 50 filas)",
                    data=dataframe_head_csv(df, n=50),
                    file_name="sample_50.csv",
                    mime="text/csv"
                )
