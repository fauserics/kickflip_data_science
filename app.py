import io
import os
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Si vas a disparar GitHub Actions desde la app, necesitás requests
try:
    import requests  # añadido a requirements.txt
except Exception:
    requests = None

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
        "- **Arquitectura recomendada:** GitHub Actions entrena y commitea el modelo; "
        "Streamlit Cloud sirve la UI y **lee** el modelo del repo.\n"
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
tab_overview, tab_train, tab_realtime, tab_batch, tab_monitor, tab_forecast = st.tabs(
    ["📊 Overview", "🛠️ Entrenar", "🧮 Scoring en tiempo real", "📂 Scoring por archivo", "📡 Monitor", "🔮 Forecasting"]
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
        st.caption("Dispara el workflow de GitHub; cuando termine y haga push, la app usará el nuevo modelo.")
        owner = st.secrets.get("GH_OWNER")
        repo = st.secrets.get("GH_REPO")
        ref = st.secrets.get("GH_REF", "main")
        workflow_path = st.secrets.get("GH_WORKFLOW", ".github/workflows/train.yml")
        token = st.secrets.get("GH_TOKEN")

        # URL útil para ver el workflow (si configuraste owner/repo)
        if owner and repo:
            actions_url = f"https://github.com/{owner}/{repo}/actions"
            st.link_button("🧭 Ver workflows en GitHub Actions", actions_url, type="secondary")

        def trigger_workflow(owner: str, repo: str, workflow_path: str, ref: str, token: str):
            """
            Dispara workflow_dispatch. GitHub acepta:
            /actions/workflows/{workflow_id|file_name}/dispatches
            """
            # Usamos el nombre del archivo del workflow
            workflow_file = workflow_path.split("/")[-1]
            url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_file}/dispatches"
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json"
            }
            data = {"ref": ref}
            return requests.post(url, headers=headers, json=data, timeout=30)

        disabled = not all([owner, repo, token]) or (requests is None)
        if not requests:
            st.warning("Falta el paquete 'requests'. Asegurate de tenerlo en requirements.txt (requests>=2.31).")

        with st.form("form_actions"):
            run_now = st.form_submit_button("🔁 Disparar re-entreno en GitHub")
        if run_now:
            if disabled:
                st.error("Faltan secretos GH_TOKEN/GH_OWNER/GH_REPO o el paquete 'requests'. Configuralos en Settings → Secrets de Streamlit.")
            else:
                with st.spinner("Lanzando workflow en GitHub Actions…"):
                    r = trigger_workflow(owner, repo, workflow_path, ref, token)
                if r.status_code in (201, 204):
                    st.success("Workflow disparado correctamente.")
                    st.info("Cuando el workflow termine y haga push, esta app leerá el nuevo modelo del repo.")
                else:
                    st.error(f"Error al disparar el workflow ({r.status_code}): {r.text}")

    # ---------- B) Entrenamiento local (opcional; puede fallar en Streamlit Cloud) ----------
    with subtab_local:
        st.caption("Ejecuta `python train.py` en el entorno actual. Útil en local; en Streamlit Cloud puede fallar por límites del entorno.")
        if st.button("▶️ Entrenar aquí (local)"):
            with st.spinner("Entrenando…"):
                exit_code = os.system("python train.py")
                time.sleep(1.0)
            if exit_code == 0:
                # Limpiar caché y forzar recarga
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                try:
                    st.cache_resource.clear()
                except Exception:
                    pass
                st.success("Entrenamiento finalizado. Refrescando métricas…")
                st.rerun()
            else:
                st.error("Falló el entrenamiento. Revisá logs (consola/Actions).")

# ========================= 3) REAL-TIME SCORING =========================
with tab_realtime:
    st.subheader("Scoring individual")
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
    st.subheader("Información del modelo")
    meta = load_metadata()
    st.json(meta if meta else {"status": "No metadata yet"})
    st.caption("Tip: en producción podés registrar estos metadatos en un dashboard propio o DataDog/Prometheus.")


# ========================= 6) FORECASTING =========================
with tab_forecast:
    import plotly.graph_objects as go
    from forecast_utils import (
        load_demo_forecast_model, load_demo_forecast_meta,
        parse_ts, fit_quick_sarimax, forecast_to_df
    )

    st.subheader("Forecasting en tiempo real")

    # --- Controles de entrenamiento dentro de Forecasting ---
with st.container(border=True):
    st.markdown("### ⚙️ Entrenamiento del modelo de forecast (demo)")
    colA, colB = st.columns(2)

    # Botón A: Disparar workflow de GitHub Actions (recomendado)
    with colA:
        try:
            import requests
        except Exception:
            requests = None

        owner = st.secrets.get("GH_OWNER")
        repo  = st.secrets.get("GH_REPO")
        ref   = st.secrets.get("GH_REF", "main")
        token = st.secrets.get("GH_TOKEN")
        wf_forecast = st.secrets.get("GH_WF_FORECAST", "retrain-forecast.yml")

        def dispatch_workflow(owner: str, repo: str, workflow_file_or_name: str, ref: str, token: str):
            workflow_id = workflow_file_or_name.split("/")[-1]
            url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches"
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
            return requests.post(url, headers=headers, json={"ref": ref}, timeout=30)

        disabled = not all([owner, repo, token]) or (requests is None)
        if st.button("🔁 Re-entrenar forecasting en GitHub"):
            if disabled:
                st.error("Faltan Secrets (GH_TOKEN/OWNER/REPO) o el paquete 'requests'.")
            else:
                with st.spinner("Lanzando workflow de forecasting en GitHub Actions…"):
                    r = dispatch_workflow(owner, repo, wf_forecast, ref, token)
                if r.status_code in (201, 204):
                    st.success("Workflow disparado. Al terminar y pushear, esta app tomará el nuevo modelo.")
                else:
                    st.error(f"Error {r.status_code}: {r.text}")

    # Botón B: Entrenamiento local (opcional; útil en local, no en Streamlit Cloud)
    with colB:
        st.caption("Opción local (puede fallar en Streamlit Cloud por límites de entorno).")
        if st.button("▶️ Entrenar aquí con forecast_train.py"):
            with st.spinner("Entrenando modelo SARIMAX…"):
                code = os.system("python forecast_train.py")
                time.sleep(1.0)
            if code == 0:
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                try:
                    st.cache_resource.clear()
                except Exception:
                    pass
                st.success("Entrenamiento finalizado. Recargando…")
                st.rerun()
            else:
                st.error("Falló el entrenamiento local. Usá el botón de GitHub Actions.")


sub_demo, sub_upload, sub_manual = st.tabs(
    ["📦 Demo pre-entrenada", "📤 Tu serie (CSV)", "✍️ Ingresar manualmente"]
)
    # --- A) DEMO PRE-ENTRENADA ---
    with sub_demo:
        fobj = load_demo_forecast_model()
        meta = load_demo_forecast_meta()
        if not fobj or not meta:
            st.warning("Aún no hay modelo de forecasting demo. Corré `forecast_train.py` en Actions y commiteá los artefactos.")
        else:
            res = fobj["model"]
            hist_index = fobj.get("endog_index", res.model.data.row_labels)
            hist = pd.Series(res.model.endog, index=hist_index, name="y")

            steps = st.number_input("Horizonte (pasos)", 1, 60, 12)
            yhat = res.get_forecast(int(steps)).predicted_mean.values
            out = forecast_to_df(hist, yhat, int(steps))

            st.write("**Meta del modelo:**", meta)

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

    # --- B) SUBIR SERIE Y ENTRENAR AL VUELO ---
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
    import plotly.graph_objects as go
    from datetime import date, timedelta

    st.caption("Cargá una serie corta a mano para probar el pronóstico al vuelo.")

    # --- 1) Editor de tabla
    st.markdown("#### Opción A: Editor de tabla")
    # plantilla con 6 puntos diarios como ejemplo
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

    st.markdown("#### Opción B: Pegar texto (CSV rápido)")
    st.caption("Formato: 2 columnas `ds,y` separadas por coma. Ejemplo:\n\n```\n2024-01-01,100\n2024-01-02,101.5\n```\n")
    txt = st.text_area("Pegar aquí", value="", height=140, key="ts_manual_text")

    steps3 = st.number_input("Horizonte a pronosticar", 1, 120, 12, key="steps_manual")
    run_manual = st.button("🔮 Entrenar & pronosticar (manual)")

    if run_manual:
        try:
            # 1) tomar la fuente: texto si no está vacío; si no, el editor
            if txt.strip():
                df_in = pd.read_csv(pd.compat.StringIO(txt.strip()), header=None, names=["ds", "y"])
            else:
                df_in = df_edit.copy()

            # 2) limpieza y chequeos básicos
            if df_in.shape[1] < 2:
                raise ValueError("Se necesitan dos columnas: fecha y valor (ds,y).")
            df_in = df_in.rename(columns={df_in.columns[0]: "ds", df_in.columns[1]: "y"})
            df_in["ds"] = pd.to_datetime(df_in["ds"], errors="coerce")
            df_in["y"] = pd.to_numeric(df_in["y"], errors="coerce")
            df_in = df_in.dropna().sort_values("ds")
            if len(df_in) < 6:
                raise ValueError("Ingresá al menos 6 observaciones para entrenar.")

            # 3) normalizar a frecuencia fija y completar huecos
            from forecast_utils import parse_ts, fit_quick_sarimax, forecast_to_df
            s, freq = parse_ts(df_in)

            with st.spinner(f"Entrenando SARIMAX (freq detectada: {freq})…"):
                res3, order, seas, aic = fit_quick_sarimax(s)
                yhat3 = res3.get_forecast(int(steps3)).predicted_mean.values

            out3 = forecast_to_df(s, yhat3, int(steps3))

            # 4) gráfico
            fig3 = go.Figure()
            fig3.add_scatter(x=s.index, y=s.values, mode="lines", name="histórico")
            fig3.add_scatter(x=out3["ds"], y=out3["yhat"], mode="lines", name="pronóstico")
            fig3.update_layout(
                title=f"Pronóstico manual (orden={order}, seasonal={seas}, AIC={aic:.1f})",
                xaxis_title="Fecha", yaxis_title="Valor"
            )
            st.plotly_chart(fig3, use_container_width=True)

            # 5) descarga
            st.download_button(
                "⬇️ Descargar pronóstico (CSV)",
                out3.to_csv(index=False).encode("utf-8"),
                "forecast_manual.csv",
                "text/csv"
            )
        except Exception as e:
            st.error(f"No se pudo entrenar/forecastear con los datos ingresados: {e}")

