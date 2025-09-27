# forecast_train.py
# Entrena un modelo SARIMAX mensual (statsmodels) y guarda artefactos en ./models
# - Usa data/ts_train.csv si existe (formato: ds,y o date,value)
# - Si no, intenta cargar AirPassengers vía get_rdataset
# - Si tampoco, genera una serie mensual sintética como fallback

import os
import json
import itertools
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

MODELS_DIR = Path("models")
OUT_MODEL = MODELS_DIR / "forecast_model.joblib"
OUT_META  = MODELS_DIR / "forecast_metadata.json"
DATA_TS   = Path("data/ts_train.csv")

def load_series() -> pd.Series:
    """Carga serie mensual (MS). Si hay gaps, interpola."""
    # 1) Datos propios en repo
    if DATA_TS.exists():
        df = pd.read_csv(DATA_TS)
        cols = {c.lower(): c for c in df.columns}
        ds = cols.get("ds") or cols.get("date") or df.columns[0]
        y  = cols.get("y")  or cols.get("value") or df.columns[1]
        s = pd.Series(df[y].values, index=pd.to_datetime(df[ds]), name="y").sort_index()
        return s.asfreq("MS").interpolate(limit_direction="both")

    # 2) AirPassengers desde R datasets (requiere internet en Actions)
    try:
        dta = sm.datasets.get_rdataset("AirPassengers", "datasets").data
        # La columna de valores suele llamarse 'value'
        s = pd.Series(
            dta["value"].values,
            index=pd.period_range("1949-01", periods=len(dta), freq="M").to_timestamp(),
            name="y",
        ).asfreq("MS")
        return s
    except Exception:
        pass

    # 3) Fallback sintético (12 años mensuales con tendencia + estacionalidad)
    rng = pd.period_range("2015-01", periods=12 * 10, freq="M").to_timestamp()
    t = np.arange(len(rng))
    season = 10 * np.sin(2 * np.pi * (t % 12) / 12.0)
    trend = 0.7 * t
    noise = np.random.normal(0, 2, size=len(rng))
    s = pd.Series(100 + trend + season + noise, index=rng, name="y").round(2)
    return s.asfreq("MS")

def fit_sarimax(y: pd.Series):
    """
    Búsqueda chica por AIC en (p,d,q)x(P,D,Q,12) con valores {0,1}.
    Rápida y suficiente para demo.
    """
    y = y.asfreq("MS").interpolate(limit_direction="both")
    p = d = q = range(0, 2)   # 0-1
    P = D = Q = range(0, 2)   # 0-1
    seasonal_period = 12

    best_aic = np.inf
    best_order = None
    best_seasonal = None
    best_res = None

    for order in itertools.product(p, d, q):
        for seas in itertools.product(P, D, Q):
            try:
                mod = sm.tsa.statespace.SARIMAX(
                    y,
                    order=order,
                    seasonal_order=(*seas, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = mod.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = order
                    best_seasonal = (*seas, seasonal_period)
                    best_res = res
            except Exception:
                continue

    if best_res is None:
        mod = sm.tsa.statespace.SARIMAX(
            y, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period),
            enforce_stationarity=False, enforce_invertibility=False,
        )
        best_res = mod.fit(disp=False)
        best_order = (1, 1, 1)
        best_seasonal = (1, 1, 1, seasonal_period)
        best_aic = float(best_res.aic)

    return best_res, best_order, best_seasonal, float(best_aic)

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    y = load_series()
    res, order, seas, aic = fit_sarimax(y)

    # Guardamos el modelo junto con el índice original (para graficar en la app)
    joblib.dump({"model": res, "endog_index": y.index}, OUT_MODEL)

    meta = {
        "framework": "statsmodels",
        "algorithm": "SARIMAX",
        "order": list(order),
        "seasonal_order": list(seas),
        "aic": aic,
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "freq": "MS",
        "observations": int(len(y)),
        "train_start": str(y.index[0].date()),
        "train_end": str(y.index[-1].date()),
        "source": "data/ts_train.csv" if DATA_TS.exists() else "AirPassengers or synthetic",
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
