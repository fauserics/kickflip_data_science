# forecast_utils.py
# Utils para forecasting con statsmodels (SARIMAX) – sin pmdarima
import json
from pathlib import Path
from typing import Tuple, Optional, Dict

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

MODELS_DIR = Path("models")
F_MODEL = MODELS_DIR / "forecast_model.joblib"
F_META  = MODELS_DIR / "forecast_metadata.json"

def load_demo_forecast_model() -> Optional[dict]:
    """Carga el dict {'model': SARIMAXResults, 'endog_index': DatetimeIndex} guardado por forecast_train.py."""
    if F_MODEL.exists():
        return joblib.load(F_MODEL)
    return None

def load_demo_forecast_meta() -> Dict:
    if F_META.exists():
        return json.loads(F_META.read_text(encoding="utf-8"))
    return {}

def _infer_freq(idx: pd.DatetimeIndex) -> str:
    try:
        f = pd.infer_freq(idx)
        if f:
            return f
    except Exception:
        pass
    # Heurística sencilla: si paso mediano >= 28 días → mensual; si no → diario
    delta_days = idx.to_series().diff().median().days
    return "MS" if (pd.notna(delta_days) and delta_days >= 28) else "D"

def parse_ts(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """
    Acepta columnas: (ds,y) o (date,value) o toma las 2 primeras columnas.
    Devuelve serie con índice de fecha, frecuencia fija y sin huecos.
    """
    cols = {c.lower(): c for c in df.columns}
    ds = cols.get("ds") or cols.get("date") or df.columns[0]
    y  = cols.get("y")  or cols.get("value") or df.columns[1]
    s = pd.Series(df[y].values, index=pd.to_datetime(df[ds]), name="y").sort_index()
    freq = _infer_freq(s.index)
    s = s.asfreq(freq)
    s = s.interpolate(limit_direction="both")
    return s, freq

def _season_length(freq: str) -> int:
    f = (freq or "").upper()
    if "MS" in f or f == "M": return 12
    if f.startswith("W"):     return 52
    if f == "D":              return 7
    return 1

def fit_quick_sarimax(y: pd.Series):
    """
    Mini-búsqueda por AIC en (p,d,q)x(P,D,Q,m) con valores {0,1}.
    Rápida y suficiente para entrenar 'al vuelo' en la UI.
    """
    m = _season_length(y.index.freqstr or "D")
    p = d = q = range(0, 2)   # 0-1
    P = D = Q = range(0, 2)   # 0-1
    best = {"aic": np.inf, "res": None, "order": None, "seasonal": None}
    for order in [(i,j,k) for i in p for j in d for k in q]:
        for seas in [(i,j,k) for i in P for j in D for k in Q]:
            try:
                mod = sm.tsa.statespace.SARIMAX(
                    y, order=order, seasonal_order=(*seas, m),
                    enforce_stationarity=False, enforce_invertibility=False
                )
                res = mod.fit(disp=False)
                if res.aic < best["aic"]:
                    best.update(aic=float(res.aic), res=res, order=order, seasonal=(*seas, m))
            except Exception:
                continue
    if best["res"] is None:
        # Fallback
        mod = sm.tsa.statespace.SARIMAX(
            y, order=(1,1,1), seasonal_order=(1,1,1,m),
            enforce_stationarity=False, enforce_invertibility=False
        )
        best_res = mod.fit(disp=False)
        best = {"aic": float(best_res.aic), "res": best_res, "order": (1,1,1), "seasonal": (1,1,1,m)}
    return best["res"], best["order"], best["seasonal"], best["aic"]

def forecast_to_df(history: pd.Series, yhat: np.ndarray, steps: int) -> pd.DataFrame:
    idx = pd.date_range(start=history.index[-1], periods=steps+1, freq=history.index.freq)[1:]
    return pd.DataFrame({"ds": idx, "yhat": yhat})
