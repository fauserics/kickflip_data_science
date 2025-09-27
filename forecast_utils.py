# forecast_utils.py
import io
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from pathlib import Path

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

MODELS_DIR = Path("models")
F_MODEL = MODELS_DIR / "forecast_model.joblib"
F_META  = MODELS_DIR / "forecast_metadata.json"

def load_demo_forecast_model() -> Optional[SARIMAXResults]:
    if F_MODEL.exists():
        return joblib.load(F_MODEL)
    return None

def load_demo_forecast_meta() -> Dict:
    if F_META.exists():
        return json.loads(F_META.read_text(encoding="utf-8"))
    return {}

def _parse_ts(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Acepta columnas 'ds'/'y' o 'date'/'value' o inferencia por 1ra y 2da col."""
    cols = [c.lower() for c in df.columns]
    m = dict(zip(cols, df.columns))
    if "ds" in m and "y" in m:
        ds, y = m["ds"], m["y"]
    elif "date" in m and "value" in m:
        ds, y = m["date"], m["value"]
    else:
        ds, y = df.columns[0], df.columns[1]
    s = pd.Series(df[y].values, index=pd.to_datetime(df[ds]), name="y").sort_index()
    # inferir frecuencia
    try:
        freq = pd.infer_freq(s.index)
    except Exception:
        freq = None
    if freq is None:
        # fallback: mensual si hay ~12 por año
        freq = "MS" if s.index.to_series().diff().median().days >= 28 else "D"
        s = s.asfreq(freq)
    else:
        s = s.asfreq(freq)
    # rellenar huecos
    s = s.interpolate(limit_direction="both")
    return s, freq

def fit_autoarima(s: pd.Series):
    """Auto-ARIMA rápido, robusto para demo/tiempo real."""
    model = auto_arima(
        s, seasonal=True, m=_season_length(s.index.freqstr),
        error_action="ignore", suppress_warnings=True, stepwise=True,
        max_p=3, max_q=3, max_P=2, max_Q=2, max_order=None
    )
    return model

def _season_length(freq: Optional[str]) -> int:
    if not freq:
        return 1
    f = freq.upper()
    if "MS" in f or f == "M": return 12
    if f.startswith("W"): return 52
    if f == "D": return 7
    return 1

def forecast_to_df(history: pd.Series, yhat: np.ndarray, steps: int) -> pd.DataFrame:
    idx = pd.date_range(start=history.index[-1], periods=steps+1, freq=history.index.freq)[1:]
    out = pd.DataFrame({"ds": idx, "yhat": yhat})
    return out
