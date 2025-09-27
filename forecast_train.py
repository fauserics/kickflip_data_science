# forecast_train.py
import json
from pathlib import Path

import joblib
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODELS_DIR = Path("models")
F_MODEL = MODELS_DIR / "forecast_model.joblib"
F_META  = MODELS_DIR / "forecast_metadata.json"

def load_airpassengers() -> pd.Series:
    data = sm.datasets.get_rdataset("AirPassengers").data  # monthly 1949-1960
    # renombrar correctamente
    s = pd.Series(data["value"].values,
                  index=pd.period_range("1949-01", periods=len(data), freq="M").to_timestamp(),
                  name="y")
    s = s.asfreq("MS")
    return s

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    s = load_airpassengers()
    # SARIMA clásico (p,d,q)(P,D,Q)m para mensual
    model = SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    joblib.dump(res, F_MODEL)

    meta = {
        "dataset": "AirPassengers (monthly)",
        "freq": "MS",
        "train_start": str(s.index[0].date()),
        "train_end": str(s.index[-1].date()),
        "order": [1,1,1],
        "seasonal_order": [1,1,1,12],
    }
    F_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
