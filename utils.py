import json
import os
import joblib
import pandas as pd
from typing import Dict, Any, List

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
META_PATH = os.path.join(MODELS_DIR, "metadata.json")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
SCHEMA_PATH = os.path.join(MODELS_DIR, "input_schema.json")

def load_metadata() -> Dict[str, Any]:
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_schema() -> Dict[str, Any]:
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_schema(schema: Dict[str, Any]) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def save_model(model) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def save_metadata(meta: Dict[str, Any]) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def validate_dataframe(df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
    expected_cols = schema.get("columns", [])
    df = df.copy()
    for col_def in expected_cols:
        name = col_def["name"]
        if name not in df.columns:
            df[name] = pd.NA
    return df[ [c["name"] for c in expected_cols] ]

def infer_schema_from_dataframe(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    schema = {"columns": [], "target": target}
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = "numeric"
            values = None
        else:
            col_type = "categorical"
            values = sorted([str(v) for v in df[col].dropna().unique()])[:20]
        schema["columns"].append({"name": col, "type": col_type, "values": values})
    return schema

def pretty_metric_table(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(metrics_list).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)