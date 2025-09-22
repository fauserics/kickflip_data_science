# Trainer & Model Selection using public OpenML 'adult' dataset (binary classification).
# - Downloads data (first run) via sklearn's fetch_openml.
# - Builds preprocessing (numeric standardization + categorical one-hot).
# - Compares baseline models via cross-validated ROC AUC.
# - Selects best, fits on full train, evaluates on hold-out test.
# - Saves best_model.joblib, metadata.json and input_schema.json under ./models.

import json
import os
import warnings
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.datasets import fetch_openml

from utils import save_model, save_metadata, infer_schema_from_dataframe, save_schema

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "class"  # for adult dataset
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_public_dataset() -> pd.DataFrame:
    # OpenML "adult" (a.k.a. Census Income) id=1590, ~48k rows
    data = fetch_openml(data_id=1590, as_frame=True)  # requires internet
    df = data.frame
    # Harmonize target name
    if "income" in df.columns and TARGET not in df.columns:
        df.rename(columns={"income": TARGET}, inplace=True)
    return df

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    X = df.drop(columns=[TARGET])
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),  # sparse-safe
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor

def candidates(random_state=RANDOM_STATE) -> Dict[str, Any]:
    return {
        "LogReg": LogisticRegression(max_iter=1000, n_jobs=None),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=random_state, n_jobs=-1, tree_method="hist"
        )
    }

def evaluate_models(X: pd.DataFrame, y: pd.Series, preprocessor) -> List[Dict[str, Any]]:
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, clf in candidates().items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        results.append({
            "model": name,
            "roc_auc": float(np.mean(scores)),
            "roc_auc_std": float(np.std(scores)),
            "cv_folds": len(scores)
        })
    return results

def fit_best(X_train, y_train, preprocessor, best_name: str):
    clf = candidates()[best_name]
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_public_dataset()
    df.dropna(how="all", axis=1, inplace=True)  # safety
    # Ensure binary target encoded as 0/1
    df[TARGET] = df[TARGET].astype(str).str.contains(">50K").astype(int)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(df)
    metrics_list = evaluate_models(X_train, y_train, preprocessor)
    metrics_list = sorted(metrics_list, key=lambda d: d["roc_auc"], reverse=True)
    best_name = metrics_list[0]["model"]

    best_pipe = fit_best(X_train, y_train, preprocessor, best_name)
    # Hold-out evaluation
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    holdout = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "threshold": 0.5,
    }

    # Persist artifacts
    save_model(best_pipe)

    schema = infer_schema_from_dataframe(df, target=TARGET)
    save_schema(schema)

    metadata = {
        "problem_type": "binary_classification",
        "dataset": "OpenML adult (Census Income, data_id=1590)",
        "target": TARGET,
        "rows": int(df.shape[0]),
        "best_model": best_name,
        "cv": metrics_list,
        "holdout": holdout,
        "random_state": RANDOM_STATE,
    }
    save_metadata(metadata)

    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()