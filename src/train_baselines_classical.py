"""Train classical ML baselines (Random Forest, XGBoost) on each data source independently.

This is the first pass of the C1 "benchmarking study" (detection-performance dimension).
Deep learning baselines (CNN/LSTM/BiGRU-attention/transformer) will run separately, likely on Kaggle.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLS = {"label", "category", "split", "source_file", "protocol", "Attack Category", "Label"}


def prep_xy(df: pd.DataFrame, label_col: str):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS
                    and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[label_col]
    return X, y, feature_cols


def run_dataset(name: str, X_train, y_train, X_test, y_test, results: dict):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42,
                                  eval_metric="mlogloss"),
    }

    results[name] = {}
    for model_name, model in models.items():
        print(f"\n[{name}] Training {model_name} ...")
        model.fit(X_train, y_train_enc)
        y_pred = model.predict(X_test)
        report = classification_report(y_test_enc, y_pred, target_names=le.classes_.astype(str),
                                        output_dict=True, zero_division=0)
        macro_f1 = f1_score(y_test_enc, y_pred, average="macro")
        print(f"[{name}] {model_name} macro-F1: {macro_f1:.4f}")
        results[name][model_name] = {"macro_f1": macro_f1, "report": report}


def main():
    results = {}

    # --- CICIoMT2024 WiFi/MQTT (coarse category, already train/test split) ---
    df = pd.read_parquet(DATA_DIR / "ciciomt2024_wifi_mqtt_consolidated.parquet")
    # subsample majority classes for tractability on a CPU-only machine; keep full minority classes
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]
    X_train, y_train, feats = prep_xy(df_train, "category")
    X_test, y_test, _ = prep_xy(df_test, "category")
    # cap huge classes to keep local CPU training tractable
    cap = 50000
    if len(X_train) > cap * 6:
        idx = df_train.groupby("category").apply(lambda g: g.sample(min(len(g), cap), random_state=42)).index.get_level_values(1)
        X_train_capped, y_train_capped = X_train.loc[idx], y_train.loc[idx]
    else:
        X_train_capped, y_train_capped = X_train, y_train
    run_dataset("CICIoMT2024_WiFi_MQTT_coarse_CAPPED", X_train_capped, y_train_capped, X_test, y_test, results)
    # full-data run for a fair comparison against the Kaggle DL results (also full-data)
    run_dataset("CICIoMT2024_WiFi_MQTT_coarse_FULL", X_train, y_train, X_test, y_test, results)

    # --- CICIoMT2024 Bluetooth ---
    df_bt = pd.read_parquet(DATA_DIR / "ciciomt2024_bluetooth_consolidated.parquet")
    df_bt_train = df_bt[df_bt["split"] == "train"]
    df_bt_test = df_bt[df_bt["split"] == "test"]
    X_train, y_train, _ = prep_xy(df_bt_train, "label")
    X_test, y_test, _ = prep_xy(df_bt_test, "label")
    run_dataset("CICIoMT2024_Bluetooth", X_train, y_train, X_test, y_test, results)

    # --- WUSTL-EHMS-2020 (no pre-existing split -> stratified 80/20) ---
    df_w = pd.read_parquet(DATA_DIR / "wustl_ehms_2020.parquet")
    X, y, _ = prep_xy(df_w, "Attack Category")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    run_dataset("WUSTL_EHMS_2020", X_train, y_train, X_test, y_test, results)

    out_path = RESULTS_DIR / "baseline_classical_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved all results to {out_path}")

    print("\n=== SUMMARY (macro-F1) ===")
    for dataset, models in results.items():
        for model_name, r in models.items():
            print(f"  {dataset:35s} {model_name:15s} {r['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
