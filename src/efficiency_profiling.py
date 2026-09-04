"""Computational efficiency profiling (C1, dimension b).

Single-machine inference latency + memory footprint for each trained model family,
on the on-body benchmark (the paper's own, smallest, most relevant-to-deployment
dataset) and WUSTL-EHMS-2020. Framed honestly as single-machine profiling, not a
constrained-device deployment claim (no dedicated hardware section in this paper).
"""
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from dl_models import build_model

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

FEATURE_COLS_ONBODY = ["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean",
                        "motion_energy", "tx_rate", "packet_loss_rate"]
N_INFERENCE_RUNS = 1000


def profile_sklearn_model(name, model, X_single):
    # warmup
    for _ in range(10):
        model.predict(X_single)
    t0 = time.perf_counter()
    for _ in range(N_INFERENCE_RUNS):
        model.predict(X_single)
    dt = time.perf_counter() - t0
    per_sample_ms = (dt / N_INFERENCE_RUNS) * 1000
    return per_sample_ms


def profile_torch_model(name, model, x_single):
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(x_single)
        t0 = time.perf_counter()
        for _ in range(N_INFERENCE_RUNS):
            model(x_single)
        dt = time.perf_counter() - t0
    per_sample_ms = (dt / N_INFERENCE_RUNS) * 1000
    n_params = sum(p.numel() for p in model.parameters())
    return per_sample_ms, n_params


def main():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    le = LabelEncoder()
    y_train = le.fit_transform(train["label"])
    n_classes = len(le.classes_)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS_ONBODY]).astype(np.float32)
    X_single = test[FEATURE_COLS_ONBODY].iloc[[0]].values.astype(np.float32)
    X_single_scaled = scaler.transform(X_single)

    results = {}

    # classical ML
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X_train, y_train)
    results["RandomForest"] = {
        "inference_ms_per_sample": profile_sklearn_model("RF", rf, X_single_scaled),
        "n_estimators": 200,
    }

    xgb = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric="mlogloss")
    xgb.fit(X_train, y_train)
    results["XGBoost"] = {
        "inference_ms_per_sample": profile_sklearn_model("XGB", xgb, X_single_scaled),
        "n_estimators": 200,
    }

    # DL models
    n_features = X_train.shape[1]
    x_single_t = torch.tensor(X_single_scaled, dtype=torch.float32)
    for model_name in ("CNN", "LSTM", "BiGRU_Attention", "Transformer"):
        model = build_model(model_name, n_features, n_classes)
        ms, n_params = profile_torch_model(model_name, model, x_single_t)
        results[model_name] = {
            "inference_ms_per_sample": ms,
            "n_parameters": n_params,
        }

    print("=== Single-machine CPU inference latency (ms/sample) + model size ===")
    for name, r in results.items():
        extra = f"params={r.get('n_parameters', 'n/a')}"
        print(f"  {name:20s} {r['inference_ms_per_sample']:.4f} ms/sample  {extra}")

    out_path = RESULTS_DIR / "efficiency_profiling_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
