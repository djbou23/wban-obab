"""Batched-throughput efficiency measurement (review issue 11).

The original profiling measured 1000 sequential single-sample .predict() calls, which for
sklearn/xgboost models is dominated by per-call Python dispatch overhead rather than the
model's actual inference cost. This adds a batched-throughput measurement (batch size 1000,
one call) alongside the original single-sample latency, for a fairer picture of ensemble
methods specifically.
"""
import json
import time
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
BATCH = 1000
N_REPEATS = 20


def main():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    le = LabelEncoder()
    y_train = le.fit_transform(train["label"])
    n_classes = len(le.classes_)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS_ONBODY]).astype(np.float32)
    X_batch = scaler.transform(test[FEATURE_COLS_ONBODY].sample(BATCH, replace=True, random_state=42)).astype(np.float32)

    results = {}

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42).fit(X_train, y_train)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric="mlogloss").fit(X_train, y_train)

    for name, model in [("RandomForest", rf), ("XGBoost", xgb)]:
        for _ in range(3):
            model.predict(X_batch)
        t0 = time.perf_counter()
        for _ in range(N_REPEATS):
            model.predict(X_batch)
        dt = time.perf_counter() - t0
        per_sample_ms = (dt / N_REPEATS / BATCH) * 1000
        results[name] = {"batched_ms_per_sample": per_sample_ms, "batch_size": BATCH}
        print(f"{name}: batched throughput = {per_sample_ms:.5f} ms/sample (batch={BATCH})")

    n_features = X_train.shape[1]
    for model_name in ("CNN", "LSTM", "BiGRU_Attention", "Transformer"):
        model = build_model(model_name, n_features, n_classes)
        model.eval()
        x_batch_t = torch.tensor(X_batch)
        with torch.no_grad():
            for _ in range(3):
                model(x_batch_t)
            t0 = time.perf_counter()
            for _ in range(N_REPEATS):
                model(x_batch_t)
            dt = time.perf_counter() - t0
        per_sample_ms = (dt / N_REPEATS / BATCH) * 1000
        results[model_name] = {"batched_ms_per_sample": per_sample_ms, "batch_size": BATCH}
        print(f"{model_name}: batched throughput = {per_sample_ms:.5f} ms/sample (batch={BATCH})")

    out_path = RESULTS_DIR / "efficiency_batched_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
