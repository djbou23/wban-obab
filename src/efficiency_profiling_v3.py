"""Fix review-round-2 issue: Fig. 5 previously paired CICIoMT2024 WiFi/MQTT accuracy (45
features) with latency measured on the on-body benchmark (8 features). Inference cost
scales with input dimensionality, so this re-measures both single-sample and batched
latency using 45-dimensional input drawn from the actual WiFi/MQTT feature space, keeping
accuracy and efficiency numbers consistent with the same dataset.
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
NON_FEATURE_COLS = {"label", "category", "split", "source_file", "protocol"}
BATCH = 1000
N_REPEATS_BATCHED = 20
N_REPEATS_SINGLE = 1000


def main():
    df = pd.read_parquet(DATA_DIR / "ciciomt2024_wifi_mqtt_consolidated.parquet")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    train_df = df[df["split"] == "train"]
    # cap for tractable local timing-only training (timing depends on architecture and
    # input width, not on how many rows the model was fit on)
    cap = 20000
    idx = train_df.groupby("category").apply(lambda g: g.sample(min(len(g), cap), random_state=42)).index.get_level_values(1)
    train_capped = train_df.loc[idx]

    le = LabelEncoder()
    y_train = le.fit_transform(train_capped["category"])
    n_classes = len(le.classes_)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_capped[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)).astype(np.float32)

    test_df = df[df["split"] == "test"].sample(BATCH, random_state=42, replace=True)
    X_batch = scaler.transform(test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)).astype(np.float32)
    X_single = X_batch[:1]

    results = {}

    # n_jobs left at default (1) to match the original on-body profiling methodology and to
    # avoid parallel-dispatch overhead contaminating the single-sample timing loop itself
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42).fit(X_train, y_train)
    xgb = XGBClassifier(n_estimators=200, max_depth=8, random_state=42, eval_metric="mlogloss").fit(X_train, y_train)

    for name, model in [("RandomForest", rf), ("XGBoost", xgb)]:
        for _ in range(10):
            model.predict(X_single)
        t0 = time.perf_counter()
        for _ in range(N_REPEATS_SINGLE):
            model.predict(X_single)
        single_ms = (time.perf_counter() - t0) / N_REPEATS_SINGLE * 1000

        for _ in range(3):
            model.predict(X_batch)
        t0 = time.perf_counter()
        for _ in range(N_REPEATS_BATCHED):
            model.predict(X_batch)
        batched_ms = (time.perf_counter() - t0) / N_REPEATS_BATCHED / BATCH * 1000

        results[name] = {"single_ms": single_ms, "batched_ms": batched_ms}
        print(f"{name}: single={single_ms:.4f} ms, batched={batched_ms:.5f} ms/sample")

    n_features = X_train.shape[1]
    x_single_t = torch.tensor(X_single)
    x_batch_t = torch.tensor(X_batch)
    for model_name in ("CNN", "LSTM", "BiGRU_Attention", "Transformer"):
        model = build_model(model_name, n_features, n_classes)
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                model(x_single_t)
            t0 = time.perf_counter()
            for _ in range(N_REPEATS_SINGLE):
                model(x_single_t)
            single_ms = (time.perf_counter() - t0) / N_REPEATS_SINGLE * 1000

            for _ in range(3):
                model(x_batch_t)
            t0 = time.perf_counter()
            for _ in range(N_REPEATS_BATCHED):
                model(x_batch_t)
            batched_ms = (time.perf_counter() - t0) / N_REPEATS_BATCHED / BATCH * 1000

        n_params = sum(p.numel() for p in model.parameters())
        results[model_name] = {"single_ms": single_ms, "batched_ms": batched_ms, "n_parameters": n_params}
        print(f"{model_name}: single={single_ms:.4f} ms, batched={batched_ms:.5f} ms/sample, params={n_params}")

    out_path = RESULTS_DIR / "efficiency_wifimqtt_45feat_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
