"""Train DL baselines (CNN, LSTM, BiGRU+Attention, Transformer) on the smaller datasets locally.

The large CICIoMT2024 WiFi/MQTT set (8.7M rows) is handled separately on Kaggle GPU
(see kaggle_train_wifi_mqtt.py) since CPU training on it would be impractically slow.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dl_models import build_model

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLS = {"label", "category", "split", "source_file", "protocol", "Attack Category", "Label"}
DEVICE = torch.device("cpu")
EPOCHS = 15
BATCH_SIZE = 256


def prep_xy(df: pd.DataFrame, label_col: str):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS
                    and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    y = df[label_col].values
    return X, y, feature_cols


def train_one_model(model_name, X_train, y_train, X_test, y_test, n_classes, class_names):
    n_features = X_train.shape[1]
    model = build_model(model_name, n_features, n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test)

    n = len(X_train_t)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if epoch == EPOCHS - 1 or epoch % 5 == 0:
            print(f"    epoch {epoch+1}/{EPOCHS} loss={total_loss/n:.4f}")

    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_test_t), BATCH_SIZE):
            out = model(X_test_t[i:i + BATCH_SIZE])
            preds.append(out.argmax(dim=1))
        y_pred = torch.cat(preds).numpy()

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    return macro_f1, report


def run_dataset(name, X, y, results):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    results[name] = {}
    for model_name in ("CNN", "LSTM", "BiGRU_Attention", "Transformer"):
        print(f"\n[{name}] Training {model_name} ...")
        t0 = time.time()
        macro_f1, report = train_one_model(model_name, X_train, y_train, X_test, y_test,
                                            n_classes, le.classes_.astype(str))
        dt = time.time() - t0
        print(f"[{name}] {model_name} macro-F1: {macro_f1:.4f} ({dt:.1f}s)")
        results[name][model_name] = {"macro_f1": macro_f1, "report": report, "train_seconds": dt}


def main():
    results = {}

    df_bt = pd.read_parquet(DATA_DIR / "ciciomt2024_bluetooth_consolidated.parquet")
    X, y, _ = prep_xy(df_bt, "label")
    run_dataset("CICIoMT2024_Bluetooth", X, y, results)

    df_w = pd.read_parquet(DATA_DIR / "wustl_ehms_2020.parquet")
    X, y, _ = prep_xy(df_w, "Attack Category")
    run_dataset("WUSTL_EHMS_2020", X, y, results)

    out_path = RESULTS_DIR / "baseline_dl_results_local.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    print("\n=== SUMMARY (macro-F1) ===")
    for dataset, models in results.items():
        for model_name, r in models.items():
            print(f"  {dataset:25s} {model_name:20s} {r['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
