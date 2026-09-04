"""DL baselines with validation-based early stopping (review issue 12), extended to the
on-body benchmark (review issue 3, which had classical ML only in the first pass).

Also produces a confusion matrix for the on-body benchmark's best model (review issue 13).
The large-scale CICIoMT2024 WiFi/MQTT Kaggle run is NOT redone here due to GPU compute cost;
this is disclosed explicitly in the paper text as a fixed-epoch-budget exception.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dl_models import build_model

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEVICE = torch.device("cpu")
MAX_EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 256
NON_FEATURE_COLS = {"label", "category", "split", "source_file", "protocol", "Attack Category", "Label", "subject", "activity"}


def prep_xy(df, label_col):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    y = df[label_col].values
    return X, y, feature_cols


def train_with_early_stopping(model_name, X_train, y_train, X_val, y_val, n_classes):
    n_features = X_train.shape[1]
    model = build_model(model_name, n_features, n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_train_t, y_train_t = torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)
    X_val_t, y_val_t = torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long)

    best_val_loss, best_state, epochs_no_improve, best_epoch = float("inf"), None, 0, 0
    n = len(X_train_t)
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        if val_loss < best_val_loss - 1e-4:
            best_val_loss, best_state, epochs_no_improve, best_epoch = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0, epoch
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_epoch + 1


def evaluate(model, X_test, y_test, class_names):
    model.eval()
    with torch.no_grad():
        preds = []
        X_test_t = torch.tensor(X_test)
        for i in range(0, len(X_test_t), BATCH_SIZE):
            preds.append(model(X_test_t[i:i + BATCH_SIZE]).argmax(dim=1))
        y_pred = torch.cat(preds).numpy()
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    return macro_f1, report, cm, y_pred


def run_dataset(name, X, y, results, fixed_split=None):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    if fixed_split is not None:
        train_idx, test_idx = fixed_split
        X_train_full, y_train_full = X[train_idx], y_enc[train_idx]
        X_test, y_test = X[test_idx], y_enc[test_idx]
    else:
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

    X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, stratify=y_train_full, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    results[name] = {}
    for model_name in ("CNN", "LSTM", "BiGRU_Attention", "Transformer"):
        model, best_epoch = train_with_early_stopping(model_name, X_tr, y_tr, X_val, y_val, n_classes)
        macro_f1, report, cm, y_pred = evaluate(model, X_test, y_test, le.classes_.astype(str))
        print(f"[{name}] {model_name}: macro-F1={macro_f1:.4f} (stopped at epoch {best_epoch})")
        results[name][model_name] = {"macro_f1": macro_f1, "report": report, "confusion_matrix": cm,
                                      "class_order": le.classes_.tolist(), "stopped_epoch": best_epoch}


def main():
    results = {}

    df_ob = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    X, y, _ = prep_xy(df_ob, "label")
    train_mask = (df_ob["split"] == "train").values
    test_mask = (df_ob["split"] == "test").values
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    run_dataset("OnBody_benchmark", X, y, results, fixed_split=(train_idx, test_idx))

    df_bt = pd.read_parquet(DATA_DIR / "ciciomt2024_bluetooth_consolidated.parquet")
    X, y, _ = prep_xy(df_bt, "label")
    run_dataset("CICIoMT2024_Bluetooth", X, y, results)

    df_w = pd.read_parquet(DATA_DIR / "wustl_ehms_2020.parquet")
    df_w.columns = [c.strip() for c in df_w.columns]
    X, y, _ = prep_xy(df_w, "Attack Category")
    run_dataset("WUSTL_EHMS_2020", X, y, results)

    out_path = RESULTS_DIR / "baseline_dl_results_v2_early_stopping.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    print("\n=== SUMMARY ===")
    for dataset, models in results.items():
        for model_name, r in models.items():
            print(f"  {dataset:25s} {model_name:20s} {r['macro_f1']:.4f} (epoch {r['stopped_epoch']})")


if __name__ == "__main__":
    main()
