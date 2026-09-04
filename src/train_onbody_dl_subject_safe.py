"""Fix review-round-2 issue: the early-stopping validation split for the on-body benchmark
was drawn at the window level, which can place windows from the same subject in both the
training and validation folds, contaminating the epoch-selection signal even though the
final test set remained subject-disjoint. This carves the validation set at the SUBJECT
level instead, from within the 18 training subjects, so validation is fully subject-safe.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dl_models import build_model

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEVICE = torch.device("cpu")
MAX_EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 256
FEATURE_COLS = ["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean", "motion_energy", "tx_rate", "packet_loss_rate"]


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
        X_test_t = torch.tensor(X_test)
        preds = [model(X_test_t[i:i + BATCH_SIZE]).argmax(dim=1) for i in range(0, len(X_test_t), BATCH_SIZE)]
        y_pred = torch.cat(preds).numpy()
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    return macro_f1, report, cm


def main():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_subjects = sorted(train_df["subject"].unique(), key=lambda s: int(s[1:]))
    rng = np.random.default_rng(42)
    n_val_subjects = max(1, round(len(train_subjects) * 0.2))
    val_subjects = set(rng.choice(train_subjects, size=n_val_subjects, replace=False))
    print(f"Validation subjects (held out from the 18 training subjects, subject-safe): {sorted(val_subjects)}")

    fit_df = train_df[~train_df["subject"].isin(val_subjects)]
    val_df = train_df[train_df["subject"].isin(val_subjects)]
    print(f"Fit windows: {len(fit_df)}, validation windows: {len(val_df)}, test windows: {len(test_df)}")

    le = LabelEncoder()
    y_fit = le.fit_transform(fit_df["label"])
    y_val = le.transform(val_df["label"])
    y_test = le.transform(test_df["label"])
    n_classes = len(le.classes_)

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(fit_df[FEATURE_COLS]).astype(np.float32)
    X_val = scaler.transform(val_df[FEATURE_COLS]).astype(np.float32)
    X_test = scaler.transform(test_df[FEATURE_COLS]).astype(np.float32)

    results = {}
    for model_name in ("CNN", "LSTM", "BiGRU_Attention", "Transformer"):
        model, best_epoch = train_with_early_stopping(model_name, X_fit, y_fit, X_val, y_val, n_classes)
        macro_f1, report, cm = evaluate(model, X_test, y_test, le.classes_.astype(str))
        print(f"{model_name}: macro-F1={macro_f1:.4f} (stopped at epoch {best_epoch}, subject-safe validation)")
        results[model_name] = {"macro_f1": macro_f1, "report": report, "confusion_matrix": cm,
                                "class_order": le.classes_.tolist(), "stopped_epoch": best_epoch}

    out_path = RESULTS_DIR / "onbody_dl_subject_safe_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
