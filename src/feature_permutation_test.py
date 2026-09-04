"""E4 (essential experiment E4): the DL baselines (dl_models.py) treat each of the 8
features as one timestep of a length-8 "sequence" with no natural temporal order
(heart rate -> PPG -> temperature -> motion is an arbitrary ordering). This retrains
each sequence model (CNN/LSTM/BiGRU-Attention/Transformer) with a fixed random
permutation of the feature order and compares macro-F1 against the reported (natural-
order) results in onbody_dl_subject_safe_results.json. If performance is materially
unchanged, the sequential-modeling assumption these architectures rely on is not
actually doing anything for this task -- exactly what reviewer Major Concern 11 asks
to be checked.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from train_onbody_dl_subject_safe import train_with_early_stopping, evaluate, FEATURE_COLS

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PERM_SEED = 999


def main():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_subjects = sorted(train_df["subject"].unique(), key=lambda s: int(s[1:]))
    rng = np.random.default_rng(42)
    n_val_subjects = max(1, round(len(train_subjects) * 0.2))
    val_subjects = set(rng.choice(train_subjects, size=n_val_subjects, replace=False))

    fit_df = train_df[~train_df["subject"].isin(val_subjects)]
    val_df = train_df[train_df["subject"].isin(val_subjects)]

    perm_rng = np.random.default_rng(PERM_SEED)
    permuted_cols = list(FEATURE_COLS)
    perm_rng.shuffle(permuted_cols)
    print(f"Natural order:  {FEATURE_COLS}")
    print(f"Permuted order: {permuted_cols}")

    le = LabelEncoder()
    y_fit = le.fit_transform(fit_df["label"])
    y_val = le.transform(val_df["label"])
    y_test = le.transform(test_df["label"])

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(fit_df[permuted_cols]).astype(np.float32)
    X_val = scaler.transform(val_df[permuted_cols]).astype(np.float32)
    X_test = scaler.transform(test_df[permuted_cols]).astype(np.float32)

    results = {"feature_order": permuted_cols, "natural_order": FEATURE_COLS}
    for model_name in ("CNN", "LSTM", "BiGRU_Attention", "Transformer"):
        model, best_epoch = train_with_early_stopping(model_name, X_fit, y_fit, X_val, y_val, len(le.classes_))
        macro_f1, report, cm = evaluate(model, X_test, y_test, le.classes_.astype(str))
        results[model_name] = {"macro_f1_permuted": macro_f1, "stopped_epoch": best_epoch}
        print(f"{model_name}: macro-F1(permuted)={macro_f1:.4f} (stopped at epoch {best_epoch})")

    natural = json.load(open(RESULTS_DIR / "onbody_dl_subject_safe_results.json"))
    for model_name in ("CNN", "LSTM", "BiGRU_Attention", "Transformer"):
        results[model_name]["macro_f1_natural_order"] = natural[model_name]["macro_f1"]
        results[model_name]["delta"] = results[model_name]["macro_f1_permuted"] - natural[model_name]["macro_f1"]
        print(f"{model_name}: natural={natural[model_name]['macro_f1']:.4f}  "
              f"permuted={results[model_name]['macro_f1_permuted']:.4f}  "
              f"delta={results[model_name]['delta']:+.4f}")

    out_path = RESULTS_DIR / "feature_permutation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
