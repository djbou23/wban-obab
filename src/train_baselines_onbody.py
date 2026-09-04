"""Baseline detection performance on the novel on-body synthetic attack benchmark.

This is the C1 benchmarking-study result for OUR OWN constructed benchmark (as opposed
to train_baselines_classical.py, which runs on the existing CICIoMT2024/WUSTL sources).
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FEATURE_COLS = ["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean",
                 "motion_energy", "tx_rate", "packet_loss_rate"]


def main():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]

    le = LabelEncoder()
    y_train, y_test = le.fit_transform(train["label"]), le.transform(test["label"])
    X_train, X_test = train[FEATURE_COLS], test[FEATURE_COLS]

    results = {}
    for name, model in [
        ("RandomForest", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)),
        ("XGBoost", XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric="mlogloss")),
    ]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, target_names=le.classes_.astype(str),
                                        output_dict=True, zero_division=0)
        results[name] = {"macro_f1": macro_f1, "report": report}
        print(f"--- {name} --- macro-F1: {macro_f1:.4f}")
        for label in le.classes_:
            r = report[label]
            print(f"    {label:20s} P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1-score']:.2f}")

    out_path = RESULTS_DIR / "onbody_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
