"""E7 (essential experiment E7): synthetic-feature ablation. Compares detection using
physiological-only features, network-proxy-only features, and the combined 8-feature
schema, to make explicit which attacks are only detectable through the synthetic
network-proxy features (tx_rate, packet_loss_rate) versus the genuinely measured
physiological ones -- directly addresses reviewer Major Concern 3's question of
whether the IDS is "detecting the attacks, or simply detecting the artificial
construction rules."
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

PHYSIO_COLS = ["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean", "motion_energy"]
NETWORK_COLS = ["tx_rate", "packet_loss_rate"]
ALL_COLS = PHYSIO_COLS + NETWORK_COLS


def run(train, test, cols, le):
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    clf.fit(train[cols], le.transform(train["label"]))
    pred = clf.predict(test[cols])
    y_test = le.transform(test["label"])
    macro_f1 = f1_score(y_test, pred, average="macro")
    report = classification_report(y_test, pred, target_names=le.classes_.astype(str),
                                    output_dict=True, zero_division=0)
    return macro_f1, report


def main():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    le = LabelEncoder().fit(df["label"])

    results = {}
    for name, cols in [("physio_only", PHYSIO_COLS), ("network_only", NETWORK_COLS), ("combined", ALL_COLS)]:
        macro_f1, report = run(train, test, cols, le)
        per_class_f1 = {k: v["f1-score"] for k, v in report.items() if k in le.classes_}
        results[name] = {"macro_f1": macro_f1, "per_class_f1": per_class_f1, "n_features": len(cols)}
        print(f"\n=== {name} ({len(cols)} features) === macro-F1: {macro_f1:.4f}")
        for cls, f1 in per_class_f1.items():
            print(f"    {cls:20s} F1={f1:.3f}")

    out_path = RESULTS_DIR / "feature_group_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
