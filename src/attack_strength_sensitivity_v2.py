"""Fix review-round-2 issue: the attack-strength sensitivity sweep used a single seed,
leaving an unexplained non-monotonic dip in the Benign class at the "moderate" level with
no way to tell if it was signal or noise. This repeats each strength level across 5 seeds
(varying both the synthetic perturbation draws and the classifier), reporting mean +/- std.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

import synthesize_onbody_attacks as syn
from attack_strength_sensitivity import LEVELS, build_windows, synthesize_at_level

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FEATURE_COLS = ["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean", "motion_energy",
                 "tx_rate", "packet_loss_rate"]
SEEDS = [42, 1, 7, 123, 2024]


def main():
    windows_df = build_windows()
    subjects = sorted(windows_df["subject"].unique(), key=lambda s: int(s[1:]))
    rng_split = np.random.default_rng(42)
    test_subjects = set(rng_split.choice(subjects, size=max(1, round(len(subjects) * 0.2)), replace=False))

    results = {}
    for level_name, dos_range, jam_range, spoof_range in LEVELS:
        macro_f1s = []
        per_class_runs = {}

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            df = synthesize_at_level(windows_df, dos_range, jam_range, spoof_range, rng)
            df["split"] = df["subject"].apply(lambda s: "test" if s in test_subjects else "train")
            train, test = df[df["split"] == "train"], df[df["split"] == "test"]

            le = LabelEncoder()
            y_train, y_test = le.fit_transform(train["label"]), le.transform(test["label"])
            clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=seed)
            clf.fit(train[FEATURE_COLS], y_train)
            pred = clf.predict(test[FEATURE_COLS])

            macro_f1s.append(f1_score(y_test, pred, average="macro"))
            for cls in le.classes_:
                cls_idx = list(le.classes_).index(cls)
                f1 = f1_score((y_test == cls_idx).astype(int), (pred == cls_idx).astype(int))
                per_class_runs.setdefault(cls, []).append(f1)

        results[level_name] = {
            "macro_f1_mean": float(np.mean(macro_f1s)), "macro_f1_std": float(np.std(macro_f1s)),
            "per_class_f1_mean": {k: float(np.mean(v)) for k, v in per_class_runs.items()},
            "per_class_f1_std": {k: float(np.std(v)) for k, v in per_class_runs.items()},
        }
        print(f"\n=== {level_name} ===")
        print(f"  macro-F1: {results[level_name]['macro_f1_mean']:.4f} +/- {results[level_name]['macro_f1_std']:.4f}")
        for cls in per_class_runs:
            m, s = results[level_name]["per_class_f1_mean"][cls], results[level_name]["per_class_f1_std"][cls]
            print(f"    {cls}: {m:.4f} +/- {s:.4f}")

    out_path = RESULTS_DIR / "attack_strength_sensitivity_v2_multiseed_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
