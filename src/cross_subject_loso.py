"""E6 (essential experiment E6): leave-one-subject-out evaluation across all 22
subjects (RandomForest, the fastest-training baseline, used so all 22 folds are
feasible). Directly answers reviewer Major Concern 7 (five seeds on a FIXED
train/test split is not five independent experiments) and Major Concern 8 (report
performance by held-out subject) with genuine subject-level resampling rather than
classifier-seed resampling on a fixed partition.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FEATURE_COLS = ["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean",
                 "motion_energy", "tx_rate", "packet_loss_rate"]


def main():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    le = LabelEncoder().fit(df["label"])
    subjects = sorted(df["subject"].unique(), key=lambda s: int(s[1:]))

    per_subject_f1 = {}
    for held_out in subjects:
        train = df[df["subject"] != held_out]
        test = df[df["subject"] == held_out]
        clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        clf.fit(train[FEATURE_COLS], le.transform(train["label"]))
        pred = clf.predict(test[FEATURE_COLS])
        f1 = f1_score(le.transform(test["label"]), pred, average="macro")
        per_subject_f1[held_out] = float(f1)
        print(f"held out {held_out:4s} (n={len(test):4d}): macro-F1={f1:.4f}")

    vals = np.array(list(per_subject_f1.values()))
    summary = {
        "per_subject_macro_f1": per_subject_f1,
        "mean": float(vals.mean()), "std": float(vals.std()),
        "min": float(vals.min()), "max": float(vals.max()),
        "min_subject": min(per_subject_f1, key=per_subject_f1.get),
        "max_subject": max(per_subject_f1, key=per_subject_f1.get),
        # for comparison: the single fixed 4-subject held-out split reported elsewhere
        "fixed_split_reference_macro_f1": 0.8859,
    }
    print(f"\nLOSO across {len(subjects)} subjects: mean={summary['mean']:.4f} +/- {summary['std']:.4f} "
          f"(min={summary['min']:.4f} @ {summary['min_subject']}, max={summary['max']:.4f} @ {summary['max_subject']})")

    out_path = RESULTS_DIR / "cross_subject_loso_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
