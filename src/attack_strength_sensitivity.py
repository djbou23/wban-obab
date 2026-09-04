"""Attack-strength sensitivity sweep (review issue 2).

The original denial-of-sleep and jamming attacks used a single, fairly aggressive
parameter range (15-40x transmission rate; 15-50% dropout), which makes their near-perfect
detection scores partly a foregone conclusion of the synthesis parameters rather than
evidence the attack is inherently easy to detect. This script re-synthesizes each attack
across a range of strengths, from subtle (close to benign) to the original aggressive
setting, and reports how detection F1 degrades as the attack becomes stealthier -- turning
a validity threat into an operating-range characterization.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

import synthesize_onbody_attacks as syn

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FEATURE_COLS = ["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean", "motion_energy",
                 "tx_rate", "packet_loss_rate"]

# strength levels: (label, dos_rate_mult_range, jam_dropout_range, spoof_hr_dev_range)
# benign natural range is tx_rate x(0.9-1.1), packet_loss_rate 0.0-0.02, no HR deviation --
# near_benign deliberately overlaps that range so detection is expected to degrade toward chance.
LEVELS = [
    ("near_benign", (0.9, 1.3), (0.0, 0.03), (1, 5)),
    ("subtle", (1.5, 3.0), (0.03, 0.10), (5, 15)),
    ("moderate", (4.0, 8.0), (0.08, 0.20), (12, 25)),
    ("original", (15.0, 40.0), (0.15, 0.50), (40, 80)),
]


def build_windows():
    records = syn.list_records()
    all_windows = []
    for subject, activity, path in records:
        df = pd.read_csv(path)
        n_windows = len(df) // syn.WINDOW_SAMPLES
        for w in range(n_windows):
            start, end = w * syn.WINDOW_SAMPLES, (w + 1) * syn.WINDOW_SAMPLES
            feat = syn.compute_window_features(df.iloc[start:end])
            all_windows.append({"subject": subject, "activity": activity, **feat})
    return pd.DataFrame(all_windows).dropna(subset=["hr_mean", "hr_std"])


def synthesize_at_level(windows_df, dos_range, jam_range, spoof_range, rng):
    rows = []
    for _, row in windows_df.iterrows():
        base_feat = row[["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean", "motion_energy"]].to_dict()
        subject, activity = row["subject"], row["activity"]

        rows.append({**base_feat, "tx_rate": syn.NORMAL_TX_RATE * rng.uniform(0.9, 1.1),
                     "packet_loss_rate": rng.uniform(0.0, 0.02), "label": "Benign",
                     "subject": subject, "activity": activity})

        f = dict(base_feat)
        if rng.random() < 0.5:
            f["hr_mean"] = f["hr_mean"] + rng.choice([-1, 1]) * rng.uniform(*spoof_range)
        else:
            f["hr_mean"] = max(0, f["hr_mean"] - rng.uniform(*spoof_range))
        f["hr_std"] = f["hr_std"] * rng.uniform(0.0, 0.3)
        f["tx_rate"] = syn.NORMAL_TX_RATE * rng.uniform(0.9, 1.2)
        f["packet_loss_rate"] = rng.uniform(0.0, 0.02)
        rows.append({**f, "label": "Biosignal_Spoofing", "subject": subject, "activity": activity})

        f2 = dict(base_feat)
        f2["tx_rate"] = syn.NORMAL_TX_RATE * rng.uniform(*dos_range)
        f2["packet_loss_rate"] = rng.uniform(0.0, 0.05)
        rows.append({**f2, "label": "Denial_of_Sleep", "subject": subject, "activity": activity})

        f3 = dict(base_feat)
        dropout = rng.uniform(*jam_range)
        f3["hr_std"] = f3["hr_std"] * (1 + dropout * 3) if not np.isnan(f3["hr_std"]) else f3["hr_std"]
        f3["ppg_std"] = f3["ppg_std"] * (1 - dropout)
        f3["tx_rate"] = syn.NORMAL_TX_RATE * (1 - dropout)
        f3["packet_loss_rate"] = dropout
        rows.append({**f3, "label": "Simulated_Jamming", "subject": subject, "activity": activity})

    return pd.DataFrame(rows)


def main():
    windows_df = build_windows()
    subjects = sorted(windows_df["subject"].unique(), key=lambda s: int(s[1:]))
    rng_split = np.random.default_rng(42)
    test_subjects = set(rng_split.choice(subjects, size=max(1, round(len(subjects) * 0.2)), replace=False))

    results = {}
    for level_name, dos_range, jam_range, spoof_range in LEVELS:
        rng = np.random.default_rng(42)
        df = synthesize_at_level(windows_df, dos_range, jam_range, spoof_range, rng)
        df["split"] = df["subject"].apply(lambda s: "test" if s in test_subjects else "train")
        train, test = df[df["split"] == "train"], df[df["split"] == "test"]

        le = LabelEncoder()
        y_train, y_test = le.fit_transform(train["label"]), le.transform(test["label"])
        clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        clf.fit(train[FEATURE_COLS], y_train)
        pred = clf.predict(test[FEATURE_COLS])

        per_class = {}
        for cls in le.classes_:
            mask = test["label"] == cls
            if mask.sum() == 0:
                continue
            per_class[cls] = f1_score((y_test == list(le.classes_).index(cls)).astype(int),
                                       (pred == list(le.classes_).index(cls)).astype(int))
        macro_f1 = f1_score(y_test, pred, average="macro")
        results[level_name] = {"macro_f1": macro_f1, "per_class_f1": per_class,
                                "dos_range": dos_range, "jam_range": jam_range, "spoof_range": spoof_range}
        print(f"\n=== {level_name} (DoS x{dos_range}, jam {jam_range}, spoof +-{spoof_range} bpm) ===")
        print(f"  macro-F1: {macro_f1:.4f}")
        for cls, f1 in per_class.items():
            print(f"    {cls}: {f1:.4f}")

    out_path = RESULTS_DIR / "attack_strength_sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
