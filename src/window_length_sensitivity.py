"""E3 (essential experiment E3): the benchmark's fundamental unit is a 10-second
window with no systematic justification for that choice. Re-extracts features and
regenerates the full attack set at window lengths of 5, 10, 20, and 30 seconds
(reusing the exact feature/attack logic in synthesize_onbody_attacks.py, parameterized
by WINDOW_SEC) and reports RandomForest macro-F1 at each length, using the same
subject-level split logic (same RNG seed, so the same test subjects are held out at
every window length for a fair comparison).
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
WINDOW_LENGTHS = [5, 10, 20, 30]


def generate_at_window_length(window_sec, seed=42):
    fs = syn.FS
    window_samples = fs * window_sec
    rng = np.random.default_rng(seed)

    records = syn.list_records()
    all_windows = []
    for subject, activity, path in records:
        df = pd.read_csv(path)
        n_windows = len(df) // window_samples
        for w in range(n_windows):
            start, end = w * window_samples, (w + 1) * window_samples
            feat = syn.compute_window_features(df.iloc[start:end])
            feat["tx_rate"] = syn.NORMAL_TX_RATE * rng.uniform(0.9, 1.1)
            feat["packet_loss_rate"] = rng.uniform(0.0, 0.02)
            all_windows.append({"subject": subject, "activity": activity, "window_idx": w, **feat})
    windows_df = pd.DataFrame(all_windows).dropna(subset=["hr_mean", "hr_std"]).reset_index(drop=True)

    subjects = sorted(windows_df["subject"].unique(), key=lambda s: int(s[1:]))
    test_subjects = set(rng.choice(subjects, size=max(1, round(len(subjects) * 0.2)), replace=False))
    windows_df["split"] = windows_df["subject"].apply(lambda s: "test" if s in test_subjects else "train")

    rows = []
    for idx, row in windows_df.iterrows():
        base_feat = row[["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean", "motion_energy"]].to_dict()
        subject, activity, split = row["subject"], row["activity"], row["split"]
        common = {"subject": subject, "activity": activity, "split": split}

        benign = dict(base_feat, tx_rate=row["tx_rate"], packet_loss_rate=row["packet_loss_rate"])
        rows.append({**benign, "label": "Benign", **common})
        rows.append({**syn.apply_biosignal_spoofing(base_feat), "label": "Biosignal_Spoofing", **common})

        pool = windows_df[(windows_df["activity"] != activity) & (windows_df["split"] == split)]
        if len(pool) > 0:
            donor = pool.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
            donor_feat = donor[["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean", "motion_energy"]].to_dict()
            rows.append({**syn.apply_replay(base_feat, donor_feat), "label": "Replay", **common})

        rows.append({**syn.apply_denial_of_sleep(base_feat), "label": "Denial_of_Sleep", **common})
        rows.append({**syn.apply_jamming(base_feat), "label": "Simulated_Jamming", **common})

    return pd.DataFrame(rows)


def main():
    results = {}
    for window_sec in WINDOW_LENGTHS:
        print(f"\n=== window = {window_sec}s ===")
        df = generate_at_window_length(window_sec)
        train, test = df[df["split"] == "train"], df[df["split"] == "test"]
        le = LabelEncoder()
        y_train, y_test = le.fit_transform(train["label"]), le.transform(test["label"])
        clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        clf.fit(train[FEATURE_COLS], y_train)
        pred = clf.predict(test[FEATURE_COLS])
        macro_f1 = f1_score(y_test, pred, average="macro")
        n_windows_total = len(df) // 5  # 5 rows (benign+4 attacks) per source window
        results[str(window_sec)] = {
            "macro_f1": float(macro_f1), "n_source_windows": int(n_windows_total),
            "n_train": int(len(train)), "n_test": int(len(test)),
        }
        print(f"  macro-F1={macro_f1:.4f}  (source windows={n_windows_total}, train={len(train)}, test={len(test)})")

    out_path = RESULTS_DIR / "window_length_sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
