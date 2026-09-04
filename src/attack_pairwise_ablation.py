"""E8 (essential experiment E8): evaluate each attack independently (binary
Benign-vs-that-attack, isolating pairwise separability from the other three attack
types) and in the full combination (reference, already reported elsewhere). Answers
whether an attack's near-perfect F1 in the full 5-class task depends on the other
classes being present as contrast, or holds up in isolation.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FEATURE_COLS = ["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean",
                 "motion_energy", "tx_rate", "packet_loss_rate"]
ATTACKS = ["Biosignal_Spoofing", "Denial_of_Sleep", "Replay", "Simulated_Jamming"]


def main():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]

    results = {}
    for attack in ATTACKS:
        tr = train[train["label"].isin(["Benign", attack])]
        te = test[test["label"].isin(["Benign", attack])]
        y_tr = (tr["label"] == attack).astype(int)
        y_te = (te["label"] == attack).astype(int)

        clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        clf.fit(tr[FEATURE_COLS], y_tr)
        pred = clf.predict(te[FEATURE_COLS])
        f1 = f1_score(y_te, pred, average="macro")
        results[attack] = {"binary_macro_f1_vs_benign_only": f1, "n_train": len(tr), "n_test": len(te)}
        print(f"{attack:20s} (vs Benign only): macro-F1={f1:.4f}")

    # Same but the other 3 attack types are also present in TRAIN (as they normally
    # would be) while the isolated pair is evaluated only against each other at test
    # time -- checks whether training alongside other attacks changes separability.
    for attack in ATTACKS:
        y_tr_full = train["label"]
        clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        clf.fit(train[FEATURE_COLS], y_tr_full)
        te = test[test["label"].isin(["Benign", attack])]
        pred = clf.predict(te[FEATURE_COLS])
        y_te_bin = (te["label"] == attack).astype(int)
        pred_bin = (pred == attack).astype(int)
        f1 = f1_score(y_te_bin, pred_bin, average="macro")
        results[attack]["binary_macro_f1_full_5class_model_restricted_eval"] = f1
        print(f"{attack:20s} (full-model, restricted eval): macro-F1={f1:.4f}")

    out_path = RESULTS_DIR / "attack_pairwise_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
