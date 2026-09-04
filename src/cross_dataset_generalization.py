"""Cross-dataset generalization experiment (C1, dimension d) -- the paper's key
empirical punchline: does a detector trained on a generic/near-generic IoMT source
transfer to genuinely WBAN-specific attacks, or vice versa?

Design rationale (see PROGRESS.md "Open design question" for the full reasoning):
CICIoMT2024, WUSTL-EHMS-2020, and the on-body synthetic benchmark do NOT share a
common feature schema, so literal cross-dataset model transfer is only defensible
between sources with genuinely comparable measured quantities. WUSTL-EHMS-2020 and
the on-body benchmark both measure heart rate in bpm and both have an explicit
loss-rate-style network feature -- CICIoMT2024 has neither (no biometrics, no
explicit loss feature), so it is excluded from this specific experiment and stays
in the paper as an in-domain-only benchmark result.

Method: binary Benign-vs-Attack classification using 3 aligned features
(heart_rate_bpm, network_rate, loss_rate), each z-scored WITHIN its own dataset
(standard practice for cross-domain transfer to correct for different collection
scales -- explicitly disclosed here and in the paper, not hidden).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_wustl_aligned():
    df = pd.read_parquet(DATA_DIR / "wustl_ehms_2020.parquet")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({
        "heart_rate_bpm": df["Pulse_Rate"],
        "network_rate": df["Rate"],
        "loss_rate": df["pLoss"],
        "binary_label": (df["Label"] == 1).astype(int),  # 1 = attack
    })
    return out.dropna()


def load_onbody_aligned():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    out = pd.DataFrame({
        "heart_rate_bpm": df["hr_mean"],
        "network_rate": df["tx_rate"],
        "loss_rate": df["packet_loss_rate"],
        "binary_label": (df["label"] != "Benign").astype(int),
        "split": df["split"],
        "fine_label": df["label"],
    })
    return out.dropna()


def zscore(df, cols):
    df = df.copy()
    for c in cols:
        mu, sigma = df[c].mean(), df[c].std()
        df[c] = (df[c] - mu) / (sigma if sigma > 0 else 1.0)
    return df


FEATURES = ["heart_rate_bpm", "network_rate", "loss_rate"]


def main():
    wustl = zscore(load_wustl_aligned(), FEATURES)
    onbody = zscore(load_onbody_aligned(), FEATURES)

    onbody_train = onbody[onbody["split"] == "train"]
    onbody_test = onbody[onbody["split"] == "test"]

    results = {}

    # --- In-domain baselines (upper bound reference) ---
    from sklearn.model_selection import train_test_split
    w_train, w_test = train_test_split(wustl, test_size=0.2, stratify=wustl["binary_label"], random_state=42)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(w_train[FEATURES], w_train["binary_label"])
    pred = clf.predict(w_test[FEATURES])
    results["in_domain_WUSTL"] = f1_score(w_test["binary_label"], pred, average="macro")

    clf2 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf2.fit(onbody_train[FEATURES], onbody_train["binary_label"])
    pred2 = clf2.predict(onbody_test[FEATURES])
    results["in_domain_OnBody"] = f1_score(onbody_test["binary_label"], pred2, average="macro")

    # --- Cross-dataset transfer ---
    # Train on WUSTL (generic/near-generic), test on OnBody (WBAN-specific attacks)
    clf3 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf3.fit(wustl[FEATURES], wustl["binary_label"])
    pred3 = clf3.predict(onbody_test[FEATURES])
    results["train_WUSTL_test_OnBody"] = f1_score(onbody_test["binary_label"], pred3, average="macro")
    # per-attack-type breakdown -- which WBAN-specific attacks does generic training miss?
    per_attack = {}
    for label in onbody_test["fine_label"].unique():
        mask = onbody_test["fine_label"] == label
        if mask.sum() == 0:
            continue
        y_true = onbody_test.loc[mask, "binary_label"]
        y_pred = pred3[mask.values]
        per_attack[label] = float((y_true == y_pred).mean())  # recall-equivalent for this class
    results["train_WUSTL_test_OnBody_per_class_accuracy"] = per_attack

    # Train on OnBody, test on WUSTL (reverse direction)
    clf4 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf4.fit(onbody_train[FEATURES], onbody_train["binary_label"])
    pred4 = clf4.predict(wustl[FEATURES])
    results["train_OnBody_test_WUSTL"] = f1_score(wustl["binary_label"], pred4, average="macro")

    print(json.dumps(results, indent=2, default=str))
    out_path = RESULTS_DIR / "cross_dataset_generalization_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
