"""Robustness fixes for the cross-dataset generalization experiment (review issues 1 and 4).

1. Isolate the synthetic-feature confound: repeat the WUSTL<->OnBody transfer using ONLY
   heart_rate_bpm, the single feature that is genuinely measured (not hand-authored) in
   both datasets, and compare against the full 3-feature result.
2. Multi-seed robustness: repeat both the 3-feature and heart-rate-only experiments across
   5 seeds (varying the WUSTL train/test split and, where applicable, RF initialization),
   reporting mean and standard deviation rather than a single point estimate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import cross_dataset_generalization as cdg

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEEDS = [42, 1, 7, 123, 2024]


def run_direction(train_df, test_df, features, seed):
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed)
    clf.fit(train_df[features], train_df["binary_label"])
    pred = clf.predict(test_df[features])
    return f1_score(test_df["binary_label"], pred, average="macro"), pred


def per_class_accuracy(test_df, pred):
    out = {}
    for label in test_df["fine_label"].dropna().unique():
        mask = test_df["fine_label"] == label
        if mask.sum() == 0:
            continue
        out[label] = float((test_df.loc[mask, "binary_label"] == pred[mask.values]).mean())
    return out


def main():
    results = {"three_feature": {}, "heart_rate_only": {}}

    for feature_set_name, features in [("three_feature", cdg.FEATURES), ("heart_rate_only", ["heart_rate_bpm"])]:
        wustl_to_onbody, onbody_to_wustl = [], []
        in_domain_wustl, in_domain_onbody = [], []
        per_class_last = None

        for seed in SEEDS:
            wustl_raw = cdg.load_wustl_aligned()
            onbody_raw = cdg.load_onbody_aligned()
            wustl = cdg.zscore(wustl_raw, features)
            onbody = cdg.zscore(onbody_raw, features)
            onbody_train = onbody[onbody["split"] == "train"]
            onbody_test = onbody[onbody["split"] == "test"]

            w_train, w_test = train_test_split(wustl, test_size=0.2, stratify=wustl["binary_label"], random_state=seed)

            f1_in_w, _ = run_direction(w_train, w_test, features, seed)
            in_domain_wustl.append(f1_in_w)

            f1_in_o, _ = run_direction(onbody_train, onbody_test, features, seed)
            in_domain_onbody.append(f1_in_o)

            f1_wo, pred_wo = run_direction(wustl, onbody_test, features, seed)
            wustl_to_onbody.append(f1_wo)
            per_class_last = per_class_accuracy(onbody_test, pred_wo)

            f1_ow, _ = run_direction(onbody_train, wustl, features, seed)
            onbody_to_wustl.append(f1_ow)

        results[feature_set_name] = {
            "in_domain_WUSTL_mean": float(np.mean(in_domain_wustl)), "in_domain_WUSTL_std": float(np.std(in_domain_wustl)),
            "in_domain_OnBody_mean": float(np.mean(in_domain_onbody)), "in_domain_OnBody_std": float(np.std(in_domain_onbody)),
            "WUSTL_to_OnBody_mean": float(np.mean(wustl_to_onbody)), "WUSTL_to_OnBody_std": float(np.std(wustl_to_onbody)),
            "OnBody_to_WUSTL_mean": float(np.mean(onbody_to_wustl)), "OnBody_to_WUSTL_std": float(np.std(onbody_to_wustl)),
            "per_class_accuracy_seed42_example": per_class_last,
            "all_seeds_WUSTL_to_OnBody": wustl_to_onbody,
        }
        print(f"\n=== {feature_set_name} ===")
        for k, v in results[feature_set_name].items():
            if not isinstance(v, dict) and not isinstance(v, list):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")

    out_path = RESULTS_DIR / "cross_dataset_robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
