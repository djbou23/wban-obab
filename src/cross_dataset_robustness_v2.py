"""Extends cross_dataset_robustness.py to address the remaining half of reviewer Major
Concern 7: in the original 5-seed sweep, the on-body subject-level train/test split was
FIXED across all seeds (only classifier randomness and the WUSTL split varied), so the
zero standard deviation on the on-body side was close to definitional, not evidence of
robustness. This repeats the WUSTL<->OnBody transfer across 10 INDEPENDENT subject-level
partitions of the on-body benchmark (a fresh 80/20 subject split drawn each repeat, RF
retrained from scratch each time), reporting the resulting spread as a genuine subject-
resampling-based confidence interval on the transfer result.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import cross_dataset_generalization as cdg

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
N_REPEATS = 10


def load_onbody_raw_with_subject():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    out = pd.DataFrame({
        "heart_rate_bpm": df["hr_mean"], "network_rate": df["tx_rate"], "loss_rate": df["packet_loss_rate"],
        "binary_label": (df["label"] != "Benign").astype(int),
        "fine_label": df["label"], "subject": df["subject"],
    })
    return out.dropna()


def main():
    wustl = cdg.zscore(cdg.load_wustl_aligned(), cdg.FEATURES)
    onbody_full = load_onbody_raw_with_subject()
    subjects = sorted(onbody_full["subject"].unique(), key=lambda s: int(s[1:]))

    wustl_to_onbody, onbody_to_wustl, in_domain_onbody = [], [], []
    for repeat in range(N_REPEATS):
        rng = np.random.default_rng(1000 + repeat)
        test_subjects = set(rng.choice(subjects, size=max(1, round(len(subjects) * 0.2)), replace=False))
        onbody = onbody_full.copy()
        onbody["split"] = onbody["subject"].apply(lambda s: "test" if s in test_subjects else "train")
        onbody = cdg.zscore(onbody, cdg.FEATURES)
        onbody_train = onbody[onbody["split"] == "train"]
        onbody_test = onbody[onbody["split"] == "test"]

        clf_in = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        clf_in.fit(onbody_train[cdg.FEATURES], onbody_train["binary_label"])
        in_domain_onbody.append(f1_score(onbody_test["binary_label"], clf_in.predict(onbody_test[cdg.FEATURES]), average="macro"))

        clf_wo = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        clf_wo.fit(wustl[cdg.FEATURES], wustl["binary_label"])
        pred_wo = clf_wo.predict(onbody_test[cdg.FEATURES])
        wustl_to_onbody.append(f1_score(onbody_test["binary_label"], pred_wo, average="macro"))

        clf_ow = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        clf_ow.fit(onbody_train[cdg.FEATURES], onbody_train["binary_label"])
        pred_ow = clf_ow.predict(wustl[cdg.FEATURES])
        onbody_to_wustl.append(f1_score(wustl["binary_label"], pred_ow, average="macro"))

        print(f"repeat {repeat}: test_subjects={sorted(test_subjects)} "
              f"in_domain_OnBody={in_domain_onbody[-1]:.4f} "
              f"WUSTL->OnBody={wustl_to_onbody[-1]:.4f} OnBody->WUSTL={onbody_to_wustl[-1]:.4f}")

    def summarize(vals):
        v = np.array(vals)
        return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max()), "all": vals}

    results = {
        "n_repeats": N_REPEATS,
        "in_domain_OnBody": summarize(in_domain_onbody),
        "WUSTL_to_OnBody": summarize(wustl_to_onbody),
        "OnBody_to_WUSTL": summarize(onbody_to_wustl),
    }
    print(f"\nWUSTL->OnBody: {results['WUSTL_to_OnBody']['mean']:.4f} +/- {results['WUSTL_to_OnBody']['std']:.4f} "
          f"(range {results['WUSTL_to_OnBody']['min']:.4f}-{results['WUSTL_to_OnBody']['max']:.4f})")
    print(f"OnBody->WUSTL: {results['OnBody_to_WUSTL']['mean']:.4f} +/- {results['OnBody_to_WUSTL']['std']:.4f} "
          f"(range {results['OnBody_to_WUSTL']['min']:.4f}-{results['OnBody_to_WUSTL']['max']:.4f})")

    out_path = RESULTS_DIR / "cross_dataset_robustness_v2_subject_resampled_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
