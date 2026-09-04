"""Addresses the remaining part of reviewer Major Concern 12: the mean/median per-feature
SHAP attribution ratio (shap_analysis_v2.py) was computed from a single 1000-sample draw
and a single trained model, with no variation reported. This repeats the explained-sample
draw 5 times (same fitted RandomForest, 5 disjoint samples of 1000 test points each) to
report a mean +/- std for both ratios, checking whether they are stable or an artifact of
one particular sample.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
WUSTL_BIOMETRIC = ["Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"]
N_SAMPLES_PER_DRAW = 1000
N_DRAWS = 5


def main():
    df = pd.read_parquet(DATA_DIR / "wustl_ehms_2020.parquet")
    df.columns = [c.strip() for c in df.columns]
    non_feat = {"Label", "Attack Category", "SrcAddr", "DstAddr", "SrcMac", "DstMac", "Dir", "Flgs"}
    feat_cols = [c for c in df.columns if c not in non_feat and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    clf.fit(X_train, y_train)
    explainer = shap.TreeExplainer(clf)

    biometric_mask = pd.Index(feat_cols).isin(WUSTL_BIOMETRIC)
    mean_ratios, median_ratios = [], []
    for draw in range(N_DRAWS):
        sample = X_test.sample(min(N_SAMPLES_PER_DRAW, len(X_test)), random_state=100 + draw)
        sv = explainer.shap_values(sample)
        sv_attack = sv[1] if isinstance(sv, list) else sv[:, :, 1]
        importance = pd.Series(np.abs(sv_attack).mean(axis=0), index=feat_cols)

        bio_mean, net_mean = importance[biometric_mask].mean(), importance[~biometric_mask].mean()
        bio_median, net_median = importance[biometric_mask].median(), importance[~biometric_mask].median()
        mean_ratios.append(bio_mean / net_mean)
        median_ratios.append(bio_median / net_median)
        print(f"draw {draw}: mean_ratio={mean_ratios[-1]:.3f}  median_ratio={median_ratios[-1]:.3f}")

    results = {
        "n_draws": N_DRAWS, "n_samples_per_draw": N_SAMPLES_PER_DRAW,
        "mean_ratio_mean": float(np.mean(mean_ratios)), "mean_ratio_std": float(np.std(mean_ratios)),
        "median_ratio_mean": float(np.mean(median_ratios)), "median_ratio_std": float(np.std(median_ratios)),
        "all_mean_ratios": mean_ratios, "all_median_ratios": median_ratios,
    }
    print(f"\nMean-attribution ratio:   {results['mean_ratio_mean']:.3f} +/- {results['mean_ratio_std']:.3f}")
    print(f"Median-attribution ratio: {results['median_ratio_mean']:.3f} +/- {results['median_ratio_std']:.3f}")

    out_path = RESULTS_DIR / "shap_robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
