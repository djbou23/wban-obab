"""SHAP re-analysis with per-feature normalization (review issue 6).

The original analysis summed raw |SHAP| across feature groups, which conflates group size
(37 network features vs 8 biometric features in WUSTL) with per-feature informativeness.
This adds a per-feature-mean comparison alongside the raw group-sum comparison.
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
    sample = X_test.sample(min(1000, len(X_test)), random_state=42)
    sv = explainer.shap_values(sample)
    sv_attack = sv[1] if isinstance(sv, list) else sv[:, :, 1]

    mean_abs = np.abs(sv_attack).mean(axis=0)
    importance = pd.Series(mean_abs, index=feat_cols)

    biometric_mask = importance.index.isin(WUSTL_BIOMETRIC)
    n_bio, n_net = biometric_mask.sum(), (~biometric_mask).sum()

    raw_bio_share = importance[biometric_mask].sum() / importance.sum()
    raw_net_share = 1 - raw_bio_share

    per_feature_bio_mean = importance[biometric_mask].mean()
    per_feature_net_mean = importance[~biometric_mask].mean()
    per_feature_ratio = per_feature_bio_mean / per_feature_net_mean

    per_feature_bio_median = importance[biometric_mask].median()
    per_feature_net_median = importance[~biometric_mask].median()
    per_feature_ratio_median = per_feature_bio_median / per_feature_net_median

    top2_network = importance[~biometric_mask].sort_values(ascending=False).head(2)
    top2_share_of_network_group = top2_network.sum() / importance[~biometric_mask].sum()

    print(f"Feature counts: {n_bio} biometric, {n_net} network")
    print(f"Raw group-sum share: biometric {raw_bio_share:.1%}, network {raw_net_share:.1%}")
    print(f"Per-feature MEAN |SHAP|: biometric {per_feature_bio_mean:.5f}, network {per_feature_net_mean:.5f}, ratio {per_feature_ratio:.3f}")
    print(f"Per-feature MEDIAN |SHAP|: biometric {per_feature_bio_median:.5f}, network {per_feature_net_median:.5f}, ratio {per_feature_ratio_median:.3f}")
    print(f"Top-2 network features' share of total network-group attribution: {top2_share_of_network_group:.1%}")
    print("\nInterpretation: the mean ratio could be skewed by a small number of dominant network")
    print("features; the median ratio and the top-2 concentration check whether that is the case.")

    result = {
        "n_biometric_features": int(n_bio), "n_network_features": int(n_net),
        "raw_biometric_share": float(raw_bio_share), "raw_network_share": float(raw_net_share),
        "per_feature_mean_biometric": float(per_feature_bio_mean),
        "per_feature_mean_network": float(per_feature_net_mean),
        "per_feature_ratio_bio_over_net_mean": float(per_feature_ratio),
        "per_feature_median_biometric": float(per_feature_bio_median),
        "per_feature_median_network": float(per_feature_net_median),
        "per_feature_ratio_bio_over_net_median": float(per_feature_ratio_median),
        "top2_network_share_of_network_group": float(top2_share_of_network_group),
    }
    out_path = RESULTS_DIR / "shap_normalized_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
