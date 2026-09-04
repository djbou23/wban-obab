"""SHAP explainability pass (C1, dimension c).

Two questions this needs to answer for the paper:
1. WUSTL-EHMS-2020: raw feature means barely differ between normal/attack for the
   biometric columns (see eda_wustl_ehms.py output) -- does the trained model
   actually use physiological features at all, or lean entirely on network features?
2. On-body benchmark: which features drive the Benign/Replay confusion
   (F1 0.60/0.72 -- the weakest pair in train_baselines_onbody.py)?
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

WUSTL_BIOMETRIC = ["Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"]


def wustl_shap():
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
    sv_attack = sv[1] if isinstance(sv, list) else sv[:, :, 1]  # attack class

    mean_abs = np.abs(sv_attack).mean(axis=0)
    importance = pd.Series(mean_abs, index=feat_cols).sort_values(ascending=False)

    biometric_total = importance[importance.index.isin(WUSTL_BIOMETRIC)].sum()
    total = importance.sum()

    print("Top 15 features by mean |SHAP| (WUSTL, attack class):")
    print(importance.head(15))
    print(f"\nBiometric feature share of total |SHAP| importance: {biometric_total/total:.1%}")
    print(f"Network feature share: {1 - biometric_total/total:.1%}")

    return {
        "top_features": importance.head(15).to_dict(),
        "biometric_share": float(biometric_total / total),
        "network_share": float(1 - biometric_total / total),
    }


def onbody_shap():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    feat_cols = ["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean", "motion_energy",
                 "tx_rate", "packet_loss_rate"]
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]

    le = LabelEncoder()
    y_train, y_test = le.fit_transform(train["label"]), le.transform(test["label"])
    X_train, X_test = train[feat_cols], test[feat_cols]

    clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    clf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(clf)
    # focus on the Benign vs Replay confusion specifically
    benign_idx = list(le.classes_).index("Benign")
    replay_idx = list(le.classes_).index("Replay")
    confused_mask = test["label"].isin(["Benign", "Replay"])
    sample = X_test[confused_mask.values]

    sv = explainer.shap_values(sample)
    sv_replay = sv[replay_idx] if isinstance(sv, list) else sv[:, :, replay_idx]

    mean_abs = np.abs(sv_replay).mean(axis=0)
    importance = pd.Series(mean_abs, index=feat_cols).sort_values(ascending=False)

    print("\nTop features by mean |SHAP| for Replay class (Benign-vs-Replay subset):")
    print(importance)

    return {"replay_vs_benign_top_features": importance.to_dict()}


def main():
    results = {}
    print("=== WUSTL-EHMS-2020 ===")
    results["wustl"] = wustl_shap()
    print("\n=== On-Body Benchmark ===")
    results["onbody"] = onbody_shap()

    out_path = RESULTS_DIR / "shap_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
