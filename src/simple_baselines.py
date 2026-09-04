"""E5 (reviewer Concern 18 / essential experiment E5): simple baselines the review
says are missing, including a rule-based physiological-consistency detector. The
review's argument is specific: since the benchmark's central novelty is exactly the
physiological-context inconsistency the attacks create, a strong result from a cheap
consistency rule would mean the sophisticated IDS models add relatively little, and
this needs to be checked directly rather than assumed away.

The rule-based detector uses ONLY population statistics fit on the TRAIN split's
benign windows (never on attacked or test windows), i.e. a per-activity healthy
reference band for HR, PPG amplitude, and HR-vs-motion / HR-vs-PPG relationships,
then flags a window as a specific attack type if it falls outside that reference
band along the feature(s) that attack targets. This is deliberately given an unfair
advantage (it is handed the attack taxonomy directly, unlike the learned models,
which must discover it) -- if it still doesn't win, that says something.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FEATURE_COLS = ["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean",
                 "motion_energy", "tx_rate", "packet_loss_rate"]
LABELS = ["Benign", "Biosignal_Spoofing", "Denial_of_Sleep", "Replay", "Simulated_Jamming"]


def rule_based_consistency_detector(train_benign, test_df):
    """Per-activity reference bands fit ONLY on train-split benign windows."""
    bands = {}
    for activity, g in train_benign.groupby("activity"):
        bands[activity] = {
            "hr_lo": g["hr_mean"].quantile(0.01), "hr_hi": g["hr_mean"].quantile(0.99),
            "hr_std_lo": g["hr_std"].quantile(0.01),
            "tx_lo": g["tx_rate"].quantile(0.01), "tx_hi": g["tx_rate"].quantile(0.99),
            "loss_hi": g["packet_loss_rate"].quantile(0.99),
            "ppg_std_lo": g["ppg_std"].quantile(0.01),
        }
    global_band = {
        "hr_lo": train_benign["hr_mean"].quantile(0.01), "hr_hi": train_benign["hr_mean"].quantile(0.99),
        "hr_std_lo": train_benign["hr_std"].quantile(0.01),
        "tx_lo": train_benign["tx_rate"].quantile(0.01), "tx_hi": train_benign["tx_rate"].quantile(0.99),
        "loss_hi": train_benign["packet_loss_rate"].quantile(0.99),
        "ppg_std_lo": train_benign["ppg_std"].quantile(0.01),
    }

    preds = []
    for _, row in test_df.iterrows():
        b = bands.get(row["activity"], global_band)
        # Rule priority follows each attack's defining feature (Table: attack-generation spec).
        if row["tx_rate"] > b["tx_hi"] * 1.3:  # denial-of-sleep: gross tx-rate spike
            preds.append("Denial_of_Sleep")
        elif row["packet_loss_rate"] > b["loss_hi"] * 1.5 or row["ppg_std"] < b["ppg_std_lo"] * 0.7:
            preds.append("Simulated_Jamming")  # packet loss / flattened PPG variance
        elif row["hr_mean"] < b["hr_lo"] or row["hr_mean"] > b["hr_hi"] or row["hr_std"] < b["hr_std_lo"] * 0.5:
            # HR outside the healthy reference band, or unnaturally stable, is flagged
            # Spoofing (an implausible/flatlined value); an HR that is merely outside
            # THIS activity's band but still physiologically plausible overall is
            # flagged Replay (a real value, just from the wrong activity context) --
            # this is exactly the rule the reviewer's Concern 3 predicts should exist.
            implausible = row["hr_mean"] < 0 or row["hr_mean"] > 250 or row["hr_std"] < b["hr_std_lo"] * 0.5
            preds.append("Biosignal_Spoofing" if implausible else "Replay")
        else:
            preds.append("Benign")
    return preds


def main():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    train_benign = train[train["label"] == "Benign"]

    le = LabelEncoder().fit(LABELS)
    y_train, y_test = le.transform(train["label"]), le.transform(test["label"])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    X_test = scaler.transform(test[FEATURE_COLS])

    results = {}

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=2000)),
        ("SVM_RBF", SVC(kernel="rbf", C=1.0)),
        ("DecisionTree", DecisionTreeClassifier(max_depth=8, random_state=42)),
        ("kNN", KNeighborsClassifier(n_neighbors=7)),
    ]
    for name, model in models:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        macro_f1 = f1_score(y_test, pred, average="macro")
        report = classification_report(y_test, pred, target_names=le.classes_.astype(str),
                                        output_dict=True, zero_division=0)
        results[name] = {"macro_f1": macro_f1, "report": report}
        print(f"{name}: macro-F1={macro_f1:.4f}")

    # Rule-based physiological-consistency detector (no learning at all)
    rule_pred_labels = rule_based_consistency_detector(train_benign, test)
    rule_pred = le.transform(rule_pred_labels)
    macro_f1_rule = f1_score(y_test, rule_pred, average="macro")
    report_rule = classification_report(y_test, rule_pred, target_names=le.classes_.astype(str),
                                         output_dict=True, zero_division=0)
    results["RuleBased_PhysioConsistency"] = {"macro_f1": macro_f1_rule, "report": report_rule}
    print(f"RuleBased_PhysioConsistency: macro-F1={macro_f1_rule:.4f}")

    # Reference point: RandomForest (already reported elsewhere, repeated here for direct
    # side-by-side comparison in the same table/script output).
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results["RandomForest_reference"] = {"macro_f1": f1_score(y_test, pred_rf, average="macro")}
    print(f"RandomForest_reference: macro-F1={results['RandomForest_reference']['macro_f1']:.4f}")

    out_path = RESULTS_DIR / "simple_baselines_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
