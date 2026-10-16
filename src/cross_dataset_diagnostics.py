"""Diagnostics for the WUSTL <-> on-body transfer experiment.

The original transfer macro-F1 values (0.167 and 0.111) coincide exactly with the macro-F1 of
constant predictions given each target's class prevalence, so they cannot by themselves be
read as a graded measure of domain shift. This script reports, for 3-feature and HR-only
transfer: (1) the fraction of target samples predicted as attack, (2) threshold-free ROC-AUC,
(3) a prevalence-matched re-run (class-weighted training, 50/50 test subsample) reporting
macro-F1, balanced accuracy and AUC, (4) per-attack ROC-AUC on the on-body target, and
(5) the single-feature attack-vs-benign AUC in each dataset (direction of each cue).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import cross_dataset_generalization as cdg

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEEDS = [42, 1, 7, 123, 2024]


def fit(train_df, features, seed, balanced):
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed,
                                 class_weight="balanced" if balanced else None)
    clf.fit(train_df[features], train_df["binary_label"])
    return clf


def balanced_subsample(df, seed):
    rng = np.random.RandomState(seed)
    pos = df[df["binary_label"] == 1]
    neg = df[df["binary_label"] == 0]
    n = min(len(pos), len(neg))
    return df.loc[np.concatenate([rng.choice(pos.index, n, replace=False),
                                  rng.choice(neg.index, n, replace=False)])]


def metrics(clf, test_df, features):
    y = test_df["binary_label"].values
    p = clf.predict_proba(test_df[features])[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "auc": roc_auc_score(y, p),
        "frac_pred_attack": float(pred.mean()),
        "attack_prevalence": float(y.mean()),
    }


def summarize(rows):
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in rows[0]}


def main():
    out = {}
    for name, features in [("three_feature", cdg.FEATURES), ("heart_rate_only", ["heart_rate_bpm"])]:
        wustl = cdg.zscore(cdg.load_wustl_aligned(), features)
        onbody = cdg.zscore(cdg.load_onbody_aligned(), features)
        ob_train, ob_test = onbody[onbody["split"] == "train"], onbody[onbody["split"] == "test"]

        res = {}
        # (1)+(2): original protocol (unweighted, native prevalence)
        rows = {"WUSTL_to_OnBody": [], "OnBody_to_WUSTL": []}
        for s in SEEDS:
            rows["WUSTL_to_OnBody"].append(metrics(fit(wustl, features, s, False), ob_test, features))
            rows["OnBody_to_WUSTL"].append(metrics(fit(ob_train, features, s, False), wustl, features))
        res["native_prevalence"] = {k: summarize(v) for k, v in rows.items()}

        # (3): prevalence-matched re-run
        rows = {"WUSTL_to_OnBody": [], "OnBody_to_WUSTL": [], "WUSTL_in_domain": [], "OnBody_in_domain": []}
        for s in SEEDS:
            w_tr, w_te = train_test_split(wustl, test_size=0.2, stratify=wustl["binary_label"], random_state=s)
            rows["WUSTL_to_OnBody"].append(metrics(fit(wustl, features, s, True), balanced_subsample(ob_test, s), features))
            rows["OnBody_to_WUSTL"].append(metrics(fit(ob_train, features, s, True), balanced_subsample(wustl, s), features))
            rows["WUSTL_in_domain"].append(metrics(fit(w_tr, features, s, True), balanced_subsample(w_te, s), features))
            rows["OnBody_in_domain"].append(metrics(fit(ob_train, features, s, True), balanced_subsample(ob_test, s), features))
        res["prevalence_matched"] = {k: summarize(v) for k, v in rows.items()}

        # (4): per-attack AUC on the on-body target for the WUSTL-trained model
        per_attack = {}
        clf = fit(wustl, features, 42, False)
        benign = ob_test[ob_test["fine_label"] == "Benign"]
        for lab in sorted(ob_test["fine_label"].unique()):
            if lab == "Benign":
                continue
            sub = ob_test[ob_test["fine_label"] == lab]
            both = pd.concat([sub, benign])
            per_attack[lab] = float(roc_auc_score(both["binary_label"], clf.predict_proba(both[features])[:, 1]))
        res["WUSTL_to_OnBody_per_attack_auc"] = per_attack

        out[name] = res

    # (5): single-feature AUC (attack vs benign) in each dataset; <0.5 means attacks have LOWER values
    wustl = cdg.zscore(cdg.load_wustl_aligned(), cdg.FEATURES)
    onbody = cdg.zscore(cdg.load_onbody_aligned(), cdg.FEATURES)
    out["single_feature_auc_attack_vs_benign"] = {
        "WUSTL": {f: float(roc_auc_score(wustl["binary_label"], wustl[f])) for f in cdg.FEATURES},
        "OnBody_all": {f: float(roc_auc_score(onbody["binary_label"], onbody[f])) for f in cdg.FEATURES},
    }
    out["class_prevalence"] = {"WUSTL_attack": float(wustl["binary_label"].mean()),
                               "OnBody_attack": float(onbody["binary_label"].mean())}

    path = RESULTS_DIR / "cross_dataset_diagnostics_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("Saved to", path)


if __name__ == "__main__":
    main()
