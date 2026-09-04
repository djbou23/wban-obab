"""E2 / E9 (essential + strongly-desirable): the main benchmark's Replay attack draws
its spliced-in donor window from ANY different-activity window in the same split,
regardless of subject -- a specific adversarial model the review calls out (Major
Concern 4) as potentially testing "activity-context inconsistency detection" rather
than general replay detection. This constructs four explicit variants and evaluates
each SEPARATELY (binary Replay-vs-Benign, RandomForest, same subject-level split) so
the paper can report which kinds of replay are actually detectable with this feature
schema, rather than only the one variant the main benchmark happens to use:

  ss_diffact : same subject,  different activity  (closest to the main benchmark)
  cs_diffact : cross subject, different activity
  ss_sameact : same subject,  same activity, a temporally distant window
               (the classic "capture now, resend later" replay threat model)
  cs_sameact : cross subject, same activity

ss_sameact is the hardest and most realistic case: the attacker replays a genuinely
plausible window for the current context, so no feature in this schema (which carries
no timestamp/sequence-position information) can distinguish it from a fresh genuine
reading -- expected to be near chance, and that is an informative negative result in
its own right, not a bug.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import synthesize_onbody_attacks as syn

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FEATURE_COLS = ["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean", "motion_energy",
                 "tx_rate", "packet_loss_rate"]
RNG = np.random.default_rng(42)


def build_windows_with_split():
    records = syn.list_records()
    all_windows = []
    for subject, activity, path in records:
        df = pd.read_csv(path)
        n_windows = len(df) // syn.WINDOW_SAMPLES
        for w in range(n_windows):
            start, end = w * syn.WINDOW_SAMPLES, (w + 1) * syn.WINDOW_SAMPLES
            feat = syn.compute_window_features(df.iloc[start:end])
            all_windows.append({"subject": subject, "activity": activity, "window_idx": w, **feat})
    windows_df = pd.DataFrame(all_windows).dropna(subset=["hr_mean", "hr_std"]).reset_index(drop=True)

    subjects = sorted(windows_df["subject"].unique(), key=lambda s: int(s[1:]))
    test_subjects = set(RNG.choice(subjects, size=max(1, round(len(subjects) * 0.2)), replace=False))
    windows_df["split"] = windows_df["subject"].apply(lambda s: "test" if s in test_subjects else "train")
    return windows_df


VARIANTS = {
    "ss_diffact": lambda w, s, a, split: w[(w["subject"] == s) & (w["activity"] != a) & (w["split"] == split)],
    "cs_diffact": lambda w, s, a, split: w[(w["subject"] != s) & (w["activity"] != a) & (w["split"] == split)],
    "ss_sameact": lambda w, s, a, split: w[(w["subject"] == s) & (w["activity"] == a) & (w["split"] == split)],
    "cs_sameact": lambda w, s, a, split: w[(w["subject"] != s) & (w["activity"] == a) & (w["split"] == split)],
}


def make_replay_rows(windows_df, variant_fn, exclude_self_idx=True, network_signature=True):
    """network_signature=False strips apply_replay's synthetic tx_rate/packet_loss_rate
    perturbation, isolating the physiological-content-mismatch signal alone (no
    'duplicate retransmission' network-layer tell)."""
    rows = []
    for idx, row in windows_df.iterrows():
        subject, activity, split = row["subject"], row["activity"], row["split"]
        pool = variant_fn(windows_df, subject, activity, split)
        if exclude_self_idx:
            pool = pool[pool.index != idx]
        if len(pool) == 0:
            continue
        donor = pool.sample(1, random_state=RNG.integers(0, 1_000_000)).iloc[0]
        base_feat = row[["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean", "motion_energy"]].to_dict()
        donor_feat = donor[["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean", "motion_energy"]].to_dict()
        replayed = syn.apply_replay(base_feat, donor_feat)
        if not network_signature:
            replayed["tx_rate"] = row["tx_rate"]
            replayed["packet_loss_rate"] = row["packet_loss_rate"]
        rows.append({**replayed, "label": 1, "subject": subject, "split": split})
        # matching benign row from the same source window
        benign = dict(base_feat, tx_rate=row["tx_rate"], packet_loss_rate=row["packet_loss_rate"])
        rows.append({**benign, "label": 0, "subject": subject, "split": split})
    return pd.DataFrame(rows)


def main():
    windows_df = build_windows_with_split()
    # benign tx_rate/packet_loss_rate weren't part of build_windows_with_split; attach them
    windows_df["tx_rate"] = syn.NORMAL_TX_RATE * RNG.uniform(0.9, 1.1, size=len(windows_df))
    windows_df["packet_loss_rate"] = RNG.uniform(0.0, 0.02, size=len(windows_df))

    results = {}
    for network_signature in (True, False):
        tag = "with_network_signature" if network_signature else "content_only"
        for variant_name, variant_fn in VARIANTS.items():
            df = make_replay_rows(windows_df, variant_fn, network_signature=network_signature)
            train, test = df[df["split"] == "train"], df[df["split"] == "test"]
            clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
            clf.fit(train[FEATURE_COLS], train["label"])
            pred = clf.predict(test[FEATURE_COLS])
            f1 = f1_score(test["label"], pred, average="macro")
            key = f"{variant_name}__{tag}"
            results[key] = {
                "macro_f1_replay_vs_benign": float(f1),
                "n_train": int(len(train)), "n_test": int(len(test)),
                "variant": variant_name, "network_signature": network_signature,
            }
            print(f"{variant_name:12s} [{tag:23s}]: macro-F1 = {f1:.4f}  (train={len(train)}, test={len(test)})")

    out_path = RESULTS_DIR / "replay_variants_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
