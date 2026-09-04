"""E1 (reviewer Major Concern 16 / essential experiment E1): prove that no source
physiological window contributes correlated descendants to both the train and test
split of the on-body benchmark, and report per-subject window counts (Major Concern 8).

Every row in onbody_synthetic_benchmark.parquet carries a `source_window_id` (the
captured 10-second window it was derived from) and, for Replay rows only, a
`replay_source_window_id` (the second, mismatched window whose HR/PPG content was
spliced in). This script checks both lineage sources against the train/test split.
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")

    report = {}

    # 1. Every source_window_id must map to exactly one subject and one split
    #    (guaranteed by construction: split is assigned by subject before attack
    #    generation, so all 5 siblings of a window inherit the same subject -> same split).
    grp = df.groupby("source_window_id").agg(n_subjects=("subject", "nunique"),
                                              n_splits=("split", "nunique"),
                                              n_rows=("label", "count"))
    report["source_window_groups"] = int(len(grp))
    report["rows_per_source_window_min_max"] = [int(grp["n_rows"].min()), int(grp["n_rows"].max())]
    report["source_windows_spanning_gt1_subject"] = int((grp["n_subjects"] > 1).sum())
    report["source_windows_spanning_gt1_split"] = int((grp["n_splits"] > 1).sum())

    # 2. Cross-split contamination via the REPLAY mechanism specifically: a replay row's
    #    spliced-in content (replay_source_window_id) could in principle originate from a
    #    window belonging to a DIFFERENT subject than the replay row itself. Check whether
    #    that donor window's subject/split ever crosses the replay row's own split.
    replay_rows = df[df["label"] == "Replay"].copy()
    window_to_subject = df.drop_duplicates("source_window_id").set_index("source_window_id")["subject"]
    window_to_split = df.drop_duplicates("source_window_id").set_index("source_window_id")["split"]
    replay_rows["donor_subject"] = replay_rows["replay_source_window_id"].map(window_to_subject)
    replay_rows["donor_split"] = replay_rows["replay_source_window_id"].map(window_to_split)
    cross_subject_replay = replay_rows["subject"] != replay_rows["donor_subject"]
    cross_split_donor = replay_rows["split"] != replay_rows["donor_split"]
    report["replay_rows_total"] = int(len(replay_rows))
    report["replay_rows_cross_subject_donor"] = int(cross_subject_replay.sum())
    report["replay_rows_cross_subject_donor_pct"] = round(100 * cross_subject_replay.mean(), 1)
    report["replay_rows_where_donor_split_differs_from_own_split"] = int(cross_split_donor.sum())

    # 3. Train/test subject sets are disjoint (structural guarantee) -- confirm explicitly.
    train_subjects = set(df[df["split"] == "train"]["subject"].unique())
    test_subjects = set(df[df["split"] == "test"]["subject"].unique())
    report["train_subjects"] = sorted(train_subjects, key=lambda s: int(s[1:]))
    report["test_subjects"] = sorted(test_subjects, key=lambda s: int(s[1:]))
    report["train_test_subject_overlap"] = sorted(train_subjects & test_subjects)

    # 4. Per-subject window counts (Concern 8): windows are counted once per source
    #    window (i.e. benign-window count per subject, since every window yields exactly
    #    one row of each of the 5 classes).
    per_subject = df[df["label"] == "Benign"].groupby("subject").size().to_dict()
    per_subject = {k: int(v) for k, v in sorted(per_subject.items(), key=lambda kv: int(kv[0][1:]))}
    report["source_windows_per_subject"] = per_subject
    report["source_windows_per_subject_min_max_mean"] = [
        min(per_subject.values()), max(per_subject.values()),
        round(sum(per_subject.values()) / len(per_subject), 1),
    ]
    report["n_subjects_total"] = len(per_subject)
    report["n_subjects_test"] = len(test_subjects)

    print(json.dumps(report, indent=2))
    out_path = RESULTS_DIR / "lineage_leakage_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
