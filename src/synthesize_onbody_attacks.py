"""Construct the novel on-body/WBAN-specific attack benchmark (Contribution 1 core).

Methodology (documented here so it's defensible in the paper's Section IV):

Real, honestly-derived physiological features (NOT fabricated):
  - Heart rate (mean/std) from R-R intervals via the 'peaks' R-peak annotation column.
  - PPG amplitude statistics (mean/std) across the 6 pleth channels.
  - Skin temperature (mean).
  - Motion energy (variance of 3-axis accelerometer magnitude) as an activity-level proxy.
We deliberately do NOT fabricate continuous SpO2/blood-pressure values to mimic
WUSTL-EHMS-2020's schema: PhysioNet PTT-PPG only provides SpO2 as a per-record
start/end summary in subjects_info.csv, not a continuous waveform, so inventing a
per-window SpO2 series would not be real data. This benchmark's feature schema is
therefore intentionally its own, not a forced match to WUSTL's.

Attack taxonomy (four classes, each with an explicit rationale for why it's
distinct from generic-IoT attacks in CICIoMT2024 / WUSTL-EHMS-2020):

  1. Biosignal spoofing: an attacker forges vitals without controlling the sensor
     itself -> physiological features are perturbed to implausible values (HR jumps
     >40 bpm within one window, or a near-flatline) while motion/temperature stay
     from real benign data (the attacker doesn't control those channels).
  2. Replay: a captured benign window from a *different* activity context is
     reinserted as if current -> physiological content is real and internally
     consistent, but mismatched with the surrounding activity-state context
     (e.g., a 'sit' HR/motion profile replayed during a 'run' sequence).
  3. Denial-of-sleep / energy-depletion: targets the radio/MAC layer of a
     battery-limited on-body node, not the sensor -> physiological features are
     untouched (real, benign); only a synthetic network-layer transmission-rate
     feature spikes far above the node's normal duty-cycled baseline.
  4. Simulated jamming (channel interference, not RF-transmitted): causes real
     packet loss/corruption -> physiological signal gaps are introduced
     (last-value-hold, a realistic jamming artifact) and a synthetic
     packet-loss-rate feature spikes.

Train/test split is done BY SUBJECT (not by window) to avoid leakage.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "PhysioNet_PTT-PPG"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500  # Hz
WINDOW_SEC = 10
WINDOW_SAMPLES = FS * WINDOW_SEC
PLETH_COLS = [f"pleth_{i}" for i in range(1, 7)]
RNG = np.random.default_rng(42)

# baseline "normal" transmission rate proxy (packets/window) for the denial-of-sleep contrast
NORMAL_TX_RATE = 10.0
DOS_TX_RATE_MULT = (15, 40)  # attack multiplies normal rate by this factor range


def list_records():
    files = sorted(RAW_DIR.glob("s*_*.csv"))
    records = []
    for f in files:
        stem = f.stem  # e.g. s10_run
        subject, activity = stem.split("_")
        records.append((subject, activity, f))
    return records


def compute_window_features(window: pd.DataFrame) -> dict:
    peak_idx = np.flatnonzero(window["peaks"].values)
    if len(peak_idx) >= 2:
        rr_intervals_s = np.diff(peak_idx) / FS
        hr_series = 60.0 / rr_intervals_s
        hr_mean, hr_std = float(np.mean(hr_series)), float(np.std(hr_series))
    else:
        hr_mean, hr_std = np.nan, np.nan

    pleth_vals = window[PLETH_COLS].values.astype(float)
    ppg_mean = float(np.mean(pleth_vals))
    ppg_std = float(np.std(pleth_vals))

    temp_mean = float(window[["temp_1", "temp_2", "temp_3"]].values.mean())

    acc_mag = np.sqrt(window["a_x"] ** 2 + window["a_y"] ** 2 + window["a_z"] ** 2)
    motion_energy = float(np.var(acc_mag))

    return {
        "hr_mean": hr_mean, "hr_std": hr_std,
        "ppg_mean": ppg_mean, "ppg_std": ppg_std,
        "temp_mean": temp_mean, "motion_energy": motion_energy,
    }


def apply_biosignal_spoofing(feat: dict) -> dict:
    f = dict(feat)
    if RNG.random() < 0.5:
        f["hr_mean"] = f["hr_mean"] + RNG.choice([-1, 1]) * RNG.uniform(40, 80)  # implausible jump
    else:
        f["hr_mean"] = RNG.uniform(0, 5)  # near-flatline
    f["hr_std"] = f["hr_std"] * RNG.uniform(0.0, 0.1)  # spoofed signal is unnaturally stable
    f["tx_rate"] = NORMAL_TX_RATE * RNG.uniform(0.8, 1.2)
    f["packet_loss_rate"] = RNG.uniform(0.0, 0.02)
    return f


def apply_replay(feat: dict, mismatched_feat: dict) -> dict:
    # Attacker replays captured HR/PPG content, but cannot fake the CURRENT physical
    # context (skin temp / motion) since that would require compromising the actual
    # sensor + environment, not just the data stream. Keeping current temp/motion
    # while swapping in mismatched HR/PPG is what makes this detectable in principle
    # (motion says "running", replayed HR/PPG says "sitting") -- and appropriately
    # difficult, since only two of six physiological features carry the tell.
    f = dict(feat)
    f["hr_mean"] = mismatched_feat["hr_mean"]
    f["hr_std"] = mismatched_feat["hr_std"]
    f["ppg_mean"] = mismatched_feat["ppg_mean"]
    f["ppg_std"] = mismatched_feat["ppg_std"]
    f["tx_rate"] = NORMAL_TX_RATE * RNG.uniform(0.9, 1.3)  # slight duplication signature
    f["packet_loss_rate"] = RNG.uniform(0.0, 0.02)
    return f


def apply_denial_of_sleep(feat: dict) -> dict:
    f = dict(feat)  # physiology untouched (real, benign)
    f["tx_rate"] = NORMAL_TX_RATE * RNG.uniform(*DOS_TX_RATE_MULT)
    f["packet_loss_rate"] = RNG.uniform(0.0, 0.05)
    return f


def apply_jamming(feat: dict) -> dict:
    f = dict(feat)
    dropout_frac = RNG.uniform(0.15, 0.5)
    # last-value-hold artifact approximated by inflating variance/uncertainty proxies
    f["hr_std"] = f["hr_std"] * (1 + dropout_frac * 3) if not np.isnan(f["hr_std"]) else f["hr_std"]
    f["ppg_std"] = f["ppg_std"] * (1 - dropout_frac)  # flatter signal during gaps
    f["tx_rate"] = NORMAL_TX_RATE * (1 - dropout_frac)
    f["packet_loss_rate"] = dropout_frac
    return f


def main():
    records = list_records()
    all_windows = []  # list of (subject, activity, feat_dict)

    for subject, activity, path in records:
        df = pd.read_csv(path)
        n_windows = len(df) // WINDOW_SAMPLES
        for w in range(n_windows):
            start, end = w * WINDOW_SAMPLES, (w + 1) * WINDOW_SAMPLES
            feat = compute_window_features(df.iloc[start:end])
            feat["tx_rate"] = NORMAL_TX_RATE * RNG.uniform(0.9, 1.1)
            feat["packet_loss_rate"] = RNG.uniform(0.0, 0.02)
            all_windows.append({"subject": subject, "activity": activity, **feat})
        print(f"{subject}_{activity}: {n_windows} windows")

    windows_df = pd.DataFrame(all_windows).dropna(subset=["hr_mean", "hr_std"])
    print(f"\nTotal benign windows (all subjects/activities): {len(windows_df)}")

    # Subject-level split is decided BEFORE any attack is generated (not after), and the
    # Replay donor pool below is restricted to windows within the SAME split as the row
    # being generated. Deciding the split afterward while letting Replay draw its
    # spliced-in HR/PPG content from the full cross-split pool would leak train-subject
    # physiological content into test-split Replay rows (and vice versa) even though the
    # row's OWN subject stays split-disjoint -- caught by the lineage audit in
    # verify_lineage_leakage.py (30.8% of Replay rows affected under the naive ordering).
    subjects = sorted(windows_df["subject"].unique(), key=lambda s: int(s[1:]))
    n_test_subjects = max(1, round(len(subjects) * 0.2))
    test_subjects = set(RNG.choice(subjects, size=n_test_subjects, replace=False))
    windows_df["split"] = windows_df["subject"].apply(lambda s: "test" if s in test_subjects else "train")

    rows = []
    for idx, row in windows_df.iterrows():
        base_feat = row[["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean", "motion_energy"]].to_dict()
        subject, activity, row_split = row["subject"], row["activity"], row["split"]
        # Every sibling row generated below (benign + 4 attack variants) shares this id:
        # they all derive from the SAME captured 10-second window. This is recorded
        # explicitly so lineage/leakage can be audited directly from the released
        # parquet, rather than only being inferable from matching feature values.
        source_window_id = f"{subject}_{activity}_{idx}"
        common = {"subject": subject, "activity": activity, "split": row_split, "source_window_id": source_window_id}

        # Benign
        benign = dict(base_feat, tx_rate=row["tx_rate"], packet_loss_rate=row["packet_loss_rate"])
        rows.append({**benign, "label": "Benign", "category": "Benign", **common, "replay_source_window_id": None})

        # Spoofing
        rows.append({**apply_biosignal_spoofing(base_feat), "label": "Biosignal_Spoofing",
                     "category": "OnBody_Attack", **common, "replay_source_window_id": None})

        # Replay: pull a random window from a DIFFERENT activity, same or different
        # subject, but constrained to the SAME split as this row (see note above --
        # this is what prevents Replay content from crossing the train/test boundary).
        mismatched_pool = windows_df[(windows_df["activity"] != activity) & (windows_df["split"] == row_split)]
        if len(mismatched_pool) > 0:
            mismatched_idx = mismatched_pool.sample(1, random_state=RNG.integers(0, 1_000_000)).index[0]
            mismatched = mismatched_pool.loc[mismatched_idx]
            mismatched_feat = mismatched[["hr_mean", "hr_std", "ppg_mean", "ppg_std",
                                           "temp_mean", "motion_energy"]].to_dict()
            replay_source_id = f"{mismatched['subject']}_{mismatched['activity']}_{mismatched_idx}"
            rows.append({**apply_replay(base_feat, mismatched_feat), "label": "Replay",
                         "category": "OnBody_Attack", **common, "replay_source_window_id": replay_source_id})

        # Denial-of-sleep
        rows.append({**apply_denial_of_sleep(base_feat), "label": "Denial_of_Sleep",
                     "category": "OnBody_Attack", **common, "replay_source_window_id": None})

        # Jamming
        rows.append({**apply_jamming(base_feat), "label": "Simulated_Jamming",
                     "category": "OnBody_Attack", **common, "replay_source_window_id": None})

    full_df = pd.DataFrame(rows)

    print(f"\nTotal samples: {len(full_df)}")
    print(full_df["label"].value_counts())
    print(f"\nTest subjects ({len(test_subjects)}): {sorted(test_subjects)}")
    print(full_df.groupby("split").size())

    out_path = OUT_DIR / "onbody_synthetic_benchmark.parquet"
    full_df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
