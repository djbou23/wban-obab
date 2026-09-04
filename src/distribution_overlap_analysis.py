"""E10 (strongly desirable experiment E10): quantify, per feature and per attack type,
how separable the attacked distribution is from the benign distribution -- Cohen's d
(standardized mean difference) and histogram overlap coefficient (fraction of shared
density). Substantiates reviewer Major Concern 3's point with numbers instead of just
narrative: which attacks are separable by construction (near-zero overlap on their
defining feature) versus genuinely close to the benign distribution (high overlap,
harder detection problem).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FEATURE_COLS = ["hr_mean", "hr_std", "ppg_mean", "ppg_std", "temp_mean",
                 "motion_energy", "tx_rate", "packet_loss_rate"]
ATTACKS = ["Biosignal_Spoofing", "Denial_of_Sleep", "Replay", "Simulated_Jamming"]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def histogram_overlap(a, b, n_bins=40):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if lo == hi:
        return 1.0
    bins = np.linspace(lo, hi, n_bins + 1)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    ha, hb = ha * bin_width, hb * bin_width  # normalize to sum to 1
    return float(np.minimum(ha, hb).sum())


def main():
    df = pd.read_parquet(DATA_DIR / "onbody_synthetic_benchmark.parquet")
    train = df[df["split"] == "train"]  # train split only, avoids peeking at test-split geometry
    benign = train[train["label"] == "Benign"]

    results = {}
    for attack in ATTACKS:
        attacked = train[train["label"] == attack]
        results[attack] = {}
        for col in FEATURE_COLS:
            a, b = attacked[col].dropna().values, benign[col].dropna().values
            results[attack][col] = {
                "cohens_d": round(cohens_d(a, b), 3),
                "histogram_overlap": round(histogram_overlap(a, b), 3),
            }
        print(f"\n=== {attack} vs Benign ===")
        for col, v in results[attack].items():
            print(f"  {col:18s} |d|={abs(v['cohens_d']):.2f}  overlap={v['histogram_overlap']:.2f}")

    out_path = RESULTS_DIR / "distribution_overlap_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
