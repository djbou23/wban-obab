# WBAN-OBAB: On-Body Attack Benchmark — Generation & Analysis Code

This repository contains the benchmark construction pipeline, sample-lineage audit
script, and the full set of analysis, ablation, and robustness-check scripts used to
produce the results in:

> D. E. Boubiche et al., "A WBAN-Specific Attack Benchmark for On-Body Medical Network Intrusion Detection,"
> [journal/venue and year to be added upon publication].

The benchmark dataset itself is published separately on IEEE DataPort:
**WBAN-OBAB**, DOI: [10.21227/5v75-b870](https://doi.org/10.21227/5v75-b870).

## Repository contents

- **`src/`** — all generation and analysis scripts, including:
  - `synthesize_onbody_attacks.py` — the benchmark construction pipeline (windowing,
    feature extraction, attack synthesis for all four attack scenarios: biosignal
    spoofing, replay, denial-of-sleep, simulated jamming).
  - `verify_lineage_leakage.py` — the sample-lineage audit script. This is the script
    that found and confirmed the fix for a cross-split leakage defect in the replay
    generator (see the paper, Section IV.C): an earlier implementation allowed 30.8%
    of replay samples to splice in donor content from the opposite train/test split.
    Any future extension of the attack taxonomy should be checked with this script.
  - Baseline training scripts (`train_baselines_*.py`), robustness and ablation
    scripts (`cross_subject_loso.py`, `feature_group_ablation.py`,
    `attack_pairwise_ablation.py`, `feature_permutation_test.py`,
    `replay_variants.py`, `window_length_sensitivity.py`,
    `distribution_overlap_analysis.py`, `simple_baselines.py`,
    `cross_dataset_generalization.py`, `cross_dataset_robustness.py`,
    `cross_dataset_robustness_v2.py`), efficiency profiling
    (`efficiency_profiling*.py`), and SHAP explainability scripts
    (`shap_analysis*.py`).
- **`results/`** — the complete experimental results referenced throughout the paper,
  as JSON files, one per experiment/script (e.g. `lineage_leakage_audit_results.json`,
  `cross_subject_loso_results.json`, `replay_variants_results.json`,
  `simple_baselines_results.json`, and so on).

## What is not in this repository

Raw and processed copies of CICIoMT2024 and WUSTL-EHMS-2020 are not redistributed
here; both remain subject to their original authors' access and citation terms (see
the paper's references). The on-body benchmark's raw physiological source
(PhysioNet PTT-PPG) and the processed benchmark itself are not duplicated here either
— the processed, ready-to-use benchmark with full lineage metadata is on IEEE
DataPort at the DOI above, alongside its own data dictionary and licensing
documentation (Open Database License, ODbL, matching the share-alike terms of its
physiological source).

## Reproducing the benchmark

```bash
python src/synthesize_onbody_attacks.py      # builds the on-body benchmark parquet
python src/verify_lineage_leakage.py         # audits it for cross-split leakage
python src/train_baselines_onbody.py         # classical baselines (RF, XGBoost)
python src/train_onbody_dl_subject_safe.py   # deep-learning baselines, subject-safe validation
```

Each of the robustness/ablation scripts in `src/` is independently runnable and
writes its own JSON result file to `results/`, matching the files already included
here.

## Dependencies

`pandas`, `numpy`, `scikit-learn`, `xgboost`, `torch`, `shap`.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The
benchmark dataset itself (on IEEE DataPort) carries the Open Database License (ODbL)
as required by its physiological source's share-alike terms — see the DataPort
listing for details.

## Citation

If you use this code or the benchmark, please cite both the paper and the dataset
(see above and the DataPort listing for the exact citation format).
