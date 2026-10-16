# WBAN-OBAB: On-Body Attack Benchmark — Generation & Analysis Code

This repository contains the benchmark construction pipeline, sample-lineage audit
script, and the full set of analysis, ablation, and robustness-check scripts used to
produce the results in:

> D. E. Boubiche et al., "A Physiologically Grounded Feature-Level Attack Benchmark for
> Intrusion Detection in On-Body Medical Networks,"
> [journal/venue and year to be added upon publication].

The benchmark dataset itself is published separately on IEEE DataPort:
**WBAN-OBAB**, DOI: [10.21227/5v75-b870](https://doi.org/10.21227/5v75-b870).

## Scope of the benchmark

WBAN-OBAB is a physiologically grounded, **feature-level** benchmark. Its physiological
features are computed from real wearable recordings (PhysioNet PTT-PPG, 22 healthy
subjects, 10-second windows), whereas its four attack scenarios (biosignal spoofing,
replay, denial-of-sleep, simulated jamming) and its network-proxy variables
(transmission rate and packet-loss rate, including those of benign windows) are
synthetic. It contains no recorded wireless communications, measured energy
consumption, or physically observed network attacks. It is intended as a controlled
testbed for detection methodology, not as a substitute for validation on physically
observed attacks or clinical populations.

How the paper's results should be read (each point is backed by a script below):

- **Denial-of-sleep and simulated jamming** are almost perfectly separable by
  construction, because their perturbations are large and prescribed directly
  (`attack_strength_sensitivity*.py` examines weaker settings).
- **Replay** detectability depends strongly on the assumed network-layer
  retransmission signature; with content-only replay it falls to or below chance for
  same-activity donors (`replay_variants.py`).
- **A decision tree and logistic regression** match or exceed the deep models, so the
  benchmark behaves as an engineered-feature tabular classification problem
  (`simple_baselines.py`, `feature_permutation_test.py`).
- **Cross-dataset transfer** between WUSTL-EHMS-2020 and the on-body benchmark collapses
  to constant predictions with below-chance ROC-AUC. The two datasets differ in attack
  definition, network-feature origin, and class prevalence, so this experiment does not
  isolate a WBAN-specific generalization effect and is not a graded measure of domain
  shift (`cross_dataset_diagnostics.py`).

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
  - `cross_dataset_diagnostics.py` — diagnostics for the WUSTL-EHMS-2020 <-> on-body
    transfer experiment (paper, Section VII.K, Table XIV): fraction of target samples
    predicted as attack, ROC-AUC, a prevalence-matched re-run, per-attack AUC, and
    single-feature AUC. It shows that the reported transfer macro-F1 values (0.167 and
    0.111) coincide with the macro-F1 of constant predictions given each target's class
    prevalence, and that the network-rate cue has opposite signs in the two datasets.
- **`results/`** — the complete experimental results referenced throughout the paper,
  as JSON files, one per experiment/script (e.g. `lineage_leakage_audit_results.json`,
  `cross_subject_loso_results.json`, `replay_variants_results.json`,
  `simple_baselines_results.json`, `cross_dataset_diagnostics_results.json`, and so on).

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
python src/cross_dataset_diagnostics.py      # transfer diagnostics (paper Table XIV)
```

The scripts read processed inputs from `data/processed/` (not included): the first
command writes `onbody_synthetic_benchmark.parquet` there, and the external datasets
must be obtained from their original sources. Each of the robustness/ablation
scripts in `src/` is independently runnable and writes its own JSON result file to
`results/`, matching the files already included here.

## Dependencies

`pandas`, `numpy`, `scikit-learn`, `xgboost`, `torch`, `shap`, and `pyarrow` (parquet
I/O). `scapy` is needed only for `extract_bluetooth_flows.py`.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The
benchmark dataset itself (on IEEE DataPort) carries the Open Database License (ODbL)
as required by its physiological source's share-alike terms — see the DataPort
listing for details.

## Citation

If you use this code or the benchmark, please cite both the paper and the dataset
(see above and the DataPort listing for the exact citation format).
