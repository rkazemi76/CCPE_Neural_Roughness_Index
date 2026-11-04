# Results map

## Empirical results (`experiments/empirical/results/`)

| Directory | Contents |
|---|---|
| `v2_16_regime_statistical_validation/` | Per-asset bootstrap/permutation distributions, regime summaries, transition statistics, and figures. |
| `v2_18_multi_horizon_incremental_information/` | 6-hour, 1-day, and 3-day incremental-information regressions, permutation null arrays, figures, and tables. |
| `v2_19_final_statistical_robustness/` | Common-timestamp panel, common-block and wild-cluster bootstrap arrays, leave-one-asset-out checks, figures, and summaries. |
| `v2_21_walkforward/` | HAR walk-forward prediction and paired-loss outputs. |
| `v2_22_constant_H/`, `v2_22_stochastic_H/` | Constant-H and stochastic-H synthetic benchmark outputs. |
| `v2_23_garch/` | GARCH walk-forward predictions, fit status, coverage, metrics, and paired bootstrap output. |
| `v2_24_*` | Constant-H dependence/three-seed checks and empirical variability null analyses. |
| `v2_25_*` | OOS block-length sensitivity and alternate initial training windows (20 and 25 timestamps). |
| `v2_26_variable_H_three_seed_common_test/` | Three-seed common-test metrics, predictions, and paired bootstrap output. |
| `ccpe_extended_oos_validation/` | Extended OOS validation metrics and report. |

The five processed rolling-estimate panels are in `experiments/empirical/inputs/results_hurst_v2_15_empirical_crypto_multiasset/`.

## Fixed-H (`experiments/fixed_h/results/reported_full/`)

Includes `design.csv`, `predictions.csv`, metric tables, paired contrasts, noiseless scale-control outputs, protocol, environment, test/status records, checkpoints, and original source-run manifest. The original `features.npz` is deliberately absent. `distributed_manifest.json` records the included output files and the omitted feature array's original hash.

## Parallel benchmark (`experiments/parallel_benchmark/evidence/`)

Worker-specific environment, raw repetitions, and summary CSVs are held under `workers_2/`, `workers_4/`, and `workers_6/`; consolidated 108-row and 12-configuration files are under `combined/`.

## Smooth semimartingale control (`experiments/smooth_semimartingale/evidence/smooth_semimartingale/`)

Contains protocol, feature parity and reproduction checks, frozen-model predictions, and summary statistics.
