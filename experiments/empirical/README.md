# Empirical cryptocurrency analysis

This folder contains the source scripts and output archives for the five-asset empirical component. The processed five-asset v2.15 rolling-estimate panel and `config.json` are included under `inputs/results_hurst_v2_15_empirical_crypto_multiasset/`. The frozen W384 checkpoint/standardizer pairs for seeds 42, 123, and 2024 are in `results_hurst_v2_13_effective_memory_saturation/seed_*/`. Saved downstream results are in `results/`.

## Environment

Use Python 3.11 where possible. Install:

```bash
python -m pip install -r requirements.txt
```

Install a platform-appropriate PyTorch build separately (CPU is sufficient for inference). The reported outputs are archived; not every analysis needs to be rerun to inspect them.

## Re-run the empirical estimator (requires data access)

From this folder:

```bash
python code/HurstNeuralEstimator_v2_15_empirical_crypto_multiasset.py \
  --model_root results_hurst_v2_13_effective_memory_saturation \
  --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT \
  --start_date 2024-01-01 --end_date 2026-09-01 \
  --context_length 384 --fine_bars_per_observation 64 \
  --out_dir results/recomputed_v2_15
```

The script may download monthly Spot Kline archives; see `DATA_SOURCES.md`. Existing derived inputs and reference outputs are retained unchanged.

## Downstream analyses

The analysis scripts are in `code/`. The default v2.16–v2.23 input root has been made portable and points to `inputs/results_hurst_v2_15_empirical_crypto_multiasset/`. The saved result directories are grouped by script version in `results/`. Recomputed outputs should use new `--out_dir` paths to avoid overwriting the archived outputs.

Examples:

```bash
python code/HurstNeuralEstimator_v2_16_regime_statistical_validation.py --out_dir results/recomputed_v2_16
python code/HurstNeuralEstimator_v2_17_conditional_incremental_information.py --out_dir results/recomputed_v2_17
python code/HurstNeuralEstimator_v2_18_multi_horizon_incremental_information.py --v215_script code/HurstNeuralEstimator_v2_15_empirical_crypto_multiasset.py --out_dir results/recomputed_v2_18
python code/HurstNeuralEstimator_v2_19_final_statistical_robustness.py --out_dir results/recomputed_v2_19
python code/HurstNeuralEstimator_v2_21_walkforward_HAR_benchmark_robust.py --v215_script code/HurstNeuralEstimator_v2_15_empirical_crypto_multiasset.py --out_dir results/recomputed_v2_21
python code/HurstNeuralEstimator_v2_23_walkforward_GARCH_benchmark.py --v215_script code/HurstNeuralEstimator_v2_15_empirical_crypto_multiasset.py --out_dir results/recomputed_v2_23
```

The walk-forward HAR/GARCH scripts reconstruct future realized-volatility targets and therefore require the Binance minute-data cache or network access.

## Portability edits

A small number of historical default path strings in analysis scripts referred to the author's local workstation directory. Those defaults were changed to paths relative to this folder. The numerical methods, output CSVs, and underlying saved result files were not changed. A record of the edits is in `docs/RELEASE_NOTES.md`.
