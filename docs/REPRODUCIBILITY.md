# Reproducibility guide

## Environment and working directories

Use Python 3.11 if practical. Install dependencies separately for the component being rerun; do not install every experiment's requirements into one environment unless you have checked version compatibility.

### Empirical estimator and downstream analysis

```bash
cd experiments/empirical
python -m pip install -r requirements.txt
# Install a compatible PyTorch CPU/GPU build separately.
python code/HurstNeuralEstimator_v2_15_empirical_crypto_multiasset.py --model_root results_hurst_v2_13_effective_memory_saturation --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT --start_date 2024-01-01 --end_date 2026-09-01 --context_length 384 --fine_bars_per_observation 64 --out_dir results/recomputed_v2_15
```

This first-stage estimator needs public Binance Spot Kline archives (downloaded if not cached); see `experiments/empirical/DATA_SOURCES.md`. The downstream v2.16–v2.23 scripts work from the included processed panel at `inputs/results_hurst_v2_15_empirical_crypto_multiasset/`. For all recomputations, specify a new output directory. The repository's existing `results/` folder is the archived reference output and should not be overwritten.

Some later analysis outputs are computed from already saved rolling estimates; the forecasting scripts that reconstruct realized-volatility targets need the corresponding minute-level data cache or internet access. The script `code/ccpe_validate_extended_oos.py` validates selected archived OOS forecasts against saved metrics and requires only NumPy and pandas.

### Fixed-H synthetic diagnostic

```bash
cd experiments/fixed_h
python -m pip install -r requirements.txt
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python run_experiment.py --threads 1
```

The runner performs input-hash validation, a genuine PyTorch smoke run, and then the full run only after smoke passes. Use `--smoke-only` for an engineering check, not for reported results. It requires a fresh output directory for each run and must not be started with `-O` or `PYTHONOPTIMIZE`.

The large `features.npz` file from the archived reported run is not in GitHub. Its original SHA-256 is retained in `results/reported_full/source_run_manifest.json`. The output array will be regenerated locally during a new full run; the compact return archive produced by the runner omits that file.

### Parallel feature benchmark

```bash
cd experiments/parallel_benchmark
python -m pip install -r code/requirements.txt
# Run the three commands in this folder's README for 2, 4, and 6 workers.
```

The timing results are hardware-dependent. The archived 108 paired comparisons are the reported benchmark evidence; reruns on other hardware are not expected to have identical times.

### OU semimartingale diagnostic

```bash
cd experiments/smooth_semimartingale
python -m pip install -r code/requirements-smooth-control.txt
python code/run_smooth_semimartingale_control.py --out evidence/smooth_semimartingale_reproduced
```

The OU outputs are out-of-domain frozen-model diagnostics, not calibrated classification performance.

## Integrity verification

From the repository root:

```bash
python scripts/verify_repository.py
```

This checks every file listed in `MANIFEST.sha256`. It also checks that the large Fixed-H `features.npz` has not accidentally been added to this GitHub package. The Fixed-H experiment contains its own separate `SOURCE_PROVENANCE.json` and `RELEASE_MANIFEST.json` for inputs required by its runner.
