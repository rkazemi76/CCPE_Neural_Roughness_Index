# A Neural Roughness Index for Cryptocurrency Volatility — Research Code and Results

This repository archives the computational materials supporting the manuscript by **Reza Kazemi** and **Shokoofeh Banihashemi**:

> *A Simulation-to-Estimation Framework for a Neural Roughness Index in Cryptocurrency Volatility: Synthetic Validation, Exploratory Empirical Forecasting, and Computational Evaluation* (CCPE manuscript version 0.25).

The repository is intended to make the reported simulations, empirical analyses, computational controls, and saved outputs inspectable. It is a research archive, not a production trading system.

## Repository contents

| Folder | Contents |
|---|---|
| `experiments/empirical/` | Five-asset rolling estimates, analysis scripts, statistical/forecasting output tables and figures, and the frozen W384 checkpoints used by the empirical estimator. |
| `experiments/fixed_h/` | Fixed-H synthetic sensitivity experiment, nine trained model checkpoints, scalers, full prediction-level output, protocol, and a portable runner. |
| `experiments/parallel_benchmark/` | The paired 2/4/6-worker CPU feature-generation benchmark code and its final 108 paired comparisons. |
| `experiments/smooth_semimartingale/` | Frozen-checkpoint OU smooth-semimartingale out-of-domain diagnostic and its evidence. |
| `docs/` | Reproduction notes, data/output descriptions, limitations, and release checklist. |
| `scripts/verify_repository.py` | Verifies the SHA-256 checksums for all files listed in `MANIFEST.sha256`. |

## Quick start

1. Download or clone this repository.
2. Read [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before running any experiment.
3. Install the requirements for the experiment you intend to run. The empirical, Fixed-H, parallel-benchmark, and OU-control components have separate dependency files.
4. From the corresponding experiment folder, run the command shown in its README or in the reproduction guide.
5. Verify the released files from the repository root:

```bash
python scripts/verify_repository.py
```

The archived `results/` directories contain the results used for manuscript reporting. Re-running scripts may be computationally expensive and some scripts download public market data. Recomputed outputs should be written to new directories rather than overwriting archived results.

## Important scope and interpretation notes

- The neural roughness index is calibrated and validated on synthetic rough-volatility generators. Its cryptocurrency application is exploratory and descriptive; it is **not** a direct observation or ground-truth measurement of latent market roughness.
- The OU experiment is an out-of-domain smooth-semimartingale stress diagnostic. It is not a calibrated smooth-versus-rough classifier, a formal test of the hypothesis `H = 0.5`, or a false-positive-rate estimate.
- The walk-forward forecasting results are model- and sample-specific. The hybrid HAR-RV plus neural-index model improves some reported RMSE/MAE point estimates, while the GARCH benchmark is better on pooled QLIKE in the reported comparison. No universal forecasting superiority is claimed.
- The parallel benchmark times synthetic path generation, 27-channel feature construction, and result transfer only. It does not time neural training/inference, GPU execution, a full market-data pipeline, multi-node execution, or production throughput.
- The Fixed-H full run's large `features.npz` array is intentionally not distributed in this GitHub archive. Its original hash is preserved in `experiments/fixed_h/results/reported_full/source_run_manifest.json`; the other run outputs and all required source inputs are present. The runner regenerates feature arrays locally when a new full run is performed.
- Raw Binance minute-level archives are not redistributed. The empirical source script downloads monthly Spot Klines from Binance Data Vision when they are not already cached. The rolling estimate CSVs used in the downstream analyses are included.

## Data and software licensing

A `LICENSE` file has not been added because the authors have not yet specified the license under which the code, checkpoints, and derived research outputs may be reused. See [`LICENSE_PENDING.md`](LICENSE_PENDING.md). Before treating this repository as an open-source or open-data release, the authors should select and add an appropriate license and confirm that all included materials may be redistributed.

## Citation

A citation metadata file is provided in [`CITATION.cff`](CITATION.cff). No DOI or public repository URL is asserted here; add those after the repository has been created and the archive has been deposited.

## Contact

For questions about the research archive, contact the corresponding author using the address in the manuscript.
