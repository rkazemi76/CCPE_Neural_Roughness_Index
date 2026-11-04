# A Neural Roughness Index for Cryptocurrency Volatility — Research Code and Results

This repository archives the computational materials supporting the manuscript by **Reza Kazemi** and **Shokoofeh Banihashemi**:

> *A Neural Roughness Index for Cryptocurrency Volatility: Simulation and Empirical Evaluation*

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

The authors have agreed to the following scoped licensing arrangement:

- **Original Python source code and trained model artifacts** (including the
  released checkpoints and scalers): MIT License; see [`LICENSE`](LICENSE).
- **Original documentation and author-created research outputs** (such as
  explanatory material, original tables, figures, summaries, and analysis
  outputs): CC BY 4.0, to the extent the authors hold the relevant rights; see
  [`LICENSE-DATA.md`](LICENSE-DATA.md).
- **Third-party material and underlying source data** are not relicensed by
  these notices. In particular, Binance market data remains subject to the
  terms of its source service. See
  [`experiments/empirical/DATA_SOURCES.md`](experiments/empirical/DATA_SOURCES.md).

Where a file contains or reflects third-party material, the license applies
only to the authors' original contributions, if any, and not to that
third-party material.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The repository URL is recorded there. A DOI will be added after a versioned release is archived in Zenodo; no DOI is claimed yet.

## Contact

For questions about the research archive, contact the corresponding author using the address in the manuscript.
