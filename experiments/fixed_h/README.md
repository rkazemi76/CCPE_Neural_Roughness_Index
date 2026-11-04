# Fixed-H controlled synthetic diagnostic

This folder contains a new controlled synthetic sensitivity study, not historical reproduction, retraining of the paper's estimator, or validation of latent market H. The full design and interpretation limits are in `PROTOCOL.md`.

Use Python 3.11 and a real PyTorch installation. From this directory:

```bash
python -m pip install -r requirements.txt
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python run_experiment.py --threads 1
```

For an engineering smoke test only:

```bash
python run_experiment.py --smoke-only --threads 1
```

The runner verifies all source hashes, performs a neural smoke run, and only then proceeds to the full run. Each run is written to a fresh UTC-stamped subdirectory under `runs/`. Do not use Python optimization flags because assertions are part of the checks.

The approximately 700 MB `features.npz` array from the original reported run is intentionally not committed. Its original checksum is preserved in `results/reported_full/source_run_manifest.json`. The other reported outputs, all nine model checkpoints, all scalers, and all 41 required provenance-tracked source inputs are present. A new full run regenerates the feature arrays locally; the returned ZIP omits only that large array.
