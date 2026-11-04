# Release notes — v1.0.0 archive assembly

- Consolidated the submitted empirical code/results and Fixed-H models/results ZIPs.
- Recovered the Fixed-H runner and required source modules from the earlier matching return package, and restored the source modules `matched_features.py`, `matched_retraining.py`, and `profile_stages_reference.py` from the source archive.
- Added W384 checkpoint and standardizer files used by the empirical v2.15 estimator from the v0.25 supplementary materials.
- Added the final i5 paired-benchmark worker manifests, repetition CSVs, and summary CSVs for 2/4/6 workers.
- Added the OU semimartingale control code, checkpoints, and output evidence from the v0.25 supplementary package.
- Removed workstation-specific absolute output paths from `experiments/fixed_h/reference/config.json`; only `config.out_dir` was changed to `runs/recomputed`. The original config SHA-256 is recorded in `CONFIG_NORMALIZATION.json` and the provenance entry.
- Replaced selected absolute or stale local-path defaults in the empirical analysis scripts with repository-relative paths. No model weights or archived result tables were changed.
- Excluded the 700 MB Fixed-H `features.npz`, retaining the original hash in the source-run manifest and creating a separate distributed-output manifest.
- Excluded old Xeon/obsolete computational-profile materials, the superseded standalone timing launcher from the empirical scripts folder, smoke-test outputs, pycache, logs, raw Binance downloads, and local virtual environments. The legacy benchmark module remains only in Fixed-H reference dependencies because its numerical definitions are used by the Fixed-H runner.

This is an archive assembly and portability pass. It does not independently rerun all empirical analyses or full Fixed-H inference.
