# GitHub publication checklist

Before setting this repository to Public:

- [x] Both authors confirmed agreement to release the code, checkpoint weights, scalers, tables, predictions, and plots under the scoped licenses documented in this repository.
- [x] Add scoped licenses: MIT for original code/model artifacts and CC BY 4.0 for original documentation/research outputs, subject to third-party rights; see `LICENSE` and `LICENSE-DATA.md`.
- [ ] Before public release, re-check applicable Binance Data Vision terms and ensure the empirical data-source notice is current. Raw Kline archives are not hosted here.
- [ ] Review the manuscript's Data Availability statement and update it to the final GitHub URL and, preferably, a persistent DOI (for example, Zenodo integration) after publication.
- [ ] Run `python scripts/verify_repository.py` from a clean clone.
- [ ] Optionally rerun the Fixed-H smoke test with real CPU PyTorch. A full run is computationally intensive.
- [ ] Check GitHub's current per-file size limits; no individual file in this archive should approach the standard GitHub 100 MB limit.
- [ ] Ensure the repository does not accidentally include `features.npz`, raw exchange downloads, logs, local credentials, or workstation-specific paths.
- [ ] Add the final repository URL and DOI to `CITATION.cff` and the manuscript after the public archive is stable.
