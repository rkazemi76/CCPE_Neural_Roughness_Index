# GitHub publication checklist

Before setting this repository to Public:

- [ ] Confirm all authors agree to the public release of the code, checkpoint weights, scalers, tables, predictions, and plots.
- [ ] Select and add a license; `LICENSE_PENDING.md` is intentionally not a license grant.
- [ ] Confirm the current Binance Data Vision terms for any data use. Raw Kline archives are not hosted here.
- [ ] Review the manuscript's Data Availability statement and update it to the final GitHub URL and, preferably, a persistent DOI (for example, Zenodo integration) after publication.
- [ ] Run `python scripts/verify_repository.py` from a clean clone.
- [ ] Optionally rerun the Fixed-H smoke test with real CPU PyTorch. A full run is computationally intensive.
- [ ] Check GitHub's current per-file size limits; no individual file in this archive should approach the standard GitHub 100 MB limit.
- [ ] Ensure the repository does not accidentally include `features.npz`, raw exchange downloads, logs, local credentials, or workstation-specific paths.
- [ ] Add the final repository URL and DOI to `CITATION.cff` and the manuscript after the public archive is stable.
