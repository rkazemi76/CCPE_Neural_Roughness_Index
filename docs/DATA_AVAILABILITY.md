# Data availability

- The five processed BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, and XRPUSDT rolling-estimate CSV files used in the saved downstream analyses are included in the repository.
- Raw one-minute Binance Spot Kline ZIP files are not included. The v2.15 script retrieves them from Binance Data Vision when needed; see `experiments/empirical/DATA_SOURCES.md`.
- Synthetic model checkpoints, scalers, and Fixed-H prediction-level outputs are included.
- The Fixed-H full-run feature tensor `features.npz` is not included because it is approximately 700 MB. Its original hash is preserved in `experiments/fixed_h/results/reported_full/source_run_manifest.json`; a new full run regenerates it locally.
- The final CPU benchmark evidence and OU control outputs are included.

No claim is made that this repository redistributes the raw exchange data or that every saved result can be regenerated without access to the corresponding source market data and compatible software environment.
