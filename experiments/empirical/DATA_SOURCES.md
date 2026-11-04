# Empirical market-data source

The raw one-minute market-price archive is not included in this repository. The v2.15 empirical script downloads monthly **Binance Spot Klines** for the selected symbol and interval from Binance Data Vision using the URL pattern:

`https://data.binance.vision/data/spot/monthly/klines/<SYMBOL>/<INTERVAL>/<SYMBOL>-<INTERVAL>-<YYYY-MM>.zip`

The manuscript's empirical window is 2024-01-01 through 2026-09-01 (UTC), for BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, and XRPUSDT, using one-minute bars. The retrieval script handles the timestamp-unit change in the archive from milliseconds to microseconds in later files. Network access is required only when the monthly ZIP is not already present in the local download cache.

The five processed rolling-estimate CSV files used by the downstream statistical and forecasting analyses are included under `inputs/results_hurst_v2_15_empirical_crypto_multiasset/`. These are derived analysis inputs, not raw exchange archives.

Users should review Binance Data Vision's current terms and any applicable data-use requirements before redistribution or downstream commercial use. This repository does not rehost the raw exchange files.
