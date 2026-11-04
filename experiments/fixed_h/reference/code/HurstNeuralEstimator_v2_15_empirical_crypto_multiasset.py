#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HurstNeuralEstimator_v2_15_empirical_crypto_multiasset
============================================

Purpose
-------
Empirical validation of the frozen stochastic-H neural estimator on real
cryptocurrency market data, without fitting the H target on real data.

Scientific design
-----------------
1) Load the already-trained v2.13 context=384 checkpoints and standardizers
   for seeds 42, 123, 2024.
2) Download public Binance Spot 1-minute BTCUSDT klines from the official
   Binance public-data archive (monthly files).
3) Construct one coarse observation from 64 consecutive 1-minute returns.
   Within each coarse observation, compute multiscale realized variance at
   M = [8, 16, 32, 64], matching the synthetic estimator's finest-resolution
   ratio Mmax/M.
4) For each rolling context of 384 coarse observations (24,576 minutes,
   about 17.1 days), recompute the exact v2.1-fix2 / v2.13 robust
   representation using only that tail.
5) Apply the frozen synthetic models. No real-data target is used for training
   or calibration.
6) Produce two classical descriptive roughness benchmarks on the same
   log-RV proxy: a log-variogram slope estimator and a DFA estimator.
7) Quantify uncertainty using synthetic-seed ensemble dispersion and moving
   block bootstrap confidence intervals for empirical summaries/correlations.
8) Primary inference uses non-overlapping windows (stride=context_length) so
   the main empirical panel is not dominated by 99% overlapping windows.
9) Optionally relate estimated roughness to future 1-day realized volatility
   as an empirical relevance check. This is descriptive/predictive, not causal.

Important limitation
--------------------
The classical benchmarks are NOT treated as ground truth. They are alternative
empirical estimators of roughness on an observed log-RV proxy. The neural
estimator itself is zero-shot on real data and inherits the synthetic training
DGP and observation assumptions of v2.7/v2.13.

Default empirical assets
------------------------
BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, 2024-01-01 through 2026-09-01
UTC, 1-minute Spot klines.

Example
-------
python HurstNeuralEstimator_v2_15_empirical_crypto_multiasset.py ^
  --model_root results_hurst_v2_13_effective_memory_saturation ^
  --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ^
  --start_date 2024-01-01 --end_date 2026-09-01 ^
  --context_length 384 --fine_bars_per_observation 64 ^
  --stride_observations 384 ^
  --benchmark_scales 1 2 4 8 16 32 64 ^
  --bootstrap 1000 --bootstrap_block 4 ^
  --out_dir results_hurst_v2_15_empirical_crypto_multiasset

Web-data source note
--------------------
Binance documents public, unauthenticated market-data endpoints and its
public-data archive. Monthly Spot kline files are available at
https://data.binance.vision/data/spot/monthly/klines/<SYMBOL>/<INTERVAL>/...
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn


GEN_REAL = "real_crypto"
REP_ROBUST = "multiscale_rv_robust"


@dataclass
class Config:
    model_root: str = "results_hurst_v2_13_effective_memory_saturation"
    symbols: Tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT")
    interval: str = "1m"
    start_date: str = "2024-01-01"
    end_date: str = "2026-09-01"

    context_length: int = 384
    fine_bars_per_observation: int = 64
    resolutions: Tuple[int, ...] = (8, 16, 32, 64)
    stride_observations: int = 384
    rolling_diagnostic_stride: int = 0  # 0 disables high-frequency rolling diagnostic
    min_valid_windows: int = 20

    temporal_lag: int = 1
    roughness_lag: int = 4
    smooth_window: int = 4

    benchmark_scales: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    dfa_scales: Tuple[int, ...] = (8, 16, 32, 64)

    future_rv_minutes: int = 1440
    bootstrap: int = 500
    bootstrap_block: int = 4
    max_bootstrap_n: int = 100000

    inference_batch_size: int = 16
    http_timeout: int = 60
    download_dir: str = "binance_cache_v2_15"
    out_dir: str = "results_hurst_v2_15_empirical_crypto_multiasset"

    h_min: float = 0.05
    h_max: float = 0.50
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.15

    device: str = "auto"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_date_utc(s: str) -> pd.Timestamp:
    ts = pd.Timestamp(s, tz="UTC")
    return ts


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    cur = start.to_period("M").to_timestamp().tz_localize("UTC")
    out = []
    while cur < end:
        out.append(cur)
        cur = (cur + pd.offsets.MonthBegin(1)).tz_convert("UTC")
    return out


def official_month_url(symbol: str, interval: str, month: pd.Timestamp) -> str:
    ym = month.strftime("%Y-%m")
    return (
        f"https://data.binance.vision/data/spot/monthly/klines/"
        f"{symbol.upper()}/{interval}/{symbol.upper()}-{interval}-{ym}.zip"
    )


def safe_download(url: str, out_path: Path, timeout: int = 60) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 HurstNeuralEstimator-v2.15"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(out_path, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception:
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        raise


def read_binance_month_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file inside {zip_path}")
        name = csv_names[0]
        with zf.open(name) as fh:
            df = pd.read_csv(
                fh,
                header=None,
                usecols=[0, 4],
                names=["open_time", "close"],
                dtype={"open_time": "int64", "close": "float64"},
            )
    # Spot archive timestamps changed from ms to microseconds from 2025 onward.
    ot = df["open_time"].to_numpy(np.int64)
    if len(ot) and int(np.nanmedian(ot)) > 10**14:
        ot = ot // 1000
    df["timestamp"] = pd.to_datetime(ot, unit="ms", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype(np.float64)
    df = df[["timestamp", "close"]].dropna()
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return df.reset_index(drop=True)


def load_symbol_1m(symbol: str, cfg: Config) -> pd.DataFrame:
    start = parse_date_utc(cfg.start_date)
    end = parse_date_utc(cfg.end_date)
    cache = Path(cfg.download_dir) / symbol.upper() / cfg.interval
    frames = []
    months = month_starts(start, end)
    print(f"  {symbol}: downloading/loading {len(months)} monthly archive(s)")
    for i, m in enumerate(months, 1):
        url = official_month_url(symbol, cfg.interval, m)
        path = cache / Path(url).name
        print(f"    [{i:02d}/{len(months)}] {path.name}")
        try:
            safe_download(url, path, timeout=cfg.http_timeout)
            df = read_binance_month_zip(path)
            df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
            if not df.empty:
                frames.append(df)
        except urllib.error.HTTPError as exc:
            # Missing archive is fatal only when the entire requested period is missing.
            print(f"      WARNING: HTTP {exc.code} for {url}; skipping month")
        except Exception as exc:
            print(f"      WARNING: failed to read {path.name}: {exc}; skipping month")

    if not frames:
        raise RuntimeError(
            f"No usable Binance data found for {symbol}. Check symbol/date range/network."
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Frozen v2.1-fix2 / v2.13 model and representation
# -----------------------------------------------------------------------------

class HurstLSTM(nn.Module):
    def __init__(self, input_size: int, cfg: Config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        h = cfg.hidden_size * 2
        self.head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, 1),
        )
        self.h_min = cfg.h_min
        self.h_max = cfg.h_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        pooled = torch.cat([h.mean(dim=1), h[:, -1, :]], dim=1)
        raw = self.head(pooled).squeeze(-1)
        return self.h_min + (self.h_max - self.h_min) * torch.sigmoid(raw)


def load_checkpoint_bundle(model_root: Path, seed: int, context_length: int, device: torch.device):
    seed_dir = model_root / f"seed_{seed}"
    model_path = seed_dir / f"model_state_W{context_length}.pt"
    std_path = seed_dir / f"standardizer_W{context_length}.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not std_path.exists():
        raise FileNotFoundError(f"Missing standardizer: {std_path}")

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")

    saved_cfg = ckpt.get("config", {})
    model_cfg = Config(
        h_min=float(saved_cfg.get("h_min", 0.05)),
        h_max=float(saved_cfg.get("h_max", 0.50)),
        hidden_size=int(saved_cfg.get("hidden_size", 64)),
        num_layers=int(saved_cfg.get("num_layers", 2)),
        dropout=float(saved_cfg.get("dropout", 0.15)),
    )
    rep = ckpt.get("representation", REP_ROBUST)
    if rep != REP_ROBUST:
        raise ValueError(f"Checkpoint representation is {rep}, expected {REP_ROBUST}")

    state = ckpt["state_dict"]
    input_size = int(state["lstm.weight_ih_l0"].shape[1])
    model = HurstLSTM(input_size, model_cfg)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    z = np.load(std_path)
    mean = np.asarray(z["mean"], np.float32)
    std = np.asarray(z["std"], np.float32)
    return model, mean, std


def causal_ma_batch(x: np.ndarray, w: int) -> np.ndarray:
    # x: (B, W, K), causal moving average along W
    if w <= 1:
        return x.copy()
    B, T, K = x.shape
    cs = np.cumsum(x, axis=1, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float32)
    for t in range(T):
        lo = max(0, t - w + 1)
        if lo == 0:
            out[:, t, :] = cs[:, t, :] / float(t + 1)
        else:
            out[:, t, :] = (cs[:, t, :] - cs[:, lo - 1, :]) / float(t - lo + 1)
    return out


def robust_representation_batch(levels: np.ndarray, cfg: Config) -> np.ndarray:
    """Exact v2.1-fix2/v2.13 robust representation, batched."""
    B, T, K = levels.shape
    centered = levels - levels.mean(axis=1, keepdims=True)
    local_scale = np.maximum(centered.std(axis=2), 1e-6)[:, :, None]

    def lagdiff(x: np.ndarray, lag: int) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.float32)
        if lag < T:
            out[:, lag:, :] = x[:, lag:, :] - x[:, :-lag, :]
        return out

    d1 = lagdiff(centered, int(cfg.temporal_lag))
    d4 = lagdiff(centered, int(cfg.roughness_lag))
    ma = causal_ma_batch(centered, int(cfg.smooth_window))
    nd1 = d1 / local_scale
    nd4 = d4 / local_scale

    cols = []
    for j in range(K):
        cols.append(centered[:, :, j:j+1])
    for j in range(K - 1):
        cols.append((levels[:, :, j:j+1] - levels[:, :, j+1:j+2]))
    for j in range(K):
        cols.append(d1[:, :, j:j+1])
    for j in range(K):
        cols.append(d4[:, :, j:j+1])
    for j in range(K):
        cols.append(ma[:, :, j:j+1])
    for j in range(K):
        cols.append(nd1[:, :, j:j+1])
    for j in range(K):
        cols.append(nd4[:, :, j:j+1])
    return np.concatenate(cols, axis=2).astype(np.float32)


# -----------------------------------------------------------------------------
# Real-market multiscale RV construction
# -----------------------------------------------------------------------------

def build_coarse_blocks(df: pd.DataFrame, cfg: Config):
    bars = int(cfg.fine_bars_per_observation)
    if max(cfg.resolutions) != bars:
        raise ValueError("For exact v2.13 matching, fine_bars_per_observation must equal max(resolutions)=64.")
    if any(bars % M != 0 for M in cfg.resolutions):
        raise ValueError("fine_bars_per_observation must be divisible by every resolution.")

    ts = df["timestamp"].to_numpy()
    close = df["close"].to_numpy(np.float64)
    logp = np.log(np.maximum(close, 1e-300))
    r = np.diff(logp)
    r_ts = ts[1:]

    # Split into contiguous one-minute runs, because silently bridging data gaps
    # would manufacture false long-memory/roughness.
    step_ns = 60 * 10**9
    dtns = np.diff(r_ts.astype("datetime64[ns]").astype(np.int64))
    breaks = np.flatnonzero(dtns != step_ns) + 1
    bounds = np.r_[0, breaks, len(r)]

    level_chunks = []
    block_ts_chunks = []
    block_end_return_index_chunks = []
    block_run_id_chunks = []

    run_id = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        run_r = r[a:b]
        run_ts = r_ts[a:b]
        n_blocks = len(run_r) // bars
        if n_blocks < cfg.context_length + 2:
            continue
        use = n_blocks * bars
        rr = run_r[:use].reshape(n_blocks, bars)
        bt = run_ts[bars - 1:use:bars]

        levels = []
        for M in cfg.resolutions:
            nsub = bars // M
            agg = rr.reshape(n_blocks, M, nsub).sum(axis=2)
            rv = np.sum(agg ** 2, axis=1)
            levels.append(np.maximum(rv, 1e-18))
        lv = np.log(np.stack(levels, axis=1)).astype(np.float32)
        level_chunks.append(lv)
        block_ts_chunks.append(bt)
        # Global return-index of each block endpoint (for future RV lookup).
        block_end_return_index_chunks.append(np.arange(a + bars - 1, a + use, bars, dtype=np.int64))
        block_run_id_chunks.append(np.full(n_blocks, run_id, dtype=np.int32))
        run_id += 1

    if not level_chunks:
        raise RuntimeError("No contiguous segment is long enough for the requested context.")

    levels = np.concatenate(level_chunks, axis=0)
    block_ts = np.concatenate(block_ts_chunks, axis=0)
    block_end_idx = np.concatenate(block_end_return_index_chunks, axis=0)
    block_run_id = np.concatenate(block_run_id_chunks, axis=0)

    # Return series for future realized variance. Reconstruct once, aligned to
    # the input df's return timestamps.
    return {
        "levels": levels,
        "block_ts": block_ts,
        "block_end_return_index": block_end_idx,
        "block_run_id": block_run_id,
        "return_timestamps": r_ts,
        "returns": r.astype(np.float64),
    }


def rolling_window_indices(run_ids: np.ndarray, W: int, stride: int) -> np.ndarray:
    """Return endpoint indices whose entire W-window stays inside one contiguous run."""
    ids = np.asarray(run_ids, np.int32)
    out = []
    stride = max(1, int(stride))
    for rid in np.unique(ids):
        pos = np.flatnonzero(ids == rid)
        if len(pos) < W:
            continue
        # Runs were concatenated in order, so position 0..len(pos)-1 maps to the
        # corresponding global endpoint indices within that contiguous segment.
        out.extend(pos[W - 1::stride].tolist())
    return np.asarray(out, dtype=np.int64)


# -----------------------------------------------------------------------------
# Classical benchmarks
# -----------------------------------------------------------------------------

def variogram_h(window_logrv: np.ndarray, scales: Sequence[int]) -> float:
    x = np.asarray(window_logrv, np.float64)
    vals = []
    used = []
    for lag in scales:
        lag = int(lag)
        if lag <= 0 or lag >= len(x):
            continue
        d = x[lag:] - x[:-lag]
        v = float(np.mean(d * d))
        if np.isfinite(v) and v > 1e-14:
            used.append(float(lag)); vals.append(v)
    if len(vals) < 3:
        return np.nan
    slope = np.polyfit(np.log(used), np.log(vals), 1)[0]
    return float(0.5 * slope)


def dfa_h(window_logrv: np.ndarray, scales: Sequence[int]) -> float:
    x = np.asarray(window_logrv, np.float64)
    x = x - np.mean(x)
    y = np.cumsum(x)
    Fs, ss = [], []
    for s in scales:
        s = int(s)
        if s < 4 or s >= len(x) // 2:
            continue
        nseg = len(y) // s
        if nseg < 2:
            continue
        y2 = y[: nseg * s].reshape(nseg, s)
        t = np.arange(s, dtype=np.float64)
        A = np.vstack([t, np.ones(s)]).T
        # Closed-form per-segment linear detrend.
        AtA_inv = np.linalg.inv(A.T @ A)
        beta = (y2 @ A) @ AtA_inv  # (nseg,2)
        trend = beta[:, 0:1] * t[None, :] + beta[:, 1:2]
        F = np.sqrt(np.mean((y2 - trend) ** 2, axis=1))
        F = float(np.sqrt(np.mean(F ** 2)))
        if np.isfinite(F) and F > 1e-14:
            ss.append(float(s)); Fs.append(F)
    if len(Fs) < 3:
        return np.nan
    slope = np.polyfit(np.log(ss), np.log(Fs), 1)[0]
    return float(slope)


# -----------------------------------------------------------------------------
# Bootstrap / statistics
# -----------------------------------------------------------------------------

def corr_safe(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    x = np.asarray(a, np.float64); y = np.asarray(b, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if len(x) < 3 or np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return np.nan
    if method == "spearman":
        rx = pd.Series(x).rank(method="average").to_numpy()
        ry = pd.Series(y).rank(method="average").to_numpy()
        return float(np.corrcoef(rx, ry)[0, 1])
    return float(np.corrcoef(x, y)[0, 1])


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    block = max(1, min(block, n))
    starts = rng.integers(0, n, size=math.ceil(n / block))
    out = []
    for s in starts:
        out.extend(((s + np.arange(block)) % n).tolist())
        if len(out) >= n:
            break
    return np.asarray(out[:n], dtype=np.int64)


def bootstrap_summary(df: pd.DataFrame, cfg: Config, seed: int) -> Dict[str, float]:
    if df.empty:
        return {}
    rng = np.random.default_rng(seed)
    n = len(df)
    reps = int(cfg.bootstrap)
    if n > cfg.max_bootstrap_n:
        # Deterministic thinning for very long empirical panels.
        keep = np.linspace(0, n - 1, cfg.max_bootstrap_n).astype(int)
        d = df.iloc[keep].reset_index(drop=True)
    else:
        d = df
    n = len(d)

    series = {
        "mean_neural_H": d["H_neural_mean"].to_numpy(float),
        "mean_variogram_H": d["H_variogram"].to_numpy(float),
        "mean_dfa_H": d["H_dfa"].to_numpy(float),
        "corr_neural_variogram": (d["H_neural_mean"].to_numpy(float), d["H_variogram"].to_numpy(float)),
        "corr_neural_dfa": (d["H_neural_mean"].to_numpy(float), d["H_dfa"].to_numpy(float)),
        "corr_neural_future_logrv1d": (d["H_neural_mean"].to_numpy(float), d["future_log_rv_1d"].to_numpy(float)),
        "corr_neural_future_delta_logrv1d": (d["H_neural_mean"].to_numpy(float), d["future_delta_log_rv_1d"].to_numpy(float)),
    }
    acc = {k: [] for k in series}
    for _ in range(reps):
        ids = moving_block_indices(n, cfg.bootstrap_block, rng)
        for k, v in series.items():
            if isinstance(v, tuple):
                acc[k].append(corr_safe(v[0][ids], v[1][ids]))
            else:
                z = v[ids]
                z = z[np.isfinite(z)]
                acc[k].append(float(np.mean(z)) if len(z) else np.nan)

    out = {}
    for k, vals in acc.items():
        a = np.asarray(vals, float)
        a = a[np.isfinite(a)]
        if len(a):
            out[f"{k}_bootstrap_mean"] = float(np.mean(a))
            out[f"{k}_ci_low"] = float(np.quantile(a, 0.025))
            out[f"{k}_ci_high"] = float(np.quantile(a, 0.975))
    return out



# -----------------------------------------------------------------------------
# Coverage and empirical panel utilities
# -----------------------------------------------------------------------------

def summarize_data_coverage(df1m: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, object]:
    if df1m.empty:
        return {
            "bars_1m": 0,
            "expected_minutes": int((end - start).total_seconds() // 60),
            "observed_minutes": 0,
            "coverage_fraction": 0.0,
            "n_contiguous_runs": 0,
            "largest_contiguous_run_minutes": 0,
        }
    ts = pd.DatetimeIndex(df1m["timestamp"])
    expected = int((end - start).total_seconds() // 60)
    observed = int(len(df1m))
    gaps = np.diff(ts.asi8)
    step = 60 * 10**9
    breaks = np.flatnonzero(gaps != step) + 1
    bounds = np.r_[0, breaks, len(ts)]
    run_lengths = [int((ts[b-1] - ts[a]).total_seconds() // 60 + 1) for a, b in zip(bounds[:-1], bounds[1:])]
    return {
        "bars_1m": observed,
        "expected_minutes": expected,
        "observed_minutes": observed,
        "coverage_fraction": float(observed / expected) if expected else np.nan,
        "n_contiguous_runs": int(len(run_lengths)),
        "largest_contiguous_run_minutes": int(max(run_lengths) if run_lengths else 0),
    }


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = pd.to_datetime(out["timestamp"], utc=True)
    out["date"] = t.dt.date.astype(str)
    out["year"] = t.dt.year.astype(int)
    out["quarter"] = t.dt.quarter.astype(int)
    out["month"] = t.dt.month.astype(int)
    return out


def aggregate_asset_summary(summaries: List[Dict[str, object]]) -> pd.DataFrame:
    if not summaries:
        return pd.DataFrame()
    return pd.DataFrame(summaries).sort_values("symbol").reset_index(drop=True)


def save_asset_comparison_plot(summary_df: pd.DataFrame, path: Path) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary_df))
    h = summary_df["mean_H_neural"].to_numpy(float)
    sd = summary_df["sd_H_neural"].to_numpy(float)
    ax.bar(x, h, yerr=sd, capsize=4, alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["symbol"].astype(str).tolist(), rotation=0)
    ax.set_ylabel("Neural H")
    ax.set_title("v2.15 empirical neural roughness across crypto assets")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_neural_vs_variogram_multiasset(asset_frames: Dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    made = False
    for symbol, df in asset_frames.items():
        a = df["H_neural_mean"].to_numpy(float)
        b = df["H_variogram"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            continue
        ax.scatter(a[ok], b[ok], s=12, alpha=0.30, label=symbol)
        made = True
    if not made:
        plt.close(fig)
        return
    all_a = np.concatenate([df["H_neural_mean"].to_numpy(float) for df in asset_frames.values()])
    all_b = np.concatenate([df["H_variogram"].to_numpy(float) for df in asset_frames.values()])
    ok = np.isfinite(all_a) & np.isfinite(all_b)
    lo = min(float(np.min(all_a[ok])), float(np.min(all_b[ok])))
    hi = max(float(np.max(all_a[ok])), float(np.max(all_b[ok])))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.9)
    ax.set_xlabel("Neural H")
    ax.set_ylabel("Variogram H")
    ax.set_title("v2.15 neural vs variogram roughness across assets")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main empirical computation
# -----------------------------------------------------------------------------

def compute_past_future_rv(
    return_series: np.ndarray,
    block_end_indices: np.ndarray,
    return_timestamps: np.ndarray,
    minutes: int,
):
    """Past/future realized variance, rejecting windows that cross data gaps."""
    r2 = return_series ** 2
    cs = np.concatenate([[0.0], np.cumsum(r2, dtype=np.float64)])
    ts_ns = pd.to_datetime(return_timestamps, utc=True).astype("int64").to_numpy()
    minute_ns = 60 * 10**9
    past = np.full(len(block_end_indices), np.nan, np.float64)
    future = np.full(len(block_end_indices), np.nan, np.float64)
    m = int(minutes)
    for i, end_idx in enumerate(block_end_indices):
        end = int(end_idx)
        # Past window: exactly m consecutive 1-minute returns ending at end.
        a0 = end - m + 1
        b0 = end + 1
        if a0 >= 0 and np.all(np.diff(ts_ns[a0:b0]) == minute_ns):
            past[i] = max(float(cs[b0] - cs[a0]), 1e-18)
        # Future window: exactly m consecutive returns immediately after end.
        a1 = end + 1
        b1 = min(len(return_series), a1 + m)
        if b1 - a1 == m and np.all(np.diff(ts_ns[a1:b1]) == minute_ns):
            future[i] = max(float(cs[b1] - cs[a1]), 1e-18)
    return past, future


def make_window_feature_batch(levels: np.ndarray, ends: np.ndarray, W: int, cfg: Config):
    X = np.stack([levels[e - W + 1:e + 1] for e in ends], axis=0).astype(np.float32)
    feats = robust_representation_batch(X, cfg)
    return X, feats


def predict_ensemble(
    levels: np.ndarray,
    block_ts: np.ndarray,
    future_rv: np.ndarray,
    models,
    cfg: Config,
    device: torch.device,
) -> pd.DataFrame:
    W = int(cfg.context_length)
    run_ids = cfg._run_ids_for_current_asset
    ends = rolling_window_indices(run_ids, W, cfg.stride_observations)
    rows = []
    for s in range(0, len(ends), cfg.inference_batch_size):
        e = min(len(ends), s + cfg.inference_batch_size)
        end_batch = ends[s:e]
        X, feats = make_window_feature_batch(levels, end_batch, W, cfg)
        preds = []
        for seed, model, mean, std in models:
            Xn = np.array(feats, np.float32, copy=True)
            Xn -= mean
            Xn /= np.maximum(std, 1e-8)
            with torch.no_grad():
                p = model(torch.from_numpy(Xn).to(device)).cpu().numpy().astype(float)
            preds.append(p)
            del Xn
        P = np.stack(preds, axis=1)
        hmean = np.mean(P, axis=1)
        hsd = np.std(P, axis=1, ddof=1) if P.shape[1] > 1 else np.zeros(len(P))
        # Empirical seed ensemble interval; deliberately not labelled a 95% CI.
        hq_lo = np.min(P, axis=1)
        hq_hi = np.max(P, axis=1)

        # Classical estimators use the finest log-RV scale (column 3 = M=64).
        finest = X[:, :, -1]
        vh = np.asarray([variogram_h(w, cfg.benchmark_scales) for w in finest], float)
        dh = np.asarray([dfa_h(w, cfg.dfa_scales) for w in finest], float)

        past_rv = cfg._past_rv_1d[end_batch]
        frv = future_rv[end_batch]
        past_logrv = np.log(np.maximum(past_rv, 1e-18))
        future_logrv = np.log(np.maximum(frv, 1e-18))
        future_delta = future_logrv - past_logrv

        for j, idx in enumerate(end_batch):
            rows.append({
                "timestamp": pd.Timestamp(block_ts[idx]),
                "block_index": int(idx),
                "H_neural_mean": float(hmean[j]),
                "H_neural_seed_sd": float(hsd[j]),
                "H_neural_seed_min": float(hq_lo[j]),
                "H_neural_seed_max": float(hq_hi[j]),
                "H_seed_42": float(P[j, 0]) if P.shape[1] >= 1 else np.nan,
                "H_seed_123": float(P[j, 1]) if P.shape[1] >= 2 else np.nan,
                "H_seed_2024": float(P[j, 2]) if P.shape[1] >= 3 else np.nan,
                "H_variogram": float(vh[j]),
                "H_dfa": float(dh[j]),
                "future_rv_1d": float(frv[j]) if np.isfinite(frv[j]) else np.nan,
                "future_log_rv_1d": float(future_logrv[j]) if np.isfinite(future_logrv[j]) else np.nan,
                "future_delta_log_rv_1d": float(future_delta[j]) if np.isfinite(future_delta[j]) else np.nan,
            })
    return pd.DataFrame(rows)


def save_h_timeseries_plot(df: pd.DataFrame, path: Path, symbol: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    t = pd.to_datetime(df["timestamp"], utc=True)
    ax.plot(t, df["H_neural_mean"], label="Neural ensemble mean", linewidth=1.1)
    ax.fill_between(t, df["H_neural_seed_min"], df["H_neural_seed_max"], alpha=0.15, label="seed ensemble range")
    ax.plot(t, df["H_variogram"], label="Variogram benchmark", linewidth=0.8, alpha=0.65)
    ax.axhline(0.5, linestyle="--", linewidth=0.8)
    ax.set_ylabel("H estimate")
    ax.set_xlabel("UTC")
    ax.set_title(f"v2.15 empirical roughness | {symbol}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_scatter(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    a = df[x].to_numpy(float); b = df[y].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]; b = b[ok]
    if len(a) < 3:
        return
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(a, b, s=8, alpha=0.35)
    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.9)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def summarize_asset(df: pd.DataFrame, symbol: str, cfg: Config) -> Dict[str, object]:
    out = {
        "symbol": symbol,
        "primary_inference_nonoverlapping": True,
        "n_estimates": int(len(df)),
        "date_start": str(pd.Timestamp(df["timestamp"].min())) if len(df) else None,
        "date_end": str(pd.Timestamp(df["timestamp"].max())) if len(df) else None,
        "mean_H_neural": float(np.nanmean(df["H_neural_mean"])),
        "median_H_neural": float(np.nanmedian(df["H_neural_mean"])),
        "sd_H_neural": float(np.nanstd(df["H_neural_mean"], ddof=1)),
        "mean_seed_sd": float(np.nanmean(df["H_neural_seed_sd"])),
        "seed_sd_to_temporal_sd_ratio": float(np.nanmean(df["H_neural_seed_sd"]) / max(np.nanstd(df["H_neural_mean"], ddof=1), 1e-12)),
        "temporal_variation_exceeds_mean_seed_sd": bool(np.nanstd(df["H_neural_mean"], ddof=1) > np.nanmean(df["H_neural_seed_sd"])),
        "corr_neural_variogram_pearson": corr_safe(df["H_neural_mean"], df["H_variogram"], "pearson"),
        "corr_neural_variogram_spearman": corr_safe(df["H_neural_mean"], df["H_variogram"], "spearman"),
        "corr_neural_dfa_pearson": corr_safe(df["H_neural_mean"], df["H_dfa"], "pearson"),
        "corr_neural_future_logrv1d_pearson": corr_safe(df["H_neural_mean"], df["future_log_rv_1d"], "pearson"),
        "corr_neural_future_delta_logrv1d_pearson": corr_safe(df["H_neural_mean"], df["future_delta_log_rv_1d"], "pearson"),
    }
    boot = bootstrap_summary(df, cfg, seed=91000 + sum(map(ord, symbol)))
    out.update(boot)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_root", default="results_hurst_v2_13_effective_memory_saturation")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start_date", default="2024-01-01")
    parser.add_argument("--end_date", default="2026-09-01")
    parser.add_argument("--context_length", type=int, default=384)
    parser.add_argument("--fine_bars_per_observation", type=int, default=64)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--stride_observations", type=int, default=384, help="Primary inference stride; 384 gives non-overlapping 17-day windows.")
    parser.add_argument("--rolling_diagnostic_stride", type=int, default=0, help="Optional secondary rolling stride; 0 disables.")
    parser.add_argument("--min_valid_windows", type=int, default=20, help="Minimum primary windows required per asset.")
    parser.add_argument("--benchmark_scales", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--dfa_scales", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--future_rv_minutes", type=int, default=1440)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--bootstrap_block", type=int, default=4, help="Block length measured in non-overlapping empirical windows.")
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--download_dir", default="binance_cache_v2_15")
    parser.add_argument("--out_dir", default="results_hurst_v2_15_empirical_crypto_multiasset")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = Config(
        model_root=args.model_root,
        symbols=tuple(s.upper() for s in args.symbols),
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        context_length=args.context_length,
        fine_bars_per_observation=args.fine_bars_per_observation,
        resolutions=tuple(args.resolutions),
        stride_observations=args.stride_observations,
        rolling_diagnostic_stride=args.rolling_diagnostic_stride,
        min_valid_windows=args.min_valid_windows,
        benchmark_scales=tuple(args.benchmark_scales),
        dfa_scales=tuple(args.dfa_scales),
        future_rv_minutes=args.future_rv_minutes,
        bootstrap=args.bootstrap,
        bootstrap_block=args.bootstrap_block,
        inference_batch_size=args.inference_batch_size,
        download_dir=args.download_dir,
        out_dir=args.out_dir,
        device=args.device,
    )

    if cfg.context_length % 32 != 0:
        raise ValueError("context_length should be divisible by H-block=32; 384 is recommended.")
    if cfg.stride_observations <= 0:
        raise ValueError("stride_observations must be positive for primary inference.")
    if cfg.rolling_diagnostic_stride < 0:
        raise ValueError("rolling_diagnostic_stride must be >= 0.")
    if cfg.fine_bars_per_observation != max(cfg.resolutions):
        raise ValueError("fine_bars_per_observation must equal max(resolutions)=64.")
    if cfg.interval != "1m":
        raise ValueError("v2.15 empirical pipeline currently requires Binance 1m klines.")

    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.device)

    print("=" * 88)
    print("HurstNeuralEstimator_v2_15_empirical_crypto_multiasset")
    print("=" * 88)
    print(f"Device: {device}")
    print("Basis: ORIGINAL v2.7 stochastic-H (v2.7-fix1 was NOT used)")
    print("Frozen estimator: v2.1-fix2 BiLSTM + mean/last pooling + bounded H")
    print("Representation: multiscale_rv_robust")
    print("Real-data target training: NONE (zero-shot empirical application)")
    print("Primary inference: NON-OVERLAPPING windows (stride=context_length by default)")
    print(f"Context: {cfg.context_length} coarse observations x {cfg.fine_bars_per_observation} 1m bars")
    print(f"Effective lookback: {cfg.context_length * cfg.fine_bars_per_observation} minutes (~{cfg.context_length * cfg.fine_bars_per_observation / 1440:.2f} days)")
    print(f"Symbols: {cfg.symbols}")
    print(f"Primary stride: {cfg.stride_observations} observations; rolling diagnostic stride: {cfg.rolling_diagnostic_stride or 'DISABLED'}")
    print(f"Dates: {cfg.start_date} -> {cfg.end_date} UTC")
    print("=" * 88)

    model_root = Path(cfg.model_root)
    seed_models = []
    for seed in (42, 123, 2024):
        model, mean, std = load_checkpoint_bundle(model_root, seed, cfg.context_length, device)
        seed_models.append((seed, model, mean, std))
        print(f"Loaded frozen synthetic model seed={seed}")

    summaries = []
    asset_frames: Dict[str, pd.DataFrame] = {}
    for symbol in cfg.symbols:
        print("\n" + "-" * 88)
        print(f"ASSET {symbol}")
        print("-" * 88)
        t0 = time.perf_counter()
        df1m = load_symbol_1m(symbol, cfg)
        start_ts = parse_date_utc(cfg.start_date)
        end_ts = parse_date_utc(cfg.end_date)
        coverage = summarize_data_coverage(df1m, start_ts, end_ts)
        print(f"  Loaded {len(df1m):,} 1m bars in {time.perf_counter() - t0:.1f}s")
        print(f"  Coverage = {coverage['coverage_fraction']:.4f}; contiguous runs = {coverage['n_contiguous_runs']}; longest run = {coverage['largest_contiguous_run_minutes']/1440:.2f} days")

        t1 = time.perf_counter()
        coarse = build_coarse_blocks(df1m, cfg)
        print(f"  Built {len(coarse['levels']):,} coarse 64-minute observations in {time.perf_counter() - t1:.1f}s")
        if len(coarse["levels"]) < cfg.context_length + 10:
            raise RuntimeError("Insufficient empirical history after continuity filtering.")

        past_rv, future_rv = compute_past_future_rv(
            coarse["returns"],
            coarse["block_end_return_index"],
            coarse["return_timestamps"],
            cfg.future_rv_minutes,
        )

        # Temporary per-asset context used by predict_ensemble to avoid copying
        # large auxiliary arrays through every function call.
        cfg._run_ids_for_current_asset = coarse["block_run_id"]
        cfg._past_rv_1d = past_rv

        t2 = time.perf_counter()
        res = predict_ensemble(
            coarse["levels"], coarse["block_ts"], future_rv,
            seed_models, cfg, device,
        )
        print(f"  Generated {len(res):,} NON-OVERLAPPING primary estimates in {time.perf_counter() - t2:.1f}s")
        if len(res) < cfg.min_valid_windows:
            raise RuntimeError(
                f"Only {len(res)} valid primary windows for {symbol}; minimum is {cfg.min_valid_windows}. "
                "Reduce context length/date range or inspect data continuity."
            )
        res = add_time_features(res)

        summary = summarize_asset(res, symbol, cfg)
        summary.update({
            "primary_stride_observations": int(cfg.stride_observations),
            "primary_windows_nonoverlapping": True,
            "bootstrap_block_windows": int(cfg.bootstrap_block),
            **coverage,
        })

        # Optional high-frequency rolling diagnostic for visualization only.
        if cfg.rolling_diagnostic_stride and cfg.rolling_diagnostic_stride < cfg.stride_observations:
            old_stride = cfg.stride_observations
            cfg.stride_observations = int(cfg.rolling_diagnostic_stride)
            try:
                roll = predict_ensemble(
                    coarse["levels"], coarse["block_ts"], future_rv,
                    seed_models, cfg, device,
                )
                roll = add_time_features(roll)
                roll.to_csv(out_root / f"{symbol}_rolling_diagnostic_stride{cfg.rolling_diagnostic_stride}.csv", index=False)
                save_h_timeseries_plot(roll, out_root / f"{symbol}_H_timeseries_rolling.png", symbol)
                print(f"  Generated optional rolling diagnostic: {len(roll):,} estimates")
            finally:
                cfg.stride_observations = old_stride

        # Save raw primary asset output.
        asset_csv = out_root / f"{symbol}_rolling_estimates.csv"
        res.to_csv(asset_csv, index=False)
        try:
            res.to_parquet(out_root / f"{symbol}_rolling_estimates.parquet", index=False)
        except Exception as exc:
            print(f"  WARNING: parquet export skipped ({exc}); CSV remains authoritative.")

        summaries.append(summary)
        asset_frames[symbol] = res.copy()
        with open(out_root / f"{symbol}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        save_h_timeseries_plot(res, out_root / f"{symbol}_H_timeseries.png", symbol)
        save_scatter(res, "H_neural_mean", "H_variogram", out_root / f"{symbol}_neural_vs_variogram.png", f"{symbol}: neural vs variogram")
        save_scatter(res, "H_neural_mean", "H_dfa", out_root / f"{symbol}_neural_vs_dfa.png", f"{symbol}: neural vs DFA")
        save_scatter(res, "H_neural_mean", "future_log_rv_1d", out_root / f"{symbol}_neural_vs_future_logrv.png", f"{symbol}: neural H vs future 1d log-RV")

        print(f"  Mean neural H      = {summary['mean_H_neural']:.4f}")
        print(f"  Median neural H    = {summary['median_H_neural']:.4f}")
        print(f"  Mean seed SD       = {summary['mean_seed_sd']:.4f}")
        print(f"  Corr neural/vario  = {summary['corr_neural_variogram_pearson']:.4f}")
        print(f"  Corr neural/futRV  = {summary['corr_neural_future_logrv1d_pearson']:.4f}")
        print(f"  Bootstrap CI corr neural/vario: [{summary.get('corr_neural_variogram_ci_low', np.nan):.4f}, {summary.get('corr_neural_variogram_ci_high', np.nan):.4f}]")

    summary_df = aggregate_asset_summary(summaries)
    summary_df.to_csv(out_root / "empirical_crypto_multiasset_summary.csv", index=False)
    # Backward-compatible alias.
    summary_df.to_csv(out_root / "empirical_crypto_summary.csv", index=False)
    save_asset_comparison_plot(summary_df, out_root / "cross_asset_neural_H.png")
    if asset_frames:
        save_neural_vs_variogram_multiasset(asset_frames, out_root / "cross_asset_neural_vs_variogram.png")
    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("\n" + "=" * 88)
    print("v2.15 EMPIRICAL CRYPTO COMPLETE")
    print("=" * 88)
    print(f"Results: {out_root}")


if __name__ == "__main__":
    main()
