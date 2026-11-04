#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HurstNeuralEstimator_v2_16_regime_statistical_validation
==========================================================

Purpose
-------
Statistical/regime validation of the zero-shot empirical neural roughness
estimator using the non-overlapping, multi-asset outputs produced by v2.15.

This version does NOT retrain the neural estimator and does NOT download market
 data. It consumes the five v2.15 *_rolling_estimates.csv files directly.

Core questions
--------------
1) Is the temporal variation in inferred H larger than the average disagreement
   between the three frozen synthetic seeds?
2) Are high/low-H regimes persistent through time beyond an iid ordering?
3) Does a high-H regime associate with higher/lower next-1-day realized volatility?
4) How strong are these effects under moving-block bootstrap uncertainty?
5) Do the results agree across BTC, ETH, BNB, SOL and XRP?

Primary empirical panel
-----------------------
57 non-overlapping 17-day windows per asset in the v2.15 dataset. The H series
are already constructed from context=384 coarse observations, each based on 64
one-minute bars. No overlapping rolling windows are introduced here.

Regime definition
-----------------
Within each asset, H_neural_mean is split into LOW / MID / HIGH by the empirical
33rd and 67th percentiles. The thresholds are fixed from the observed H series
and are never recomputed inside a bootstrap sample. This avoids changing the
meaning of the observed regime labels during resampling.

Main statistics
---------------
- temporal SD of H and seed-uncertainty ratio
- lag-1 Pearson/Spearman H persistence
- same-regime persistence probability
- high->high and low->low transition probabilities
- mean regime run length
- permutation p-values for H persistence and state persistence
- future 1-day log-RV mean difference: HIGH minus LOW
- future 1-day RV mean ratio: HIGH / LOW
- block-bootstrap confidence intervals for all main statistics
- simple OLS of future log-RV on H with HAC (Newey-West) standard errors
- cross-asset correlation matrix of H using common timestamps
- multiple-testing adjusted q-values (Benjamini-Hochberg) for the five asset
  persistence tests and five asset future-RV association tests

Interpretation guardrails
-------------------------
This script is descriptive/associational. It does not establish that H causes
future volatility. The neural H remains a model-implied latent roughness measure,
not an observed ground truth. Small-sample inference is therefore reported with
bootstrap/permutation uncertainty rather than only asymptotic p-values.

Example
-------
python HurstNeuralEstimator_v2_16_regime_statistical_validation.py ^
  --input_root results_hurst_v2_15_empirical_crypto_multiasset ^
  --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ^
  --bootstrap 5000 ^
  --permutations 5000 ^
  --bootstrap_block 4 ^
  --out_dir results_hurst_v2_16_regime_statistical_validation
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm


@dataclass
class Config:
    input_root: str = "inputs/results_hurst_v2_15_empirical_crypto_multiasset"
    symbols: Tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT")
    bootstrap: int = 5000
    bootstrap_block: int = 4
    permutations: int = 5000
    seed: int = 20261007
    min_n: int = 20
    future_col: str = "future_log_rv_1d"
    out_dir: str = "results_hurst_v2_16_regime_statistical_validation"
    hac_lags: int = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def fisher_z_corr(x: np.ndarray, y: np.ndarray) -> float:
    r = safe_corr(x, y)
    if not np.isfinite(r):
        return np.nan
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def lag1_corr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return np.nan
    return safe_corr(x[:-1], x[1:])


def rankdata_average(x: np.ndarray) -> np.ndarray:
    """Small dependency-free average-rank implementation."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return safe_corr(rankdata_average(x), rankdata_average(y))


def circular_moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    block = max(1, min(int(block), n))
    starts_needed = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=starts_needed)
    chunks = [((s + np.arange(block)) % n) for s in starts]
    return np.concatenate(chunks)[:n]


def circular_moving_block_index_matrix(
    n: int, block: int, B: int, rng: np.random.Generator
) -> np.ndarray:
    """Vectorized circular moving-block bootstrap index matrix, shape (B,n)."""
    if n <= 0 or B <= 0:
        return np.empty((max(B, 0), max(n, 0)), dtype=np.int32)
    block = max(1, min(int(block), n))
    k = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(B, k), dtype=np.int32)
    offsets = np.arange(block, dtype=np.int32)[None, None, :]
    idx = (starts[:, :, None] + offsets) % n
    idx = idx.reshape(B, k * block)[:, :n]
    return idx


def rowwise_corr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson correlation per row for equal-shape 2D arrays."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    num = np.sum(Xc * Yc, axis=1)
    den = np.sqrt(np.sum(Xc * Xc, axis=1) * np.sum(Yc * Yc, axis=1))
    out = np.full(X.shape[0], np.nan, dtype=float)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def future_high_low_diff_fixed(h: np.ndarray, future: np.ndarray, q1: float, q2: float) -> float:
    labels = labels_from_thresholds(h, q1, q2)
    high = future[labels == 2]
    low = future[labels == 0]
    if len(high) == 0 or len(low) == 0:
        return np.nan
    return float(np.mean(high) - np.mean(low))


def vectorized_future_high_low_diff(
    labels_2d: np.ndarray, future_1d: np.ndarray
) -> np.ndarray:
    high = labels_2d == 2
    low = labels_2d == 0
    high_n = high.sum(axis=1)
    low_n = low.sum(axis=1)
    high_mean = np.divide(
        np.sum(high * future_1d, axis=1),
        high_n,
        out=np.full(labels_2d.shape[0], np.nan, dtype=float),
        where=high_n > 0,
    )
    low_mean = np.divide(
        np.sum(low * future_1d, axis=1),
        low_n,
        out=np.full(labels_2d.shape[0], np.nan, dtype=float),
        where=low_n > 0,
    )
    return high_mean - low_mean


def vectorized_future_high_low_ratio(
    labels_2d: np.ndarray, future_rv_1d: np.ndarray
) -> np.ndarray:
    high = labels_2d == 2
    low = labels_2d == 0
    high_n = high.sum(axis=1)
    low_n = low.sum(axis=1)
    high_mean = np.divide(
        np.sum(high * future_rv_1d, axis=1),
        high_n,
        out=np.full(labels_2d.shape[0], np.nan, dtype=float),
        where=high_n > 0,
    )
    low_mean = np.divide(
        np.sum(low * future_rv_1d, axis=1),
        low_n,
        out=np.full(labels_2d.shape[0], np.nan, dtype=float),
        where=low_n > 0,
    )
    return high_mean / low_mean

def pct_ci(vals: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan)
    return tuple(np.quantile(vals, [alpha / 2, 1 - alpha / 2]).tolist())


def empirical_two_sided_perm_p(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if not np.isfinite(observed) or len(null_values) == 0:
        return np.nan
    return float((1 + np.sum(np.abs(null_values) >= abs(observed))) / (len(null_values) + 1))


def state_labels(h: np.ndarray) -> Tuple[np.ndarray, float, float]:
    h = np.asarray(h, dtype=float)
    q1, q2 = np.quantile(h, [1 / 3, 2 / 3])
    labels = np.full(len(h), 1, dtype=int)  # 0 low, 1 mid, 2 high
    labels[h <= q1] = 0
    labels[h >= q2] = 2
    return labels, float(q1), float(q2)


def labels_from_thresholds(h: np.ndarray, q1: float, q2: float) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    labels = np.full(h.shape, 1, dtype=np.int8)
    labels[h <= q1] = 0
    labels[h >= q2] = 2
    return labels


def transition_matrix(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    mat = np.zeros((3, 3), dtype=float)
    for a, b in zip(labels[:-1], labels[1:]):
        mat[a, b] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = mat / row_sums
    return probs


def same_state_rate(labels: np.ndarray) -> float:
    if len(labels) < 2:
        return np.nan
    return float(np.mean(labels[:-1] == labels[1:]))


def conditional_stay(labels: np.ndarray, state: int) -> float:
    labels = np.asarray(labels, dtype=int)
    current = labels[:-1] == state
    if not np.any(current):
        return np.nan
    return float(np.mean(labels[1:][current] == state))


def run_lengths(labels: np.ndarray) -> List[int]:
    labels = np.asarray(labels, dtype=int)
    if len(labels) == 0:
        return []
    out = []
    cur = labels[0]
    n = 1
    for x in labels[1:]:
        if x == cur:
            n += 1
        else:
            out.append(n)
            cur = x
            n = 1
    out.append(n)
    return out


def mean_run_length(labels: np.ndarray) -> float:
    rl = run_lengths(labels)
    return float(np.mean(rl)) if rl else np.nan


def permuted_lag1_null(h: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    out = np.empty(B, dtype=float)
    for b in range(B):
        xp = rng.permutation(h)
        out[b] = lag1_corr(xp)
    return out


def permuted_same_state_null(labels: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    out = np.empty(B, dtype=float)
    for b in range(B):
        lp = rng.permutation(labels)
        out[b] = same_state_rate(lp)
    return out


def future_high_low_diff(h: np.ndarray, future: np.ndarray) -> float:
    labels, _, _ = state_labels(h)
    high = future[labels == 2]
    low = future[labels == 0]
    if len(high) == 0 or len(low) == 0:
        return np.nan
    return float(np.mean(high) - np.mean(low))


def future_high_low_ratio(h: np.ndarray, future: np.ndarray, q1: float | None = None, q2: float | None = None) -> float:
    if q1 is None or q2 is None:
        labels, _, _ = state_labels(h)
    else:
        labels = labels_from_thresholds(h, q1, q2)
    high = future[labels == 2]
    low = future[labels == 0]
    if len(high) == 0 or len(low) == 0:
        return np.nan
    return float(np.mean(np.exp(high)) / np.mean(np.exp(low)))


def load_asset(root: Path, symbol: str) -> pd.DataFrame:
    candidates = [
        root / f"{symbol}_rolling_estimates.csv",
        root / f"{symbol}_rolling_estimates.parquet",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
            else:
                df = pd.read_parquet(p)
            break
    else:
        raise FileNotFoundError(f"No rolling-estimates file for {symbol} in {root}")

    required = [
        "timestamp", "H_neural_mean", "H_neural_seed_sd",
        "H_seed_42", "H_seed_123", "H_seed_2024",
        "H_variogram", "H_dfa", "future_rv_1d", "future_log_rv_1d",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns: {missing}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def hac_regression(h: np.ndarray, future: np.ndarray, hac_lags: int) -> Dict[str, float]:
    mask = np.isfinite(h) & np.isfinite(future)
    h = np.asarray(h[mask], dtype=float)
    future = np.asarray(future[mask], dtype=float)
    if len(h) < 10:
        return {"n": len(h), "alpha": np.nan, "beta_H": np.nan, "beta_pvalue_H_hac": np.nan,
                "r2": np.nan, "beta_ci_low": np.nan, "beta_ci_high": np.nan}
    X = sm.add_constant(h)
    model = sm.OLS(future, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
    ci = model.conf_int(alpha=0.05)
    return {
        "n": int(len(h)),
        "alpha": float(model.params[0]),
        "beta_H": float(model.params[1]),
        "beta_pvalue_H_hac": float(model.pvalues[1]),
        "r2": float(model.rsquared),
        "beta_ci_low": float(ci[1, 0]),
        "beta_ci_high": float(ci[1, 1]),
    }


def analyze_asset(df: pd.DataFrame, cfg: Config, asset_rng: np.random.Generator) -> Tuple[Dict[str, float], Dict[str, object], Dict[str, object]]:
    df = df.copy()
    h = df["H_neural_mean"].to_numpy(float)
    future = df[cfg.future_col].to_numpy(float)
    future_raw_rv = df["future_rv_1d"].to_numpy(float)
    seed_sd = df["H_neural_seed_sd"].to_numpy(float)

    valid = np.isfinite(h) & np.isfinite(future) & np.isfinite(future_raw_rv)
    h = h[valid]
    future = future[valid]
    future_raw_rv = future_raw_rv[valid]
    seed_sd = seed_sd[valid]
    clean = df.loc[valid].reset_index(drop=True)
    n = len(h)
    if n < cfg.min_n:
        raise ValueError(f"Only {n} valid windows; min_n={cfg.min_n}")

    labels, q1, q2 = state_labels(h)
    trans = transition_matrix(labels)

    temporal_sd = float(np.std(h, ddof=1))
    mean_seed_sd = float(np.mean(seed_sd))
    temporal_to_seed = float(temporal_sd / mean_seed_sd) if mean_seed_sd > 0 else np.nan

    # Future regime summaries
    regime_names = ["LOW", "MID", "HIGH"]
    regime_rows = []
    for s, name in enumerate(regime_names):
        mask = labels == s
        regime_rows.append({
            "regime": name,
            "n": int(np.sum(mask)),
            "mean_H": float(np.mean(h[mask])) if np.any(mask) else np.nan,
            "median_H": float(np.median(h[mask])) if np.any(mask) else np.nan,
            "mean_future_logrv_1d": float(np.mean(future[mask])) if np.any(mask) else np.nan,
            "median_future_logrv_1d": float(np.median(future[mask])) if np.any(mask) else np.nan,
            "mean_future_rv_1d": float(np.mean(future_raw_rv[mask])) if np.any(mask) else np.nan,
        })
    regime_df = pd.DataFrame(regime_rows)

    observed = {
        "n_windows": n,
        "mean_H": float(np.mean(h)),
        "median_H": float(np.median(h)),
        "temporal_sd_H": temporal_sd,
        "mean_seed_sd": mean_seed_sd,
        "temporal_sd_to_mean_seed_sd": temporal_to_seed,
        "temporal_variation_exceeds_seed_sd": bool(temporal_sd > mean_seed_sd),
        "regime_q33": q1,
        "regime_q67": q2,
        "lag1_pearson_H": lag1_corr(h),
        "lag1_spearman_H": safe_corr(rankdata_average(h[:-1]), rankdata_average(h[1:])),
        "same_regime_rate": same_state_rate(labels),
        "high_stay_probability": conditional_stay(labels, 2),
        "low_stay_probability": conditional_stay(labels, 0),
        "mean_run_length_windows": mean_run_length(labels),
        "future_logrv_high_minus_low": future_high_low_diff(h, future),
        "future_rv_high_to_low_ratio": future_high_low_ratio(h, future_raw_rv, q1, q2),
        "corr_H_future_logrv1d_pearson": safe_corr(h, future),
        "corr_H_future_logrv1d_spearman": spearman_corr(h, future),
        "corr_H_variogram_pearson": safe_corr(h, clean["H_variogram"].to_numpy(float)),
        "corr_H_variogram_spearman": spearman_corr(h, clean["H_variogram"].to_numpy(float)),
    }
    observed.update({f"transition_{a}_{b}": trans[a, b] for a, b in [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)
    ]})
    observed.update({f"regime_{r['regime']}_{k}": r[k] for r in regime_rows for k in ["n", "mean_H", "mean_future_logrv_1d", "mean_future_rv_1d"]})
    observed.update({f"ols_{k}": v for k, v in hac_regression(h, future, cfg.hac_lags).items()})

    # Bootstrap distributions using a single vectorized set of circular
    # moving-block indices. Regime thresholds stay fixed at the observed q33/q67.
    fixed_q1, fixed_q2 = q1, q2
    fixed_labels = labels.copy()
    B = int(cfg.bootstrap)
    boot = {}
    boot_idx = circular_moving_block_index_matrix(n, cfg.bootstrap_block, B, asset_rng)
    h_boot = h[boot_idx]
    f_log_boot = future[boot_idx]
    f_raw_boot = future_raw_rv[boot_idx]
    vario_boot = clean["H_variogram"].to_numpy(float)[boot_idx]

    mean_vals = h_boot.mean(axis=1)
    sd_vals = h_boot.std(axis=1, ddof=1)
    lag_vals = rowwise_corr(h_boot[:, :-1], h_boot[:, 1:])
    labels_boot = labels_from_thresholds(h_boot, fixed_q1, fixed_q2)
    same_vals = (labels_boot[:, :-1] == labels_boot[:, 1:]).mean(axis=1)
    hi_stay_vals = np.divide(
        ((labels_boot[:, :-1] == 2) & (labels_boot[:, 1:] == 2)).sum(axis=1),
        (labels_boot[:, :-1] == 2).sum(axis=1),
        out=np.full(B, np.nan), where=(labels_boot[:, :-1] == 2).sum(axis=1) > 0,
    )
    lo_stay_vals = np.divide(
        ((labels_boot[:, :-1] == 0) & (labels_boot[:, 1:] == 0)).sum(axis=1),
        (labels_boot[:, :-1] == 0).sum(axis=1),
        out=np.full(B, np.nan), where=(labels_boot[:, :-1] == 0).sum(axis=1) > 0,
    )
    hl_vals = vectorized_future_high_low_diff(labels_boot, f_log_boot)
    ratio_vals = vectorized_future_high_low_ratio(labels_boot, f_raw_boot)
    corr_future_vals = rowwise_corr(h_boot, f_log_boot)
    corr_vario_vals = rowwise_corr(h_boot, vario_boot)

    boot = {
        "mean_H": mean_vals,
        "temporal_sd_H": sd_vals,
        "lag1_pearson_H": lag_vals,
        "same_regime_rate": same_vals,
        "high_minus_low_future_logrv": hl_vals,
        "high_low_future_rv_ratio": ratio_vals,
        "corr_H_future_logrv1d": corr_future_vals,
        "corr_H_variogram": corr_vario_vals,
        "high_stay_probability": hi_stay_vals,
        "low_stay_probability": lo_stay_vals,
    }
    for name, vals in boot.items():
        lo, hi = pct_ci(vals)
        observed[f"{name}_boot_ci_low"] = lo
        observed[f"{name}_boot_ci_high"] = hi
        observed[f"{name}_boot_mean"] = float(np.nanmean(vals))

    # Permutation nulls for temporal persistence, using vectorized random
    # score sorting to construct row-wise random permutations.
    P = int(cfg.permutations)
    perm_order = np.argsort(asset_rng.random((P, n)), axis=1)
    h_perm = h[perm_order]
    lag_null = rowwise_corr(h_perm[:, :-1], h_perm[:, 1:])

    label_perm = fixed_labels[perm_order]
    same_null = (label_perm[:, :-1] == label_perm[:, 1:]).mean(axis=1)
    future_hl_null = vectorized_future_high_low_diff(label_perm, future)

    observed["lag1_pearson_permutation_p"] = empirical_two_sided_perm_p(
        observed["lag1_pearson_H"], lag_null
    )
    observed["same_regime_permutation_p"] = empirical_two_sided_perm_p(
        observed["same_regime_rate"] - 1 / 3, same_null - 1 / 3
    )
    observed["future_logrv_high_minus_low_permutation_p"] = empirical_two_sided_perm_p(
        observed["future_logrv_high_minus_low"], future_hl_null
    )

    boot_store = {
        "boot": boot,
        "perm_lag1": lag_null,
        "perm_same": same_null,
        "perm_future_high_low": future_hl_null,
    }
    return observed, boot_store, {"regime_df": regime_df, "labels": labels, "clean": clean}


def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not np.any(valid):
        return q
    pv = p[valid]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.clip(adj, 0, 1)
    tmp = np.empty_like(out)
    tmp[order] = out
    q[valid] = tmp
    return q


def plot_regime_timeseries(results: Dict[str, Dict], out: Path) -> None:
    symbols = list(results.keys())
    fig, axes = plt.subplots(len(symbols), 1, figsize=(11, 2.4 * len(symbols)), sharex=False)
    if len(symbols) == 1:
        axes = [axes]
    for ax, sym in zip(axes, symbols):
        clean = results[sym]["clean"]
        h = clean["H_neural_mean"].to_numpy(float)
        q1, q2 = np.quantile(h, [1 / 3, 2 / 3])
        ax.plot(clean["timestamp"], h, linewidth=1.4)
        ax.axhline(q1, linestyle="--", linewidth=0.9)
        ax.axhline(q2, linestyle="--", linewidth=0.9)
        ax.set_title(sym)
        ax.set_ylabel("Neural H")
        ax.set_ylim(0.05, 0.50)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_future_regime_effect(summary: pd.DataFrame, out: Path) -> None:
    pivot = summary.pivot(index="symbol", columns="regime", values="mean_future_logrv_1d")
    order = [c for c in ["LOW", "MID", "HIGH"] if c in pivot.columns]
    ax = pivot[order].plot(kind="bar", figsize=(10, 5), width=0.75)
    ax.set_ylabel("Mean future 1-day log-RV")
    ax.set_title("Future 1-day log-RV by inferred H regime")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="H regime")
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()


def plot_persistence(summary: pd.DataFrame, out: Path) -> None:
    cols = ["same_regime_rate", "high_stay_probability", "low_stay_probability"]
    plot_df = summary.set_index("symbol")[cols]
    ax = plot_df.plot(kind="bar", figsize=(10, 5), width=0.75)
    ax.axhline(1 / 3, linestyle="--", linewidth=1.0)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Persistence of inferred H regimes")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()


def save_transition_heatmaps(transitions: Dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(1, len(transitions), figsize=(4.2 * len(transitions), 3.8))
    if len(transitions) == 1:
        axes = [axes]
    for ax, (sym, mat) in zip(axes, transitions.items()):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(sym)
        ax.set_xticks([0,1,2], ["LOW", "MID", "HIGH"])
        ax.set_yticks([0,1,2], ["LOW", "MID", "HIGH"])
        for i in range(3):
            for j in range(3):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    fig.suptitle("H-regime transition matrices", y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default=Config.input_root)
    parser.add_argument("--symbols", nargs="+", default=list(Config.symbols))
    parser.add_argument("--bootstrap", type=int, default=Config.bootstrap)
    parser.add_argument("--bootstrap_block", type=int, default=Config.bootstrap_block)
    parser.add_argument("--permutations", type=int, default=Config.permutations)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--min_n", type=int, default=Config.min_n)
    parser.add_argument("--future_col", type=str, default=Config.future_col)
    parser.add_argument("--out_dir", type=str, default=Config.out_dir)
    parser.add_argument("--hac_lags", type=int, default=Config.hac_lags)
    args = parser.parse_args()

    cfg = Config(
        input_root=args.input_root,
        symbols=tuple(args.symbols),
        bootstrap=args.bootstrap,
        bootstrap_block=args.bootstrap_block,
        permutations=args.permutations,
        seed=args.seed,
        min_n=args.min_n,
        future_col=args.future_col,
        out_dir=args.out_dir,
        hac_lags=args.hac_lags,
    )
    set_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("HurstNeuralEstimator_v2_16_regime_statistical_validation")
    print("=" * 88)
    print(f"Input: {cfg.input_root}")
    print(f"Assets: {cfg.symbols}")
    print(f"Bootstrap: {cfg.bootstrap} | block={cfg.bootstrap_block}")
    print(f"Permutation tests: {cfg.permutations}")
    print(f"Future target: {cfg.future_col}")
    print("Primary sample: v2.15 non-overlapping 384-observation windows")
    print("Regimes: within-asset terciles of H_neural_mean")
    print("No neural retraining; no market-data download")

    rng = np.random.default_rng(cfg.seed)
    asset_results = {}
    obs_rows = []
    regime_rows = []
    transitions = {}

    print("\n" + "-" * 88)
    for symbol in cfg.symbols:
        df = load_asset(Path(cfg.input_root), symbol)
        print(f"{symbol}: n={len(df)} | {df['timestamp'].min()} -> {df['timestamp'].max()}")
        observed, store, extra = analyze_asset(df, cfg, rng)
        asset_results[symbol] = {
            "observed": observed,
            "boot_store": store,
            "clean": extra["clean"],
            "labels": extra["labels"],
        }
        row = {"symbol": symbol, **observed}
        obs_rows.append(row)
        rdf = extra["regime_df"].copy()
        rdf.insert(0, "symbol", symbol)
        regime_rows.append(rdf)
        transitions[symbol] = transition_matrix(extra["labels"])

        print(
            f"  Mean H={observed['mean_H']:.4f} | SD(H)={observed['temporal_sd_H']:.4f} | "
            f"SD/seedSD={observed['temporal_sd_to_mean_seed_sd']:.2f}"
        )
        print(
            f"  lag1={observed['lag1_pearson_H']:.3f} (perm p={observed['lag1_pearson_permutation_p']:.4f}) | "
            f"same-regime={observed['same_regime_rate']:.3f}"
        )
        print(
            f"  High-stay={observed['high_stay_probability']:.3f} | Low-stay={observed['low_stay_probability']:.3f} | "
            f"mean run={observed['mean_run_length_windows']:.2f}"
        )
        print(
            f"  Future logRV HIGH-LOW={observed['future_logrv_high_minus_low']:.4f} "
            f"(perm p={observed['future_logrv_high_minus_low_permutation_p']:.4f}) | "
            f"Corr={observed['corr_H_future_logrv1d_pearson']:.3f}"
        )
        print(
            f"  HAC beta(H)={observed['ols_beta_H']:.4f} "
            f"[p={observed['ols_beta_pvalue_H_hac']:.4f}; "
            f"95% CI {observed['ols_beta_ci_low']:.4f},{observed['ols_beta_ci_high']:.4f}]"
        )

    summary = pd.DataFrame(obs_rows)
    regime_summary = pd.concat(regime_rows, ignore_index=True)

    # BH-adjust the primary permutation p-values across assets.
    summary["lag1_p_fdr_q"] = benjamini_hochberg(summary["lag1_pearson_permutation_p"].to_numpy(float))
    summary["future_high_low_p_fdr_q"] = benjamini_hochberg(summary["future_logrv_high_minus_low_permutation_p"].to_numpy(float))

    # Cross-asset H correlation matrix on common timestamps.
    series = {}
    for sym in cfg.symbols:
        c = asset_results[sym]["clean"][["timestamp", "H_neural_mean"]].copy()
        c = c.rename(columns={"H_neural_mean": sym}).set_index("timestamp")
        series[sym] = c
    cross = pd.concat(series.values(), axis=1, join="inner")
    corr_mat = cross.corr(method="pearson")

    # Aggregate diagnostics across assets (simple descriptive mean; not a pooled
    # inferential statistic because scales/regime thresholds are asset-specific).
    aggregate = {
        "n_assets": len(cfg.symbols),
        "mean_of_asset_mean_H": float(summary["mean_H"].mean()),
        "mean_of_asset_temporal_sd_H": float(summary["temporal_sd_H"].mean()),
        "mean_of_asset_temporal_to_seed_ratio": float(summary["temporal_sd_to_mean_seed_sd"].mean()),
        "mean_same_regime_rate": float(summary["same_regime_rate"].mean()),
        "mean_high_stay_probability": float(summary["high_stay_probability"].mean()),
        "mean_low_stay_probability": float(summary["low_stay_probability"].mean()),
        "mean_future_high_minus_low_logrv": float(summary["future_logrv_high_minus_low"].mean()),
        "mean_H_future_logrv_corr": float(summary["corr_H_future_logrv1d_pearson"].mean()),
        "mean_H_variogram_corr": float(summary["corr_H_variogram_pearson"].mean()),
    }

    # Cross-asset factor summary using equal-weight cross-sectional daily/window
    # means. This is descriptive only because all assets have the same 57 windows.
    common_H = cross.mean(axis=1)
    aggregate["cross_asset_common_window_H_sd"] = float(common_H.std(ddof=1))
    aggregate["mean_pairwise_H_corr"] = float(
        np.nanmean(corr_mat.to_numpy()[np.triu_indices_from(corr_mat, k=1)])
    )

    summary.to_csv(out / "regime_statistical_summary.csv", index=False)
    regime_summary.to_csv(out / "regime_level_future_rv.csv", index=False)
    corr_mat.to_csv(out / "cross_asset_H_correlation.csv")
    cross.reset_index().to_csv(out / "common_timestamp_H_panel.csv", index=False)

    transition_rows = []
    for sym, mat in transitions.items():
        for i, a in enumerate(["LOW", "MID", "HIGH"]):
            for j, b in enumerate(["LOW", "MID", "HIGH"]):
                transition_rows.append({"symbol": sym, "from": a, "to": b, "probability": mat[i, j]})
    pd.DataFrame(transition_rows).to_csv(out / "regime_transition_probabilities.csv", index=False)

    with open(out / "aggregate_descriptive_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Detailed per-asset JSON and bootstrap arrays.
    for sym, d in asset_results.items():
        safe = d["observed"].copy()
        with open(out / f"{sym}_regime_statistical_summary.json", "w", encoding="utf-8") as f:
            json.dump(safe, f, indent=2, default=float)
        np.savez_compressed(out / f"{sym}_bootstrap_distributions.npz", **d["boot_store"]["boot"])
        np.savez_compressed(
            out / f"{sym}_permutation_nulls.npz",
            lag1=d["boot_store"]["perm_lag1"],
            same_regime=d["boot_store"]["perm_same"],
            future_high_low=d["boot_store"]["perm_future_high_low"],
        )

    plot_regime_timeseries(asset_results, out / "H_regime_timeseries.png")
    plot_future_regime_effect(regime_summary, out / "future_logrv_by_H_regime.png")
    plot_persistence(summary, out / "H_regime_persistence.png")
    save_transition_heatmaps(transitions, out / "H_regime_transition_matrices.png")

    # Compact conclusion file for reproducibility.
    conclusions = {
        "primary_context": 384,
        "primary_assets": list(cfg.symbols),
        "temporal_variation_exceeds_seed_sd_for_all_assets": bool(summary["temporal_variation_exceeds_seed_sd"].all()),
        "assets_with_positive_lag1_persistence": summary.loc[summary["lag1_pearson_H"] > 0, "symbol"].tolist(),
        "assets_with_fdr_supported_lag1_persistence_q_le_0_05": summary.loc[summary["lag1_p_fdr_q"] <= 0.05, "symbol"].tolist(),
        "assets_with_positive_high_minus_low_future_logrv": summary.loc[summary["future_logrv_high_minus_low"] > 0, "symbol"].tolist(),
        "assets_with_fdr_supported_high_low_future_logrv_q_le_0_05": summary.loc[summary["future_high_low_p_fdr_q"] <= 0.05, "symbol"].tolist(),
        "mean_pairwise_H_correlation": aggregate["mean_pairwise_H_corr"],
    }
    with open(out / "primary_conclusions.json", "w", encoding="utf-8") as f:
        json.dump(conclusions, f, indent=2)

    print("\n" + "=" * 88)
    print("v2.16 REGIME + STATISTICAL VALIDATION COMPLETE")
    print("=" * 88)
    print(summary[[
        "symbol", "mean_H", "temporal_sd_H", "temporal_sd_to_mean_seed_sd",
        "lag1_pearson_H", "lag1_pearson_permutation_p", "lag1_p_fdr_q",
        "same_regime_rate", "high_stay_probability", "low_stay_probability",
        "future_logrv_high_minus_low", "future_logrv_high_minus_low_permutation_p",
        "future_high_low_p_fdr_q", "ols_beta_H", "ols_beta_pvalue_H_hac",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFiles written to:", out)


if __name__ == "__main__":
    main()
