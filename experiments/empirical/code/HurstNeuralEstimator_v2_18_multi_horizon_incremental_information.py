"""
HurstNeuralEstimator_v2_18_multi_horizon_incremental_information.py

Purpose
-------
Test whether the frozen neural roughness estimate H_t contains incremental
information about future realized volatility across multiple horizons:
    6h, 1d, 3d

This is an empirical follow-up to v2.17 and uses ONLY the v2.15 empirical
panel for H estimates plus the original 1-minute market data. It does not
retrain the neural estimator.

Primary per-asset models at each horizon h:
    M0: FutureLogRV_h ~ 1 + CurrentLogRV_h
    M1: FutureLogRV_h ~ 1 + CurrentLogRV_h + NeuralH
    M2: FutureLogRV_h ~ 1 + CurrentLogRV_h + NeuralH + VariogramH

The current-volatility control is measured over the SAME horizon length as
future volatility. This asks whether H_t adds information beyond the most
natural matched-horizon volatility state.

Inference
---------
- HAC/Newey-West p-value for H in M1.
- Partial correlation H vs FutureLogRV conditional on CurrentLogRV.
- Incremental R^2.
- Moving-block bootstrap CIs for beta_H, partial correlation, incremental R^2.
- Freedman-Lane-style circular block permutation p-value for incremental H.
- Benjamini-Hochberg FDR across all asset x horizon tests.
- Fixed-effects pooled panel regression by horizon with cluster-by-timestamp
  standard errors.

Data source
-----------
The script imports the v2.15 empirical data loader so the same Binance Spot
1-minute data/cache can be reused. If archives are absent locally, the v2.15
loader will download them through the same public archive mechanism.

Important interpretation
------------------------
This tests predictive incremental information, not causality.
With 57 non-overlapping windows per asset in the v2.15 design, inference is
necessarily modest in sample size. Results are intended to distinguish a
robust horizon profile of H information, not establish universal forecasting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore", category=RuntimeWarning)


# -----------------------------------------------------------------------------
# Basic numerics
# -----------------------------------------------------------------------------

def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x = x[m]
    y = y[m]
    sx = np.std(x, ddof=0)
    sy = np.std(y, ddof=0)
    if sx <= 1e-15 or sy <= 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def add_const(x: np.ndarray) -> np.ndarray:
    return sm.add_constant(np.asarray(x, float), has_constant="add")


def residualize(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = add_const(x)
    beta = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)[0]
    fitted = X @ beta
    return np.asarray(y, float) - fitted, fitted


def partial_corr_h_current(h: np.ndarray, y: np.ndarray, current: np.ndarray) -> float:
    rh, _ = residualize(h, current)
    ry, _ = residualize(y, current)
    return corr(rh, ry)


def fit_ols_hac(
    y: np.ndarray,
    current: np.ndarray,
    h: np.ndarray | None = None,
    variogram: np.ndarray | None = None,
    maxlags: int = 4,
):
    cols = [np.ones(len(y)), current]
    names = ["const", "current_log_rv"]
    if h is not None:
        cols.append(h)
        names.append("H_neural_mean")
    if variogram is not None:
        cols.append(variogram)
        names.append("H_variogram")
    X = np.column_stack(cols)
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlags)}), names


def rsq(y: np.ndarray, X_no_const: np.ndarray) -> float:
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X_no_const, float)])
    fit = sm.OLS(y, X).fit()
    return float(fit.rsquared)


# -----------------------------------------------------------------------------
# Bootstrap / permutation
# -----------------------------------------------------------------------------

def circular_block_index_matrix(n: int, block_len: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized circular moving-block bootstrap indices, shape (reps, n)."""
    n = int(n)
    reps = int(reps)
    L = max(1, min(int(block_len), n))
    nb = int(math.ceil(n / L))
    starts = rng.integers(0, n, size=(reps, nb), dtype=np.int64)
    offsets = np.arange(L, dtype=np.int64)[None, None, :]
    idx3 = (starts[:, :, None] + offsets) % n
    return idx3.reshape(reps, nb * L)[:, :n]


def ols_beta_h_and_delta_r2(y: np.ndarray, current: np.ndarray, h: np.ndarray) -> Tuple[float, float]:
    """Closed-form small OLS helper used inside bootstrap/permutation loops."""
    y = np.asarray(y, float)
    c = np.asarray(current, float)
    hh = np.asarray(h, float)
    X0 = np.column_stack([np.ones(len(y)), c])
    X1 = np.column_stack([np.ones(len(y)), c, hh])
    b1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    yhat0 = X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    yhat1 = X1 @ b1
    sse0 = float(np.sum((y - yhat0) ** 2))
    sse1 = float(np.sum((y - yhat1) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    delta = (sse0 - sse1) / max(sst, 1e-18)
    return float(b1[2]), float(delta)


def moving_block_bootstrap(
    y: np.ndarray,
    current: np.ndarray,
    h: np.ndarray,
    n_boot: int,
    block_len: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    n = len(y)
    idx_mat = circular_block_index_matrix(n, block_len, n_boot, rng)
    beta = np.full(n_boot, np.nan, float)
    delta = np.full(n_boot, np.nan, float)
    partial = np.full(n_boot, np.nan, float)
    for b in range(n_boot):
        idx = idx_mat[b]
        yy = y[idx]
        cc = current[idx]
        hh = h[idx]
        try:
            beta[b], delta[b] = ols_beta_h_and_delta_r2(yy, cc, hh)
            partial[b] = partial_corr_h_current(hh, yy, cc)
        except Exception:
            continue
    return {"beta_h": beta, "delta_r2": delta, "partial_corr": partial}


def permutation_incremental_h(
    y: np.ndarray,
    current: np.ndarray,
    h: np.ndarray,
    n_perm: int,
    block_len: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Freedman-Lane-style block permutation conditional on current volatility."""
    n = len(y)
    m0 = sm.OLS(y, add_const(current)).fit()
    ry, _ = residualize(y, current)
    rh, fh = residualize(h, current)
    _, observed_delta = ols_beta_h_and_delta_r2(y, current, h)

    idx_mat = circular_block_index_matrix(n, block_len, n_perm, rng)
    null = np.full(n_perm, np.nan, float)
    for b in range(n_perm):
        h_star = fh + rh[idx_mat[b]]
        try:
            _, null[b] = ols_beta_h_and_delta_r2(y, current, h_star)
        except Exception:
            continue
    valid = null[np.isfinite(null)]
    if len(valid):
        p = float((1.0 + np.sum(valid >= observed_delta)) / (len(valid) + 1.0))
    else:
        p = float("nan")
    return {
        "observed_delta_r2": float(observed_delta),
        "null_delta_r2": valid,
        "p_value_one_sided": p,
    }


def percentile_ci(x: np.ndarray, q=(2.5, 97.5)) -> Tuple[float, float]:
    z = np.asarray(x, float)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return float("nan"), float("nan")
    return tuple(np.percentile(z, q).astype(float))


def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    m = np.isfinite(p)
    if not np.any(m):
        return out
    pv = p[m]
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[m] = tmp
    return out


# -----------------------------------------------------------------------------
# v2.15 loader bridge + raw 1m RV construction
# -----------------------------------------------------------------------------

def load_v215_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("v215_empirical", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load v2.15 script: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_raw_returns(df1m: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    d = df1m.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    close = d["close"].to_numpy(np.float64)
    logp = np.log(np.maximum(close, 1e-300))
    r = np.diff(logp)
    ts = pd.to_datetime(d["timestamp"].iloc[1:], utc=True).to_numpy()
    return r.astype(np.float64), ts


def horizon_rv_at_endpoints(
    returns: np.ndarray,
    return_ts: np.ndarray,
    endpoint_ts: pd.DatetimeIndex,
    horizon_minutes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Matched-horizon past/future RV around each H endpoint.

    Returns arrays: current_rv, future_rv, current_valid, future_valid.
    Both windows must be exactly horizon_minutes consecutive 1-minute returns.
    """
    r = np.asarray(returns, float)
    ts = pd.to_datetime(return_ts, utc=True)
    ts_ns = ts.astype("int64").to_numpy()
    m = int(horizon_minutes)
    minute_ns = 60 * 10**9
    r2 = r * r
    cs = np.concatenate([[0.0], np.cumsum(r2, dtype=np.float64)])

    end_indices = np.searchsorted(ts_ns, endpoint_ts.astype("int64").to_numpy())
    exact = (end_indices < len(ts_ns))
    exact &= (ts_ns[np.minimum(end_indices, len(ts_ns) - 1)] == endpoint_ts.astype("int64").to_numpy())

    current = np.full(len(endpoint_ts), np.nan, float)
    future = np.full(len(endpoint_ts), np.nan, float)
    cur_ok = np.zeros(len(endpoint_ts), dtype=bool)
    fut_ok = np.zeros(len(endpoint_ts), dtype=bool)

    # Prefix count of gap locations: gap_prefix[k] counts gaps among diff(ts)
    # before return index k. A window of m timestamps is contiguous iff no gap
    # lies between its first and last timestamps.
    gaps = np.diff(ts_ns) != minute_ns
    gap_prefix = np.concatenate([[0], np.cumsum(gaps.astype(np.int64))])

    for j, (e, ex) in enumerate(zip(end_indices, exact)):
        if not ex:
            continue
        e = int(e)

        # Past: m returns ending at endpoint e: [e-m+1, e]
        a0 = e - m + 1
        b0 = e + 1
        if a0 >= 0 and (gap_prefix[e] - gap_prefix[a0]) == 0:
            current[j] = max(float(cs[b0] - cs[a0]), 1e-18)
            cur_ok[j] = True

        # Future: m returns immediately after endpoint e: [e+1, e+m]
        a1 = e + 1
        b1 = e + 1 + m
        if b1 <= len(r) and (gap_prefix[b1 - 1] - gap_prefix[a1 - 1] if m > 0 else 0) == 0:
            # For m timestamps from a1..b1-1, there are m-1 internal gaps.
            if m == 1 or (gap_prefix[b1 - 1] - gap_prefix[a1]) == 0:
                future[j] = max(float(cs[b1] - cs[a1]), 1e-18)
                fut_ok[j] = True

    return current, future, cur_ok, fut_ok


# -----------------------------------------------------------------------------
# Per-asset / pooled analysis
# -----------------------------------------------------------------------------

def analyze_one_asset_horizon(
    d: pd.DataFrame,
    symbol: str,
    horizon_label: str,
    horizon_minutes: int,
    bootstrap_reps: int,
    permutation_reps: int,
    block_len: int,
    hac_lags: int,
    seed: int,
    out_dir: Path,
) -> Dict[str, object]:
    needed = ["H_neural_mean", "H_neural_seed_sd", "H_variogram", "current_log_rv", "future_log_rv"]
    x = d.dropna(subset=needed).copy().reset_index(drop=True)
    n = len(x)
    if n < 12:
        return {
            "symbol": symbol,
            "horizon": horizon_label,
            "horizon_minutes": int(horizon_minutes),
            "n": int(n),
            "status": "insufficient_n",
        }

    y = x["future_log_rv"].to_numpy(float)
    c = x["current_log_rv"].to_numpy(float)
    h = x["H_neural_mean"].to_numpy(float)
    v = x["H_variogram"].to_numpy(float)

    m0, _ = fit_ols_hac(y, c, None, None, hac_lags)
    m1, names1 = fit_ols_hac(y, c, h, None, hac_lags)
    m2, names2 = fit_ols_hac(y, c, h, v, hac_lags)
    ih1 = names1.index("H_neural_mean")
    ih2 = names2.index("H_neural_mean")
    iv2 = names2.index("H_variogram")

    partial = partial_corr_h_current(h, y, c)
    delta_r2 = float(m1.rsquared - m0.rsquared)
    delta_adj_r2 = float(m1.rsquared_adj - m0.rsquared_adj)

    rng_b = np.random.default_rng(int(seed) + 10001)
    boot = moving_block_bootstrap(y, c, h, bootstrap_reps, block_len, rng_b)
    b_beta_lo, b_beta_hi = percentile_ci(boot["beta_h"])
    b_partial_lo, b_partial_hi = percentile_ci(boot["partial_corr"])
    b_dr2_lo, b_dr2_hi = percentile_ci(boot["delta_r2"])

    rng_p = np.random.default_rng(int(seed) + 20001)
    perm = permutation_incremental_h(y, c, h, permutation_reps, block_len, rng_p)

    rh, _ = residualize(h, c)
    ry, _ = residualize(y, c)

    row = {
        "symbol": symbol,
        "horizon": horizon_label,
        "horizon_minutes": int(horizon_minutes),
        "n": int(n),
        "bootstrap": int(bootstrap_reps),
        "permutations": int(permutation_reps),
        "block_len": int(block_len),
        "hac_lags": int(hac_lags),
        "mean_current_logrv": float(np.mean(c)),
        "mean_future_logrv": float(np.mean(y)),
        "corr_H_current_logrv": corr(h, c),
        "corr_H_future_logrv": corr(h, y),
        "model0_R2_current_only": float(m0.rsquared),
        "model1_R2_current_plus_H": float(m1.rsquared),
        "incremental_R2_H_given_current": delta_r2,
        "incremental_adj_R2_H_given_current": delta_adj_r2,
        "delta_AIC_H_minus_current": float(m1.aic - m0.aic),
        "delta_BIC_H_minus_current": float(m1.bic - m0.bic),
        "beta_H_HAC": float(m1.params[ih1]),
        "se_H_HAC": float(m1.bse[ih1]),
        "p_H_HAC": float(m1.pvalues[ih1]),
        "partial_corr_H_future_given_current": float(partial),
        "bootstrap_beta_H_ci_low": b_beta_lo,
        "bootstrap_beta_H_ci_high": b_beta_hi,
        "bootstrap_partial_corr_ci_low": b_partial_lo,
        "bootstrap_partial_corr_ci_high": b_partial_hi,
        "bootstrap_incremental_R2_ci_low": b_dr2_lo,
        "bootstrap_incremental_R2_ci_high": b_dr2_hi,
        "permutation_delta_R2_observed": float(perm["observed_delta_r2"]),
        "permutation_delta_R2_p_one_sided": float(perm["p_value_one_sided"]),
        "beta_H_controlled_for_variogram": float(m2.params[ih2]),
        "p_H_controlled_for_variogram_HAC": float(m2.pvalues[ih2]),
        "beta_variogram_control": float(m2.params[iv2]),
        "p_variogram_control_HAC": float(m2.pvalues[iv2]),
        "model2_R2_current_H_variogram": float(m2.rsquared),
        "durbin_watson_model1_resid": float(durbin_watson(m1.resid)),
        "temporal_variation_over_seed_sd": bool(
            np.std(h, ddof=1) > np.mean(x["H_neural_seed_sd"].to_numpy(float))
        ),
    }

    # Save residualized scatter for the main 1d horizon only; all horizons are
    # summarized by compact forest/heatmap figures later.
    if horizon_label == "1d":
        fig, ax = plt.subplots(figsize=(7.0, 5.3))
        ax.scatter(rh, ry, s=22, alpha=0.75)
        if np.std(rh) > 1e-15:
            slope = float(np.cov(rh, ry, bias=True)[0, 1] / np.var(rh))
            intercept = float(np.mean(ry) - slope * np.mean(rh))
            xs = np.linspace(np.min(rh), np.max(rh), 100)
            ax.plot(xs, intercept + slope * xs, linewidth=1.2)
        ax.axhline(0, linewidth=0.7)
        ax.axvline(0, linewidth=0.7)
        ax.set_xlabel("Neural H residualized on current log-RV")
        ax.set_ylabel("Future 1d log-RV residualized on current log-RV")
        ax.set_title(f"{symbol}: incremental H information (1d)")
        ax.text(0.03, 0.97,
                f"partial r = {partial:.3f}\nΔR² = {delta_r2:.3f}\np(H) HAC = {m1.pvalues[ih1]:.3g}",
                transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.12))
        fig.tight_layout()
        fig.savefig(out_dir / f"{symbol}_partial_H_future_control_current_vol_1d.png", dpi=170)
        plt.close(fig)

    # Store compact permutation null for optional later inspection.
    null_path = out_dir / f"{symbol}_{horizon_label}_permutation_null_delta_R2.npy"
    np.save(null_path, perm["null_delta_r2"])
    return row


def pooled_fixed_effects_by_horizon(panel: pd.DataFrame, horizon_label: str, hac_lags: int) -> Dict[str, object]:
    d = panel[panel["horizon"] == horizon_label].dropna(subset=[
        "future_log_rv", "current_log_rv", "H_neural_mean", "symbol"
    ]).copy()
    if len(d) < 20:
        return {"horizon": horizon_label, "n": int(len(d)), "status": "insufficient_n"}

    y = d["future_log_rv"].to_numpy(float)
    c = d["current_log_rv"].to_numpy(float)
    h = d["H_neural_mean"].to_numpy(float)
    assets = pd.get_dummies(d["symbol"], drop_first=True, dtype=float).to_numpy()
    # Two-way structure needed here is time clustering + asset FE. Time FE is not
    # needed for the primary H coefficient because there is at most one row per
    # asset per timestamp, while we still cluster the common timestamp shocks.
    X = np.column_stack([np.ones(len(d)), c, h, assets])
    names = ["const", "current_log_rv", "H_neural_mean"] + [f"FE_{s}" for s in sorted(d["symbol"].unique())[1:]]
    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.to_datetime(d["timestamp"], utc=True).astype("int64").to_numpy()})
    ih = names.index("H_neural_mean")
    return {
        "horizon": horizon_label,
        "n": int(len(d)),
        "n_assets": int(d["symbol"].nunique()),
        "beta_H_cluster_timestamp": float(fit.params[ih]),
        "se_H_cluster_timestamp": float(fit.bse[ih]),
        "p_H_cluster_timestamp": float(fit.pvalues[ih]),
        "R2": float(fit.rsquared),
        "adj_R2": float(fit.rsquared_adj),
    }


# -----------------------------------------------------------------------------
# Figures and summary
# -----------------------------------------------------------------------------

def save_forest(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    d = summary.dropna(subset=["beta_H_HAC", "bootstrap_beta_H_ci_low", "bootstrap_beta_H_ci_high"]).copy()
    d = d.sort_values(["horizon_order", "beta_H_HAC"])
    fig, ax = plt.subplots(figsize=(10, 7.5))
    y = np.arange(len(d))
    beta = d["beta_H_HAC"].to_numpy(float)
    lo = d["bootstrap_beta_H_ci_low"].to_numpy(float)
    hi = d["bootstrap_beta_H_ci_high"].to_numpy(float)
    xerr = np.vstack([beta - lo, hi - beta])
    ax.errorbar(beta, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels((d["symbol"] + " | " + d["horizon"]).tolist())
    ax.set_xlabel("H coefficient after controlling for matched-horizon current log-RV")
    ax.set_title("Multi-horizon incremental neural-H effect")
    fig.tight_layout()
    fig.savefig(out_dir / "v218_H_coefficient_forest.png", dpi=180)
    plt.close(fig)


def save_heatmap(summary: pd.DataFrame, value_col: str, title: str, path: Path, vmin=None, vmax=None) -> None:
    piv = summary.pivot(index="symbol", columns="horizon", values=value_col)
    order = [c for c in ["6h", "1d", "3d"] if c in piv.columns]
    piv = piv.reindex(columns=order)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    im = ax.imshow(piv.to_numpy(float), aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(np.arange(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_title(title)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_incremental_r2_plot(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    horizons = ["6h", "1d", "3d"]
    positions = np.arange(len(horizons))
    for i, sym in enumerate(sorted(summary["symbol"].unique())):
        d = summary[summary["symbol"] == sym].set_index("horizon")
        vals = [float(d.loc[h, "incremental_R2_H_given_current"]) if h in d.index else np.nan for h in horizons]
        ax.plot(positions, vals, marker="o", label=sym, linewidth=1.1)
    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.set_xticks(positions); ax.set_xticklabels(horizons)
    ax.set_ylabel("Incremental R² of H")
    ax.set_title("Incremental information of neural H across forecast horizons")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "v218_incremental_R2_by_horizon.png", dpi=180)
    plt.close(fig)


def json_safe(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input_root", default="inputs/results_hurst_v2_15_empirical_crypto_multiasset")
    ap.add_argument("--v215_script", default="code/HurstNeuralEstimator_v2_15_empirical_crypto_multiasset.py")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"])
    ap.add_argument("--start_date", default="2024-01-01")
    ap.add_argument("--end_date", default="2026-09-01")
    ap.add_argument("--download_dir", default="binance_cache_v2_15")
    ap.add_argument("--horizons", nargs="+", choices=["6h", "1d", "3d"], default=["6h", "1d", "3d"])
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--permutations", type=int, default=5000)
    ap.add_argument("--block_len", type=int, default=4)
    ap.add_argument("--hac_lags", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--out_dir", default="results_hurst_v2_18_multi_horizon_incremental_information")
    args = ap.parse_args()

    horizon_minutes = {"6h": 360, "1d": 1440, "3d": 4320}
    horizon_order = {"6h": 0, "1d": 1, "3d": 2}

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v215_script = Path(args.v215_script)
    if not v215_script.exists():
        raise FileNotFoundError(f"Missing v2.15 script: {v215_script}")
    v215 = load_v215_module(v215_script)

    print("=" * 88)
    print("HurstNeuralEstimator_v2_18_multi_horizon_incremental_information")
    print("=" * 88)
    print("Purpose: test incremental information of Neural H beyond matched-horizon current volatility")
    print(f"Horizons: {args.horizons} = {[horizon_minutes[h] for h in args.horizons]} minutes")
    print("Input H panel: v2.15 non-overlapping empirical estimates (context=384)")
    print("Neural retraining: NONE")
    print(f"Bootstrap={args.bootstrap}, permutations={args.permutations}, block={args.block_len}, HAC lags={args.hac_lags}")
    print("Current-vol control horizon = future-vol horizon for each test")

    rows: List[Dict[str, object]] = []
    merged_panel_parts: List[pd.DataFrame] = []

    for sidx, symbol in enumerate(args.symbols):
        csv_path = input_root / f"{symbol}_rolling_estimates.csv"
        if not csv_path.exists():
            print(f"WARNING: missing {csv_path}; skipping {symbol}")
            continue
        hdf = pd.read_csv(csv_path)
        hdf["timestamp"] = pd.to_datetime(hdf["timestamp"], utc=True)
        hdf = hdf.sort_values("timestamp").reset_index(drop=True)
        print("\n" + "-" * 88)
        print(f"ASSET {symbol}")
        print("-" * 88)
        print(f"  H estimates from v2.15: n={len(hdf)}")

        cfg = v215.Config()
        cfg.symbols = (symbol,)
        cfg.start_date = args.start_date
        cfg.end_date = args.end_date
        cfg.download_dir = args.download_dir
        cfg.interval = "1m"
        print("  Loading 1m data/cache using v2.15 loader ...")
        df1m = v215.load_symbol_1m(symbol, cfg)
        print(f"  Loaded {len(df1m):,} 1m bars")
        returns, return_ts = make_raw_returns(df1m)
        print(f"  Built {len(returns):,} 1m returns")

        asset_records = []
        for hidx, hlabel in enumerate(args.horizons):
            hm = horizon_minutes[hlabel]
            cur_rv, fut_rv, ok_cur, ok_fut = horizon_rv_at_endpoints(
                returns, return_ts, pd.DatetimeIndex(hdf["timestamp"]), hm
            )
            d = hdf[["timestamp", "H_neural_mean", "H_neural_seed_sd", "H_variogram"]].copy()
            d["horizon"] = hlabel
            d["horizon_minutes"] = hm
            d["current_rv"] = cur_rv
            d["future_rv"] = fut_rv
            d["current_valid"] = ok_cur
            d["future_valid"] = ok_fut
            d["current_log_rv"] = np.log(np.maximum(d["current_rv"], 1e-18))
            d["future_log_rv"] = np.log(np.maximum(d["future_rv"], 1e-18))
            d.loc[~d["current_valid"], "current_log_rv"] = np.nan
            d.loc[~d["future_valid"], "future_log_rv"] = np.nan
            d["symbol"] = symbol
            asset_records.append(d)

            row = analyze_one_asset_horizon(
                d, symbol, hlabel, hm,
                args.bootstrap, args.permutations, args.block_len, args.hac_lags,
                args.seed + 1000 * sidx + 100 * hidx,
                out_dir,
            )
            rows.append(row)
            print(
                f"  {hlabel:>2}: n={row.get('n', 0)} | "
                f"beta_H={row.get('beta_H_HAC', np.nan):+.4f} | "
                f"p={row.get('p_H_HAC', np.nan):.4g} | "
                f"ΔR²={row.get('incremental_R2_H_given_current', np.nan):+.4f} | "
                f"perm_p={row.get('permutation_delta_R2_p_one_sided', np.nan):.4g}"
            )

        merged_panel_parts.append(pd.concat(asset_records, ignore_index=True))

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No usable v2.15 asset panels found.")

    summary["horizon_order"] = summary["horizon"].map(horizon_order)
    summary["q_H_HAC_all_asset_horizon"] = benjamini_hochberg(summary["p_H_HAC"].to_numpy(float))
    summary["q_permutation_all_asset_horizon"] = benjamini_hochberg(summary["permutation_delta_R2_p_one_sided"].to_numpy(float))
    summary = summary.sort_values(["symbol", "horizon_order"]).reset_index(drop=True)

    panel = pd.concat(merged_panel_parts, ignore_index=True)
    panel = panel.sort_values(["timestamp", "symbol", "horizon_order"]).reset_index(drop=True)
    panel.to_csv(out_dir / "v218_analysis_ready_panel.csv", index=False)

    pooled_rows = []
    for hlabel in args.horizons:
        pr = pooled_fixed_effects_by_horizon(panel, hlabel, args.hac_lags)
        pooled_rows.append(pr)
        if pr.get("status") != "insufficient_n":
            print(
                f"POOLED {hlabel}: beta_H={pr['beta_H_cluster_timestamp']:+.5f} "
                f"p={pr['p_H_cluster_timestamp']:.5g} | R2={pr['R2']:.4f}"
            )
    pooled = pd.DataFrame(pooled_rows)

    summary.to_csv(out_dir / "v218_multi_horizon_by_asset.csv", index=False)
    pooled.to_csv(out_dir / "v218_pooled_FE_by_horizon.csv", index=False)

    # Compact primary summary: per asset and horizon status after FDR.
    primary = summary[[
        "symbol", "horizon", "horizon_minutes", "n",
        "beta_H_HAC", "p_H_HAC", "q_H_HAC_all_asset_horizon",
        "partial_corr_H_future_given_current", "incremental_R2_H_given_current",
        "bootstrap_beta_H_ci_low", "bootstrap_beta_H_ci_high",
        "bootstrap_partial_corr_ci_low", "bootstrap_partial_corr_ci_high",
        "bootstrap_incremental_R2_ci_low", "bootstrap_incremental_R2_ci_high",
        "permutation_delta_R2_p_one_sided", "q_permutation_all_asset_horizon",
        "beta_H_controlled_for_variogram", "p_H_controlled_for_variogram_HAC",
    ]].copy()
    primary.to_csv(out_dir / "v218_primary_results.csv", index=False)

    # Figures.
    save_forest(summary, out_dir)
    save_heatmap(summary, "incremental_R2_H_given_current", "Incremental R² of H by asset and horizon", out_dir / "v218_incremental_R2_heatmap.png")
    save_heatmap(summary, "p_H_HAC", "HAC p-values for incremental H", out_dir / "v218_pvalue_heatmap.png", vmin=0.0, vmax=0.2)
    save_incremental_r2_plot(summary, out_dir)

    # Horizon-level descriptive aggregates.
    agg = summary.groupby("horizon").agg(
        n_assets=("symbol", "nunique"),
        mean_beta_H=("beta_H_HAC", "mean"),
        median_beta_H=("beta_H_HAC", "median"),
        mean_incremental_R2=("incremental_R2_H_given_current", "mean"),
        median_incremental_R2=("incremental_R2_H_given_current", "median"),
        n_HAC_FDR05=("q_H_HAC_all_asset_horizon", lambda s: int(np.sum(np.asarray(s, float) < 0.05))),
        n_perm_FDR05=("q_permutation_all_asset_horizon", lambda s: int(np.sum(np.asarray(s, float) < 0.05))),
    ).reset_index()
    agg["horizon_order"] = agg["horizon"].map(horizon_order)
    agg = agg.sort_values("horizon_order")
    agg.to_csv(out_dir / "v218_horizon_aggregate_summary.csv", index=False)

    # Identify robust cells under both FDR-adjusted tests.
    robust = summary[
        (summary["q_H_HAC_all_asset_horizon"] < 0.05) &
        (summary["q_permutation_all_asset_horizon"] < 0.05) &
        (summary["bootstrap_beta_H_ci_low"] > 0)
    ].copy()

    conclusions = {
        "n_asset_horizon_tests": int(len(summary)),
        "robust_positive_cells_both_FDR_and_bootstrap": int(len(robust)),
        "robust_positive_asset_horizons": [
            {"symbol": str(r.symbol), "horizon": str(r.horizon)} for r in robust.itertuples()
        ],
        "best_horizon_by_mean_incremental_R2": (
            str(agg.loc[agg["mean_incremental_R2"].idxmax(), "horizon"]) if len(agg) else None
        ),
        "pooled_results": [
            {k: json_safe(v) for k, v in row.items()} for row in pooled.to_dict(orient="records")
        ],
        "interpretation_note": (
            "These are predictive incremental-information tests, not causal effects. "
            "Horizon-specific controls are matched to each future-volatility horizon."
        ),
    }
    with open(out_dir / "v218_conclusions.json", "w", encoding="utf-8") as f:
        json.dump(conclusions, f, indent=2)

    cfg_out = {
        "symbols": list(args.symbols),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "horizons_minutes": {h: horizon_minutes[h] for h in args.horizons},
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "block_len": args.block_len,
        "hac_lags": args.hac_lags,
        "input_root": str(input_root),
        "v215_script": str(v215_script),
        "download_dir": args.download_dir,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)

    print("\n" + "=" * 88)
    print("v2.18 MULTI-HORIZON INCREMENTAL-INFORMATION COMPLETE")
    print("=" * 88)
    print(f"Robust positive cells (both FDR tests + bootstrap CI > 0): {len(robust)}/{len(summary)}")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
