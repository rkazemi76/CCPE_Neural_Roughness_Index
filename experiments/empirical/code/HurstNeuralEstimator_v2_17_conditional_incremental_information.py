from __future__ import annotations

"""
HurstNeuralEstimator_v2_17_conditional_incremental_information.py

Purpose
-------
Test whether neural roughness H contains incremental information about next-day
volatility AFTER controlling for current 1-day realized volatility.

Input
-----
The non-overlapping v2.15 empirical CSVs, e.g.
  results_hurst_v2_15_empirical_crypto_multiasset/BTCUSDT_rolling_estimates.csv

Key identity from v2.15
-----------------------
future_delta_log_rv_1d = future_log_rv_1d - current_log_rv_1d
Therefore:
current_log_rv_1d = future_log_rv_1d - future_delta_log_rv_1d

Primary models per asset
------------------------
M0: FutureLogRV ~ 1 + CurrentLogRV
M1: FutureLogRV ~ 1 + CurrentLogRV + NeuralH
M2: FutureLogRV ~ 1 + CurrentLogRV + NeuralH + VariogramH  [secondary]

Primary quantities
------------------
- HAC/Newey-West coefficient and p-value for NeuralH in M1
- partial correlation between NeuralH and FutureLogRV | CurrentLogRV
- incremental R^2: R2(M1) - R2(M0)
- adjusted R^2 and information criteria
- moving-block bootstrap CIs for beta_H, partial correlation, delta-R2
- Freedman-Lane-style block permutation p-value for the incremental H effect,
  preserving the H-vs-current-volatility relation under the null

Important
---------
This is inference on the v2.15 empirical panel. It does NOT retrain the neural
estimator and does NOT use future information when constructing the current-vol
control.
"""

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore", category=RuntimeWarning)

REQUIRED = [
    "timestamp",
    "H_neural_mean",
    "H_neural_seed_sd",
    "H_variogram",
    "future_log_rv_1d",
    "future_delta_log_rv_1d",
]


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x = x[m]
    y = y[m]
    if np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def residualize(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = sm.add_constant(np.asarray(x, float), has_constant="add")
    fit = sm.OLS(np.asarray(y, float), X).fit()
    return np.asarray(fit.resid, float), np.asarray(fit.fittedvalues, float)


def partial_corr_h_current(
    h: np.ndarray, y: np.ndarray, current: np.ndarray
) -> float:
    rh, _ = residualize(h, current)
    ry, _ = residualize(y, current)
    return corr(rh, ry)


def fit_ols_hac(
    y: np.ndarray,
    current: np.ndarray,
    h: np.ndarray | None,
    variogram: np.ndarray | None,
    maxlags: int,
):
    cols = [np.ones(len(y)), current]
    names = ["const", "current_log_rv_1d"]
    if h is not None:
        cols.append(h)
        names.append("H_neural_mean")
    if variogram is not None:
        cols.append(variogram)
        names.append("H_variogram")
    X = np.column_stack(cols)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlags), "use_correction": True})
    return model, names


def r2_ols(y: np.ndarray, X_no_const: np.ndarray | None) -> float:
    if X_no_const is None:
        return float("nan")
    X = sm.add_constant(X_no_const, has_constant="add")
    fit = sm.OLS(y, X).fit()
    return float(fit.rsquared)


def circular_blocks(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap indices of length n."""
    L = max(1, int(block_len))
    starts = rng.integers(0, n, size=int(math.ceil(n / L)))
    out: List[int] = []
    for s in starts:
        out.extend(((s + np.arange(L)) % n).tolist())
        if len(out) >= n:
            break
    return np.asarray(out[:n], dtype=int)


def moving_block_bootstrap(
    y: np.ndarray,
    current: np.ndarray,
    h: np.ndarray,
    variogram: np.ndarray,
    n_boot: int,
    block_len: int,
    maxlags: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    beta = np.full(n_boot, np.nan)
    pval = np.full(n_boot, np.nan)
    partial = np.full(n_boot, np.nan)
    delta_r2 = np.full(n_boot, np.nan)
    aic_diff = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = circular_blocks(len(y), block_len, rng)
        yy = y[idx]
        cc = current[idx]
        hh = h[idx]
        vv = variogram[idx]
        try:
            m0, _ = fit_ols_hac(yy, cc, None, None, maxlags=maxlags)
            m1, names = fit_ols_hac(yy, cc, hh, None, maxlags=maxlags)
            pos_h = names.index("H_neural_mean")
            beta[b] = float(m1.params[pos_h])
            pval[b] = float(m1.pvalues[pos_h])
            partial[b] = partial_corr_h_current(hh, yy, cc)
            delta_r2[b] = float(m1.rsquared - m0.rsquared)
            aic_diff[b] = float(m1.aic - m0.aic)
        except Exception:
            continue
    return {
        "beta_h": beta,
        "pval_h": pval,
        "partial_corr": partial,
        "delta_r2": delta_r2,
        "aic_diff": aic_diff,
    }


def permutation_incremental_h(
    y: np.ndarray,
    current: np.ndarray,
    h: np.ndarray,
    n_perm: int,
    block_len: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """
    Freedman-Lane-style block permutation for the incremental H effect.

    1) Regress H on current volatility.
    2) Keep fitted component fixed.
    3) Circular-block permute H residuals.
    4) Reconstruct pseudo-H.
    5) Compare delta-R2 of current-only vs current+pseudo-H.

    This targets the null that H adds no information conditional on current vol.
    """
    n = len(y)
    m0 = sm.OLS(y, sm.add_constant(current, has_constant="add")).fit()
    rh, fh = residualize(h, current)
    observed_m1 = sm.OLS(y, sm.add_constant(np.column_stack([current, h]), has_constant="add")).fit()
    observed_delta = float(observed_m1.rsquared - m0.rsquared)

    null = np.full(n_perm, np.nan)
    L = max(1, int(block_len))
    for b in range(n_perm):
        idx = circular_blocks(n, L, rng)
        h_star = fh + rh[idx]
        try:
            m1 = sm.OLS(y, sm.add_constant(np.column_stack([current, h_star]), has_constant="add")).fit()
            null[b] = float(m1.rsquared - m0.rsquared)
        except Exception:
            continue
    valid = null[np.isfinite(null)]
    if len(valid):
        # one-sided test for positive incremental explanatory power
        p = float((1.0 + np.sum(valid >= observed_delta)) / (len(valid) + 1.0))
    else:
        p = float("nan")
    return {
        "observed_delta_r2": observed_delta,
        "null_delta_r2": valid,
        "p_value_one_sided": p,
    }


def percentile_ci(x: np.ndarray, q=(2.5, 97.5)) -> Tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    return tuple(np.percentile(x, q).astype(float))


def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    m = np.isfinite(p)
    pv = p[m]
    if len(pv) == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[m] = tmp
    return out


def summarize_asset(
    df: pd.DataFrame,
    symbol: str,
    bootstrap: int,
    permutations: int,
    block_len: int,
    hac_maxlags: int,
    seed: int,
    out_dir: Path,
) -> Tuple[Dict[str, object], Dict[str, object], np.ndarray]:
    d = df.copy()
    # v2.15 identity: future_delta = future - past, so recover current/past log-RV exactly.
    d["current_log_rv_1d"] = d["future_log_rv_1d"] - d["future_delta_log_rv_1d"]
    needed = ["H_neural_mean", "H_neural_seed_sd", "H_variogram", "current_log_rv_1d", "future_log_rv_1d"]
    d = d.dropna(subset=needed).reset_index(drop=True)
    n = len(d)

    y = d["future_log_rv_1d"].to_numpy(float)
    c = d["current_log_rv_1d"].to_numpy(float)
    h = d["H_neural_mean"].to_numpy(float)
    v = d["H_variogram"].to_numpy(float)

    m0, names0 = fit_ols_hac(y, c, None, None, maxlags=hac_maxlags)
    m1, names1 = fit_ols_hac(y, c, h, None, maxlags=hac_maxlags)
    m2, names2 = fit_ols_hac(y, c, h, v, maxlags=hac_maxlags)
    ih = names1.index("H_neural_mean")
    ih2 = names2.index("H_neural_mean")
    iv2 = names2.index("H_variogram")

    partial = partial_corr_h_current(h, y, c)
    delta_r2 = float(m1.rsquared - m0.rsquared)
    delta_adj_r2 = float(m1.rsquared_adj - m0.rsquared_adj)

    boot_rng = np.random.default_rng(seed + 1001)
    boot = moving_block_bootstrap(y, c, h, v, bootstrap, block_len, hac_maxlags, boot_rng)
    b_beta_lo, b_beta_hi = percentile_ci(boot["beta_h"])
    b_pc_lo, b_pc_hi = percentile_ci(boot["partial_corr"])
    b_dr2_lo, b_dr2_hi = percentile_ci(boot["delta_r2"])

    perm_rng = np.random.default_rng(seed + 2002)
    perm = permutation_incremental_h(y, c, h, permutations, block_len, perm_rng)
    perm_null = perm["null_delta_r2"]

    # Secondary M2 checks whether neural H remains useful after adding classical variogram H.
    beta_h_m2 = float(m2.params[ih2])
    p_h_m2 = float(m2.pvalues[ih2])
    beta_v_m2 = float(m2.params[iv2])
    p_v_m2 = float(m2.pvalues[iv2])

    row = {
        "symbol": symbol,
        "n": int(n),
        "hac_maxlags": int(hac_maxlags),
        "bootstrap": int(bootstrap),
        "permutations": int(permutations),
        "block_len": int(block_len),
        "mean_current_logrv": float(np.mean(c)),
        "sd_current_logrv": float(np.std(c, ddof=1)),
        "mean_future_logrv": float(np.mean(y)),
        "corr_H_current_logrv": corr(h, c),
        "corr_H_future_logrv": corr(h, y),
        "model0_R2_current_only": float(m0.rsquared),
        "model0_adj_R2_current_only": float(m0.rsquared_adj),
        "model1_R2_current_plus_H": float(m1.rsquared),
        "model1_adj_R2_current_plus_H": float(m1.rsquared_adj),
        "incremental_R2_H_given_current": delta_r2,
        "incremental_adj_R2_H_given_current": delta_adj_r2,
        "AIC_current_only": float(m0.aic),
        "AIC_current_plus_H": float(m1.aic),
        "delta_AIC_H_minus_current": float(m1.aic - m0.aic),
        "BIC_current_only": float(m0.bic),
        "BIC_current_plus_H": float(m1.bic),
        "delta_BIC_H_minus_current": float(m1.bic - m0.bic),
        "beta_H_HAC": float(m1.params[ih]),
        "se_H_HAC": float(m1.bse[ih]),
        "p_H_HAC": float(m1.pvalues[ih]),
        "partial_corr_H_future_given_current": partial,
        "bootstrap_beta_H_ci_low": b_beta_lo,
        "bootstrap_beta_H_ci_high": b_beta_hi,
        "bootstrap_partial_corr_ci_low": b_pc_lo,
        "bootstrap_partial_corr_ci_high": b_pc_hi,
        "bootstrap_incremental_R2_ci_low": b_dr2_lo,
        "bootstrap_incremental_R2_ci_high": b_dr2_hi,
        "permutation_delta_R2_observed": float(perm["observed_delta_r2"]),
        "permutation_delta_R2_p_one_sided": float(perm["p_value_one_sided"]),
        "beta_H_controlled_for_variogram": beta_h_m2,
        "p_H_controlled_for_variogram_HAC": p_h_m2,
        "beta_variogram_control": beta_v_m2,
        "p_variogram_control_HAC": p_v_m2,
        "model2_R2_current_H_variogram": float(m2.rsquared),
        "durbin_watson_model1_resid": float(durbin_watson(m1.resid)),
        "temporal_variation_over_seed_sd": bool(
            np.std(h, ddof=1) > np.mean(d["H_neural_seed_sd"].to_numpy(float))
        ),
    }

    boot_rows = {
        "symbol": symbol,
        "bootstrap_beta_H": boot["beta_h"],
        "bootstrap_partial_corr": boot["partial_corr"],
        "bootstrap_incremental_R2": boot["delta_r2"],
        "bootstrap_AIC_diff": boot["aic_diff"],
        "permutation_null_incremental_R2": perm_null,
    }

    # Save residualized scatter.
    rh, _ = residualize(h, c)
    ry, _ = residualize(y, c)
    fig, ax = plt.subplots(figsize=(7.0, 5.3))
    ax.scatter(rh, ry, alpha=0.75)
    if np.std(rh) > 1e-15:
        slope = np.cov(rh, ry, bias=True)[0, 1] / np.var(rh)
        intercept = float(np.mean(ry) - slope * np.mean(rh))
        xs = np.linspace(np.min(rh), np.max(rh), 100)
        ax.plot(xs, intercept + slope * xs, linewidth=1.2)
    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    ax.set_xlabel("Neural H residualized on current 1d log-RV")
    ax.set_ylabel("Future 1d log-RV residualized on current 1d log-RV")
    ax.set_title(f"{symbol}: incremental H information")
    ax.text(
        0.03, 0.97,
        f"partial r = {partial:.3f}\nΔR² = {delta_r2:.3f}\np(H) HAC = {m1.pvalues[ih]:.3g}",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.12),
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{symbol}_partial_H_future_control_current_vol.png", dpi=170)
    plt.close(fig)

    # Save permutation null.
    if len(perm_null):
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.hist(perm_null, bins=35, alpha=0.8)
        ax.axvline(perm["observed_delta_r2"], linestyle="--", linewidth=1.3, label="Observed ΔR²")
        ax.set_xlabel("Permutation ΔR²")
        ax.set_ylabel("Count")
        ax.set_title(f"{symbol}: conditional-H permutation null")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{symbol}_conditional_H_permutation.png", dpi=170)
        plt.close(fig)

    return row, boot_rows, d[["timestamp", "H_neural_mean", "H_neural_seed_sd", "H_variogram", "current_log_rv_1d", "future_log_rv_1d", "future_delta_log_rv_1d"]].copy()


def build_forest(summary: pd.DataFrame, out_dir: Path) -> None:
    d = summary.sort_values("beta_H_HAC").copy()
    y = np.arange(len(d))
    lo = d["bootstrap_beta_H_ci_low"].to_numpy(float)
    hi = d["bootstrap_beta_H_ci_high"].to_numpy(float)
    beta = d["beta_H_HAC"].to_numpy(float)
    xerr = np.vstack([beta - lo, hi - beta])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.errorbar(beta, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["symbol"].tolist())
    ax.set_xlabel("H coefficient in future-logRV ~ current-logRV + H")
    ax.set_title("Incremental neural-H effect after controlling for current volatility")
    fig.tight_layout()
    fig.savefig(out_dir / "v217_H_coefficient_forest.png", dpi=180)
    plt.close(fig)



def pooled_fixed_effects(panel: pd.DataFrame, block_len: int, n_boot: int, seed: int) -> Tuple[Dict[str, object], np.ndarray]:
    """
    Pooled asset fixed-effects regression.

    Point-estimate inference uses timestamp-clustered covariance (57 common time
    clusters) rather than HAC over the row ordering, because rows are stacked by
    asset and HAC on that ordering would mix unrelated assets.

    Bootstrap resamples common timestamp blocks, preserving cross-asset dependence
    at each timestamp and short-run temporal dependence across timestamps.
    """
    d = panel.dropna(subset=["symbol", "timestamp", "future_log_rv_1d", "current_log_rv_1d", "H_neural_mean"]).copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    syms = sorted(d["symbol"].unique().tolist())
    times = pd.Index(sorted(d["timestamp"].unique()))

    def design(dd: pd.DataFrame):
        X = [np.ones(len(dd)), dd["current_log_rv_1d"].to_numpy(float), dd["H_neural_mean"].to_numpy(float)]
        names = ["const", "current_log_rv_1d", "H_neural_mean"]
        for s in syms[1:]:
            X.append((dd["symbol"].to_numpy() == s).astype(float))
            names.append(f"FE_{s}")
        return np.column_stack(X), names

    X, names = design(d)
    y = d["future_log_rv_1d"].to_numpy(float)
    groups = pd.factorize(d["timestamp"])[0]
    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})
    ih = names.index("H_neural_mean")

    X0 = np.column_stack([X[:, 0], X[:, 1], X[:, 3:]])
    fit0 = sm.OLS(y, X0).fit()
    delta_r2 = float(fit.rsquared - fit0.rsquared)

    rng = np.random.default_rng(seed)
    L = max(1, int(block_len))
    boot_beta = np.full(n_boot, np.nan)
    boot_delta = np.full(n_boot, np.nan)
    time_values = np.asarray(times)
    n_t = len(time_values)
    # Build a lookup from timestamp to row subset.
    by_time = {t: d.index[d["timestamp"] == t].to_numpy() for t in time_values}
    for b in range(n_boot):
        starts = rng.integers(0, n_t, size=int(math.ceil(n_t / L)))
        chosen = []
        for st in starts:
            for j in range(L):
                chosen.append(time_values[(st + j) % n_t])
                if len(chosen) >= n_t:
                    break
            if len(chosen) >= n_t:
                break
        idx_rows = np.concatenate([by_time[t] for t in chosen[:n_t]])
        dd = d.loc[idx_rows].copy()
        Xb, nb = design(dd)
        yb = dd["future_log_rv_1d"].to_numpy(float)
        try:
            mb = sm.OLS(yb, Xb).fit()
            boot_beta[b] = float(mb.params[nb.index("H_neural_mean")])
            Xb0 = np.column_stack([Xb[:, 0], Xb[:, 1], Xb[:, 3:]])
            mb0 = sm.OLS(yb, Xb0).fit()
            boot_delta[b] = float(mb.rsquared - mb0.rsquared)
        except Exception:
            continue

    beta_lo, beta_hi = percentile_ci(boot_beta)
    delta_lo, delta_hi = percentile_ci(boot_delta)
    return {
        "pooled_n": int(len(d)),
        "pooled_n_timestamps": int(n_t),
        "n_assets": int(len(syms)),
        "assets": ",".join(syms),
        "pooled_beta_H_cluster_timestamp": float(fit.params[ih]),
        "pooled_se_H_cluster_timestamp": float(fit.bse[ih]),
        "pooled_p_H_cluster_timestamp": float(fit.pvalues[ih]),
        "pooled_beta_H_bootstrap_ci_low": beta_lo,
        "pooled_beta_H_bootstrap_ci_high": beta_hi,
        "pooled_R2": float(fit.rsquared),
        "pooled_adj_R2": float(fit.rsquared_adj),
        "pooled_incremental_R2_H": delta_r2,
        "pooled_incremental_R2_bootstrap_ci_low": delta_lo,
        "pooled_incremental_R2_bootstrap_ci_high": delta_hi,
        "pooled_AIC": float(fit.aic),
        "pooled_BIC": float(fit.bic),
    }, boot_beta


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input_root", default="inputs/results_hurst_v2_15_empirical_crypto_multiasset")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"])
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--permutations", type=int, default=5000)
    ap.add_argument("--block_len", type=int, default=4, help="Moving-block length in v2.15 windows (~17 days each).")
    ap.add_argument("--hac_maxlags", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--out_dir", default="results_hurst_v2_17_conditional_incremental_information")
    args = ap.parse_args()

    in_root = Path(args.input_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("HurstNeuralEstimator_v2_17_conditional_incremental_information")
    print("=" * 88)
    print("Purpose: test whether Neural H adds information beyond current 1-day realized volatility")
    print("Input: v2.15 non-overlapping empirical panel")
    print("Primary model: future_log_RV ~ current_log_RV + Neural_H")
    print("Secondary: + variogram H")
    print(f"Bootstrap={args.bootstrap}, permutations={args.permutations}, block={args.block_len}, HAC lags={args.hac_maxlags}")

    all_rows: List[Dict[str, object]] = []
    boot_records: List[Dict[str, object]] = []
    panel_parts: List[pd.DataFrame] = []

    for k, sym in enumerate(args.symbols):
        csv_path = in_root / f"{sym}_rolling_estimates.csv"
        if not csv_path.exists():
            print(f"WARNING: missing {csv_path}; skipping {sym}")
            continue
        d = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED if c not in d.columns]
        if missing:
            raise ValueError(f"{sym}: missing required columns: {missing}")
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        print(f"\nASSET {sym}: n_raw={len(d)}")
        row, boot, p = summarize_asset(
            d, sym,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            block_len=args.block_len,
            hac_maxlags=args.hac_maxlags,
            seed=args.seed + 7919 * k,
            out_dir=out_root,
        )
        all_rows.append(row)
        boot_records.append(boot)
        p["symbol"] = sym
        panel_parts.append(p)
        print(f"  beta_H HAC={row['beta_H_HAC']:+.5f}, p={row['p_H_HAC']:.4g}")
        print(f"  partialCorr={row['partial_corr_H_future_given_current']:+.4f}")
        print(f"  ΔR²={row['incremental_R2_H_given_current']:+.5f}")
        print(f"  permutation p={row['permutation_delta_R2_p_one_sided']:.4g}")
        print(f"  bootstrap CI beta_H=[{row['bootstrap_beta_H_ci_low']:+.5f}, {row['bootstrap_beta_H_ci_high']:+.5f}]")

    if not all_rows:
        raise RuntimeError("No usable v2.15 asset files found.")

    summary = pd.DataFrame(all_rows)
    summary["q_H_HAC_across_assets"] = benjamini_hochberg(summary["p_H_HAC"].to_numpy(float))
    summary["q_permutation_across_assets"] = benjamini_hochberg(summary["permutation_delta_R2_p_one_sided"].to_numpy(float))
    summary.to_csv(out_root / "v217_conditional_regression_by_asset.csv", index=False)

    # Long-form bootstrap tables.
    boot_long = []
    perm_long = []
    for rec in boot_records:
        sym = rec["symbol"]
        for z in np.asarray(rec["bootstrap_beta_H"], float):
            boot_long.append({"symbol": sym, "stat": "beta_H", "value": float(z)})
        for z in np.asarray(rec["bootstrap_partial_corr"], float):
            boot_long.append({"symbol": sym, "stat": "partial_corr", "value": float(z)})
        for z in np.asarray(rec["bootstrap_incremental_R2"], float):
            boot_long.append({"symbol": sym, "stat": "incremental_R2", "value": float(z)})
        for z in np.asarray(rec["bootstrap_AIC_diff"], float):
            boot_long.append({"symbol": sym, "stat": "delta_AIC", "value": float(z)})
        for z in np.asarray(rec["permutation_null_incremental_R2"], float):
            perm_long.append({"symbol": sym, "value": float(z)})
    pd.DataFrame(boot_long).to_csv(out_root / "v217_bootstrap_distributions.csv", index=False)
    pd.DataFrame(perm_long).to_csv(out_root / "v217_permutation_null_distributions.csv", index=False)

    panel = pd.concat(panel_parts, ignore_index=True)
    panel.to_csv(out_root / "v217_analysis_ready_panel.csv", index=False)

    pooled, pooled_boot_beta = pooled_fixed_effects(panel, block_len=args.block_len, n_boot=args.bootstrap, seed=args.seed + 3333)
    pd.DataFrame({"bootstrap_pooled_beta_H": pooled_boot_beta}).to_csv(out_root / "v217_pooled_bootstrap_beta_H.csv", index=False)
    with open(out_root / "v217_pooled_fixed_effects.json", "w", encoding="utf-8") as f:
        json.dump(pooled, f, indent=2)

    build_forest(summary, out_root)

    # Compact conclusion flags for downstream manuscript writing.
    conclusion = {
        "assets_analyzed": summary["symbol"].tolist(),
        "n_assets": int(len(summary)),
        "assets_with_positive_beta_H": int(np.sum(summary["beta_H_HAC"] > 0)),
        "assets_with_H_HAC_p_lt_0_05": int(np.sum(summary["p_H_HAC"] < 0.05)),
        "assets_with_H_HAC_q_lt_0_05": int(np.sum(summary["q_H_HAC_across_assets"] < 0.05)),
        "assets_with_permutation_p_lt_0_05": int(np.sum(summary["permutation_delta_R2_p_one_sided"] < 0.05)),
        "assets_with_permutation_q_lt_0_05": int(np.sum(summary["q_permutation_across_assets"] < 0.05)),
        "mean_incremental_R2": float(summary["incremental_R2_H_given_current"].mean()),
        "median_incremental_R2": float(summary["incremental_R2_H_given_current"].median()),
        "pooled_beta_H_cluster_timestamp": pooled["pooled_beta_H_cluster_timestamp"],
        "pooled_p_H_cluster_timestamp": pooled["pooled_p_H_cluster_timestamp"],
        "interpretation_note": "A positive/robust H effect here means Neural H contains incremental information about next-day log-RV beyond current 1-day log-RV; it does not by itself establish causal forecasting power.",
    }
    with open(out_root / "v217_primary_conclusions.json", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, indent=2)

    print("\n" + "=" * 88)
    print("v2.17 COMPLETE")
    print("=" * 88)
    print(summary[[
        "symbol", "beta_H_HAC", "p_H_HAC", "q_H_HAC_across_assets",
        "partial_corr_H_future_given_current", "incremental_R2_H_given_current",
        "permutation_delta_R2_p_one_sided"
    ]].to_string(index=False))
    print(f"Pooled FE/cluster-by-timestamp beta_H={pooled['pooled_beta_H_cluster_timestamp']:+.5f} p={pooled['pooled_p_H_cluster_timestamp']:.4g}")
    print(f"Results: {out_root}")


if __name__ == "__main__":
    main()
