"""
HurstNeuralEstimator_v2_19_final_statistical_robustness.py

Purpose
-------
Final statistical robustness for the empirical contribution established in
v2.17-v2.18.  No neural retraining and no data download are required.
The script reads the v2.15 non-overlapping empirical panels (context=384),
constructs the matched-horizon 1-day current/future log-RV variables exactly
as in v2.17, and focuses only on the 1-day horizon.

Primary pooled fixed-effects model
----------------------------------
    FutureLogRV_1d = asset FE + beta_V CurrentLogRV_1d + beta_H NeuralH + e

Core robustness tests
---------------------
1) Common time-block bootstrap: resample timestamp blocks while keeping all
   assets at a sampled timestamp together; estimate beta_H and incremental R2.
2) Wild cluster bootstrap: Rademacher weights at the timestamp-cluster level
   around the unrestricted pooled FE fit; estimate beta_H uncertainty.
3) Leave-one-asset-out pooled FE: omit each asset and re-estimate beta_H.
4) Joint asset-specific H test: allow H coefficient to differ by asset and
   jointly test all five H coefficients = 0 using timestamp-cluster covariance.
5) Variogram-controlled pooled FE model as an additional robustness check.
6) A simple leave-one-asset-out influence summary and sign-consistency report.

Interpretation
--------------
These are predictive incremental-information tests, not causal effects.
The effective independent time dimension is the number of common timestamps
(~57 in the v2.15 design), so uncertainty is organized around timestamp blocks
rather than treating the 5-asset panel rows as independent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm


# -----------------------------------------------------------------------------
# Numerics
# -----------------------------------------------------------------------------

def add_const(x: np.ndarray) -> np.ndarray:
    return sm.add_constant(np.asarray(x, float), has_constant="add")


def r2_from_fit(y: np.ndarray, fitted: np.ndarray) -> float:
    y = np.asarray(y, float)
    fitted = np.asarray(fitted, float)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst <= 1e-18:
        return float("nan")
    return float(1.0 - np.sum((y - fitted) ** 2) / sst)


def make_fe_design(symbol: np.ndarray, current: np.ndarray, h: np.ndarray,
                   include_variogram: bool = False,
                   asset_order: List[str] | None = None) -> Tuple[np.ndarray, List[str]]:
    """Asset fixed effects using one omitted baseline asset."""
    symbol = np.asarray(symbol).astype(str)
    current = np.asarray(current, float)
    h = np.asarray(h, float)
    assets = asset_order or sorted(pd.unique(symbol).tolist())
    if len(assets) < 2:
        raise ValueError("Need at least two assets for asset fixed effects.")
    baseline = assets[0]
    cols = [np.ones(len(symbol)), current]
    names = ["const", "current_log_rv"]
    for a in assets[1:]:
        cols.append((symbol == a).astype(float))
        names.append(f"FE_{a}")
    cols.append(h)
    names.append("H_neural_mean")
    if include_variogram:
        raise ValueError("Use make_asset_specific_design for variogram/joint specifications.")
    return np.column_stack(cols), names


def make_asset_specific_design(symbol: np.ndarray, current: np.ndarray, h: np.ndarray,
                               variogram: np.ndarray | None = None,
                               asset_order: List[str] | None = None) -> Tuple[np.ndarray, List[str], List[int]]:
    """Asset FE plus asset-specific H slopes; optionally add variogram H."""
    symbol = np.asarray(symbol).astype(str)
    current = np.asarray(current, float)
    h = np.asarray(h, float)
    assets = asset_order or sorted(pd.unique(symbol).tolist())
    if len(assets) < 2:
        raise ValueError("Need at least two assets.")
    cols = [np.ones(len(symbol)), current]
    names = ["const", "current_log_rv"]
    for a in assets[1:]:
        cols.append((symbol == a).astype(float))
        names.append(f"FE_{a}")

    h_indices: List[int] = []
    # Baseline asset slope
    cols.append(h * (symbol == assets[0]).astype(float))
    names.append(f"H_{assets[0]}")
    h_indices.append(len(names) - 1)
    for a in assets[1:]:
        cols.append(h * (symbol == a).astype(float))
        names.append(f"H_{a}")
        h_indices.append(len(names) - 1)

    if variogram is not None:
        cols.append(np.asarray(variogram, float))
        names.append("H_variogram")

    return np.column_stack(cols), names, h_indices


def fit_cluster_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray):
    model = sm.OLS(np.asarray(y, float), np.asarray(X, float)).fit(
        cov_type="cluster", cov_kwds={"groups": np.asarray(groups)}
    )
    return model


def fit_ols(y: np.ndarray, X: np.ndarray):
    return sm.OLS(np.asarray(y, float), np.asarray(X, float)).fit()


def circular_block_indices(n_time: int, block_len: int, reps: int,
                           rng: np.random.Generator) -> np.ndarray:
    n_time = int(n_time)
    reps = int(reps)
    L = max(1, min(int(block_len), n_time))
    nb = int(math.ceil(n_time / L))
    starts = rng.integers(0, n_time, size=(reps, nb), dtype=np.int64)
    offsets = np.arange(L, dtype=np.int64)[None, None, :]
    idx3 = (starts[:, :, None] + offsets) % n_time
    return idx3.reshape(reps, nb * L)[:, :n_time]


def percentile_ci(values: np.ndarray) -> Tuple[float, float]:
    z = np.asarray(values, float)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(z, 2.5)), float(np.percentile(z, 97.5))


# -----------------------------------------------------------------------------
# Data loading and construction
# -----------------------------------------------------------------------------

def load_v215_panel(input_root: Path, symbols: List[str]) -> pd.DataFrame:
    parts = []
    for symbol in symbols:
        p = input_root / f"{symbol}_rolling_estimates.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing v2.15 panel file: {p}")
        d = pd.read_csv(p)
        if "timestamp" not in d.columns:
            raise ValueError(f"Missing timestamp in {p}")
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d["symbol"] = symbol
        parts.append(d)
    panel = pd.concat(parts, ignore_index=True)
    needed = [
        "timestamp", "symbol", "H_neural_mean", "H_neural_seed_sd",
        "H_variogram", "future_log_rv_1d", "future_delta_log_rv_1d"
    ]
    missing = [c for c in needed if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # v2.17 reconstruction: current 1d log-RV = future log-RV - delta log-RV.
    panel["current_log_rv_1d"] = panel["future_log_rv_1d"] - panel["future_delta_log_rv_1d"]
    panel = panel[needed + ["current_log_rv_1d"]].copy()
    panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return panel


def complete_common_timestamps(panel: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    counts = panel.groupby("timestamp")["symbol"].nunique()
    common = counts[counts == len(symbols)].index
    x = panel[panel["timestamp"].isin(common)].copy()
    x = x.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return x


# -----------------------------------------------------------------------------
# Primary pooled FE
# -----------------------------------------------------------------------------

def fit_pooled_fe(panel: pd.DataFrame, symbols: List[str], include_variogram: bool = False):
    x = panel.dropna(subset=[
        "future_log_rv_1d", "current_log_rv_1d", "H_neural_mean"
    ] + (["H_variogram"] if include_variogram else [])).copy()
    y = x["future_log_rv_1d"].to_numpy(float)
    c = x["current_log_rv_1d"].to_numpy(float)
    h = x["H_neural_mean"].to_numpy(float)
    v = x["H_variogram"].to_numpy(float) if include_variogram else None
    X, names = make_fe_design(x["symbol"].to_numpy(), c, h, False, symbols)
    if include_variogram:
        X = np.column_stack([X, v])
        names = names + ["H_variogram"]
    groups = pd.factorize(x["timestamp"])[0]
    m = fit_cluster_ols(y, X, groups)
    ih = names.index("H_neural_mean")
    return x, m, names, ih, groups


def fit_pooled_model0(panel: pd.DataFrame, symbols: List[str]):
    x = panel.dropna(subset=["future_log_rv_1d", "current_log_rv_1d"]).copy()
    y = x["future_log_rv_1d"].to_numpy(float)
    c = x["current_log_rv_1d"].to_numpy(float)
    symbol = x["symbol"].to_numpy()
    assets = symbols
    cols = [np.ones(len(x)), c]
    for a in assets[1:]:
        cols.append((symbol == a).astype(float))
    X = np.column_stack(cols)
    groups = pd.factorize(x["timestamp"])[0]
    m = fit_cluster_ols(y, X, groups)
    return x, m, X


# -----------------------------------------------------------------------------
# Time-block bootstrap on common timestamp blocks
# -----------------------------------------------------------------------------

def pooled_time_block_bootstrap(panel: pd.DataFrame, symbols: List[str],
                                reps: int, block_len: int, seed: int):
    x = panel.dropna(subset=[
        "future_log_rv_1d", "current_log_rv_1d", "H_neural_mean"
    ]).copy()
    x = x.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    times = pd.Index(sorted(x["timestamp"].unique()))
    time_to_rows = {
        t: np.flatnonzero(x["timestamp"].to_numpy() == t) for t in times
    }
    rng = np.random.default_rng(seed)
    idx_time = circular_block_indices(len(times), block_len, reps, rng)
    betas = np.full(reps, np.nan, float)
    delta_r2 = np.full(reps, np.nan, float)

    sym = x["symbol"].to_numpy()
    y0 = x["future_log_rv_1d"].to_numpy(float)
    c0 = x["current_log_rv_1d"].to_numpy(float)
    h0 = x["H_neural_mean"].to_numpy(float)

    for b in range(reps):
        row_idx = np.concatenate([time_to_rows[times[k]] for k in idx_time[b]])
        y = y0[row_idx]
        c = c0[row_idx]
        h = h0[row_idx]
        s = sym[row_idx]
        # Current-only and current+H FE designs with the same sampled timestamps.
        cols0 = [np.ones(len(row_idx)), c]
        cols1 = [np.ones(len(row_idx)), c]
        for a in symbols[1:]:
            d = (s == a).astype(float)
            cols0.append(d); cols1.append(d)
        cols1.append(h)
        X0 = np.column_stack(cols0)
        X1 = np.column_stack(cols1)
        try:
            m0 = fit_ols(y, X0)
            m1 = fit_ols(y, X1)
            betas[b] = float(m1.params[-1])
            delta_r2[b] = float(m1.rsquared - m0.rsquared)
        except Exception:
            continue
    return betas, delta_r2


# -----------------------------------------------------------------------------
# Wild cluster bootstrap at common timestamps
# -----------------------------------------------------------------------------

def wild_cluster_bootstrap_beta(panel: pd.DataFrame, symbols: List[str],
                                reps: int, seed: int):
    x = panel.dropna(subset=[
        "future_log_rv_1d", "current_log_rv_1d", "H_neural_mean"
    ]).copy()
    x = x.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    y = x["future_log_rv_1d"].to_numpy(float)
    c = x["current_log_rv_1d"].to_numpy(float)
    h = x["H_neural_mean"].to_numpy(float)
    s = x["symbol"].to_numpy()
    X, names = make_fe_design(s, c, h, False, symbols)
    groups = pd.factorize(x["timestamp"])[0]
    m = fit_ols(y, X)
    resid = m.resid
    fitted = m.fittedvalues
    unique_g = np.unique(groups)
    rng = np.random.default_rng(seed)
    beta = np.full(reps, np.nan, float)
    ih = names.index("H_neural_mean")
    for b in range(reps):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(unique_g))
        w = signs[groups]
        y_star = fitted + resid * w
        try:
            mb = fit_ols(y_star, X)
            beta[b] = float(mb.params[ih])
        except Exception:
            continue
    return beta


# -----------------------------------------------------------------------------
# Leave-one-asset-out and joint tests
# -----------------------------------------------------------------------------

def leave_one_asset_out(panel: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    rows = []
    for omit in symbols:
        keep = [s for s in symbols if s != omit]
        d = panel[panel["symbol"].isin(keep)].copy()
        d = d.dropna(subset=["future_log_rv_1d", "current_log_rv_1d", "H_neural_mean"])
        y = d["future_log_rv_1d"].to_numpy(float)
        c = d["current_log_rv_1d"].to_numpy(float)
        h = d["H_neural_mean"].to_numpy(float)
        X, names = make_fe_design(d["symbol"].to_numpy(), c, h, False, keep)
        groups = pd.factorize(d["timestamp"])[0]
        m = fit_cluster_ols(y, X, groups)
        ih = names.index("H_neural_mean")
        rows.append({
            "omitted_asset": omit,
            "assets_in_model": ",".join(keep),
            "n_rows": int(len(d)),
            "n_common_timestamps": int(d["timestamp"].nunique()),
            "beta_H": float(m.params[ih]),
            "se_H_cluster_timestamp": float(m.bse[ih]),
            "p_H_cluster_timestamp": float(m.pvalues[ih]),
            "ci95_low_normal_cluster": float(m.conf_int(alpha=0.05)[ih, 0]),
            "ci95_high_normal_cluster": float(m.conf_int(alpha=0.05)[ih, 1]),
        })
    return pd.DataFrame(rows)


def joint_asset_specific_test(panel: pd.DataFrame, symbols: List[str]) -> Dict[str, object]:
    d = panel.dropna(subset=["future_log_rv_1d", "current_log_rv_1d", "H_neural_mean"]).copy()
    y = d["future_log_rv_1d"].to_numpy(float)
    c = d["current_log_rv_1d"].to_numpy(float)
    h = d["H_neural_mean"].to_numpy(float)
    X, names, h_idx = make_asset_specific_design(d["symbol"].to_numpy(), c, h, None, symbols)
    groups = pd.factorize(d["timestamp"])[0]
    m = fit_cluster_ols(y, X, groups)
    R = np.zeros((len(h_idx), X.shape[1]), float)
    for j, idx in enumerate(h_idx):
        R[j, idx] = 1.0
    wt = m.wald_test(R, use_f=True)
    stat = float(np.asarray(wt.statistic).squeeze())
    p = float(np.asarray(wt.pvalue).squeeze())
    df_num = int(len(h_idx))
    df_denom = float(wt.df_denom) if hasattr(wt, "df_denom") else float("nan")
    coef_rows = []
    for idx in h_idx:
        coef_rows.append({"parameter": names[idx], "beta": float(m.params[idx]), "p_cluster": float(m.pvalues[idx])})
    return {
        "n_rows": int(len(d)),
        "n_common_timestamps": int(d["timestamp"].nunique()),
        "joint_F": stat,
        "joint_p_cluster_timestamp": p,
        "joint_df_num": df_num,
        "joint_df_denom": df_denom,
        "asset_specific_H_coefficients": coef_rows,
    }


def pooled_variogram_control(panel: pd.DataFrame, symbols: List[str]) -> Dict[str, float]:
    d = panel.dropna(subset=[
        "future_log_rv_1d", "current_log_rv_1d", "H_neural_mean", "H_variogram"
    ]).copy()
    y = d["future_log_rv_1d"].to_numpy(float)
    c = d["current_log_rv_1d"].to_numpy(float)
    h = d["H_neural_mean"].to_numpy(float)
    v = d["H_variogram"].to_numpy(float)
    cols = [np.ones(len(d)), c]
    names = ["const", "current_log_rv"]
    s = d["symbol"].to_numpy()
    for a in symbols[1:]:
        cols.append((s == a).astype(float)); names.append(f"FE_{a}")
    cols.extend([h, v]); names.extend(["H_neural_mean", "H_variogram"])
    X = np.column_stack(cols)
    groups = pd.factorize(d["timestamp"])[0]
    m = fit_cluster_ols(y, X, groups)
    ih = names.index("H_neural_mean")
    iv = names.index("H_variogram")
    return {
        "n_rows": int(len(d)),
        "beta_H_neural": float(m.params[ih]),
        "p_H_neural_cluster": float(m.pvalues[ih]),
        "beta_H_variogram": float(m.params[iv]),
        "p_H_variogram_cluster": float(m.pvalues[iv]),
        "R2": float(m.rsquared),
    }


# -----------------------------------------------------------------------------
# Visuals / outputs
# -----------------------------------------------------------------------------

def save_leave_one_out_plot(df: pd.DataFrame, out_path: Path) -> None:
    z = df.sort_values("omitted_asset").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    y = np.arange(len(z))
    lo = z["ci95_low_normal_cluster"].to_numpy(float)
    hi = z["ci95_high_normal_cluster"].to_numpy(float)
    beta = z["beta_H"].to_numpy(float)
    ax.errorbar(beta, y, xerr=[beta - lo, hi - beta], fmt="o", capsize=4)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels(z["omitted_asset"])
    ax.set_xlabel("Pooled FE beta_H (1-day), leaving out one asset")
    ax.set_title("Leave-one-asset-out robustness")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_bootstrap_hist(values_a: np.ndarray, values_b: np.ndarray, observed: float,
                        out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    va = values_a[np.isfinite(values_a)]
    vb = values_b[np.isfinite(values_b)]
    if len(va): ax.hist(va, bins=45, alpha=0.55, label="Common time-block bootstrap")
    if len(vb): ax.hist(vb, bins=45, alpha=0.55, label="Wild cluster bootstrap")
    ax.axvline(observed, linewidth=1.2, label=f"Observed beta_H={observed:.3f}")
    ax.set_xlabel("beta_H")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def json_safe(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input_root", default="inputs/results_hurst_v2_15_empirical_crypto_multiasset")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"])
    ap.add_argument("--bootstrap", type=int, default=5000, help="common timestamp-block bootstrap repetitions")
    ap.add_argument("--wild_bootstrap", type=int, default=5000, help="wild cluster bootstrap repetitions")
    ap.add_argument("--block_len", type=int, default=4, help="timestamp blocks; 4 means ~68 days")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--out_dir", default="results_hurst_v2_19_final_statistical_robustness")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = list(args.symbols)

    print("=" * 88)
    print("HurstNeuralEstimator_v2_19_final_statistical_robustness")
    print("=" * 88)
    print("Focus: 1-day incremental predictive information of Neural H")
    print("Input: v2.15 non-overlapping empirical panel (context=384)")
    print("Neural retraining: NONE")
    print(f"Assets: {symbols}")
    print(f"Common timestamp-block bootstrap={args.bootstrap}")
    print(f"Wild cluster bootstrap={args.wild_bootstrap}")
    print(f"Block length={args.block_len} timestamps")
    print("Inference cluster: common timestamp across assets")

    panel = load_v215_panel(input_root, symbols)
    panel = complete_common_timestamps(panel, symbols)
    if panel.empty:
        raise RuntimeError("No common timestamps across all requested assets.")

    panel.to_csv(out_dir / "v219_common_timestamp_panel.csv", index=False)
    n_time = int(panel["timestamp"].nunique())
    n_rows = int(len(panel))
    print(f"Common timestamps={n_time}, rows={n_rows}")

    # Primary pooled FE current + H.
    x1, m1, names1, ih1, groups1 = fit_pooled_fe(panel, symbols, include_variogram=False)
    x0, m0, X0 = fit_pooled_model0(panel, symbols)
    beta_obs = float(m1.params[ih1])
    p_obs = float(m1.pvalues[ih1])
    delta_r2_obs = float(m1.rsquared - m0.rsquared)
    print(f"\nPRIMARY POOLED FE: beta_H={beta_obs:+.6f} | p_cluster={p_obs:.6g} | ΔR²={delta_r2_obs:+.6f}")

    # Common timestamp-block bootstrap.
    print("Running common timestamp-block bootstrap ...")
    boot_beta, boot_dr2 = pooled_time_block_bootstrap(
        panel, symbols, args.bootstrap, args.block_len, args.seed + 101
    )
    boot_beta_ci = percentile_ci(boot_beta)
    boot_dr2_ci = percentile_ci(boot_dr2)
    print(f"  time-block bootstrap beta CI={boot_beta_ci}")
    print(f"  time-block bootstrap ΔR² CI={boot_dr2_ci}")

    # Wild cluster bootstrap.
    print("Running wild cluster bootstrap ...")
    wild_beta = wild_cluster_bootstrap_beta(
        panel, symbols, args.wild_bootstrap, args.seed + 202
    )
    wild_beta_ci = percentile_ci(wild_beta)
    wv = wild_beta[np.isfinite(wild_beta)]
    wild_two_sided_p = float(2.0 * min(np.mean(wv <= 0.0), np.mean(wv >= 0.0))) if len(wv) else float("nan")
    wild_two_sided_p = min(wild_two_sided_p, 1.0) if np.isfinite(wild_two_sided_p) else wild_two_sided_p
    print(f"  wild bootstrap beta CI={wild_beta_ci} | two-sided sign p~={wild_two_sided_p:.6g}")

    # Leave-one-asset-out.
    loo = leave_one_asset_out(panel, symbols)
    loo.to_csv(out_dir / "v219_leave_one_asset_out.csv", index=False)
    print("\nLEAVE-ONE-ASSET-OUT")
    for r in loo.itertuples():
        print(f"  omit {r.omitted_asset:8s}: beta_H={r.beta_H:+.5f} | p={r.p_H_cluster_timestamp:.6g}")

    # Joint asset-specific H significance.
    joint = joint_asset_specific_test(panel, symbols)
    print("\nJOINT ASSET-SPECIFIC H TEST")
    print(f"  joint F={joint['joint_F']:.6f} | p={joint['joint_p_cluster_timestamp']:.6g} | df={joint['joint_df_num']}")
    for item in joint["asset_specific_H_coefficients"]:
        print(f"  {item['parameter']:16s}: beta={item['beta']:+.5f} | p={item['p_cluster']:.6g}")

    # Variogram-controlled robustness.
    vario = pooled_variogram_control(panel, symbols)
    print("\nVARIOGRAM-CONTROLLED POOLED FE")
    print(f"  beta_H_neural={vario['beta_H_neural']:+.6f} | p={vario['p_H_neural_cluster']:.6g}")
    print(f"  beta_H_variogram={vario['beta_H_variogram']:+.6f} | p={vario['p_H_variogram_cluster']:.6g}")

    # Robustness summary.
    signs_positive = bool(np.all(loo["beta_H"].to_numpy(float) > 0.0) and beta_obs > 0.0)
    loo_beta_min = float(np.min(loo["beta_H"].to_numpy(float)))
    loo_beta_max = float(np.max(loo["beta_H"].to_numpy(float)))
    loo_p_max = float(np.max(loo["p_H_cluster_timestamp"].to_numpy(float)))
    pooled_supported = bool(
        beta_obs > 0 and p_obs < 0.05 and boot_beta_ci[0] > 0 and wild_beta_ci[0] > 0
    )

    primary = {
        "n_assets": len(symbols),
        "n_common_timestamps": n_time,
        "n_rows": n_rows,
        "beta_H_pooled_FE_cluster_timestamp": beta_obs,
        "p_H_pooled_FE_cluster_timestamp": p_obs,
        "incremental_R2_H_given_current": delta_r2_obs,
        "timeblock_bootstrap_beta_ci_low": boot_beta_ci[0],
        "timeblock_bootstrap_beta_ci_high": boot_beta_ci[1],
        "timeblock_bootstrap_incremental_R2_ci_low": boot_dr2_ci[0],
        "timeblock_bootstrap_incremental_R2_ci_high": boot_dr2_ci[1],
        "wild_cluster_bootstrap_beta_ci_low": wild_beta_ci[0],
        "wild_cluster_bootstrap_beta_ci_high": wild_beta_ci[1],
        "wild_cluster_bootstrap_two_sided_sign_p_approx": wild_two_sided_p,
        "leave_one_asset_out_beta_min": loo_beta_min,
        "leave_one_asset_out_beta_max": loo_beta_max,
        "leave_one_asset_out_max_p": loo_p_max,
        "all_leave_one_asset_out_betas_positive": signs_positive,
        "pooled_effect_supported_by_p_and_both_CIs": pooled_supported,
        "joint_asset_specific_H_F": joint["joint_F"],
        "joint_asset_specific_H_p": joint["joint_p_cluster_timestamp"],
        "variogram_control_beta_H_neural": vario["beta_H_neural"],
        "variogram_control_p_H_neural": vario["p_H_neural_cluster"],
    }

    pd.DataFrame([primary]).to_csv(out_dir / "v219_primary_robustness_summary.csv", index=False)
    pd.DataFrame(joint["asset_specific_H_coefficients"]).to_csv(
        out_dir / "v219_joint_asset_specific_H_coefficients.csv", index=False
    )

    # Store bootstrap arrays for reproducibility.
    np.save(out_dir / "v219_common_timeblock_bootstrap_beta_H.npy", boot_beta)
    np.save(out_dir / "v219_common_timeblock_bootstrap_incremental_R2.npy", boot_dr2)
    np.save(out_dir / "v219_wild_cluster_bootstrap_beta_H.npy", wild_beta)

    # Figures.
    save_leave_one_out_plot(loo, out_dir / "v219_leave_one_asset_out_forest.png")
    save_bootstrap_hist(
        boot_beta, wild_beta, beta_obs,
        out_dir / "v219_beta_H_bootstrap_distributions.png",
        "Pooled FE beta_H uncertainty: time-block vs wild cluster bootstrap",
    )

    # Compact config / conclusions.
    config = {
        "symbols": symbols,
        "input_root": str(input_root),
        "bootstrap": int(args.bootstrap),
        "wild_bootstrap": int(args.wild_bootstrap),
        "block_len": int(args.block_len),
        "seed": int(args.seed),
        "focus": "1-day future log-RV ~ current 1-day log-RV + Neural H",
        "no_neural_retraining": True,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    conclusions = {
        "primary_positive_and_supported": pooled_supported,
        "all_leave_one_asset_out_betas_positive": signs_positive,
        "joint_asset_specific_H_rejection": bool(joint["joint_p_cluster_timestamp"] < 0.05),
        "variogram_controlled_H_rejection": bool(vario["p_H_neural_cluster"] < 0.05),
        "interpretation": (
            "The analysis tests incremental predictive information of Neural H beyond matched-horizon current realized volatility. "
            "It does not establish causality. Inference is organized around common timestamps because the five assets share the same "
            "57 non-overlapping empirical windows."
        ),
    }
    with open(out_dir / "v219_conclusions.json", "w", encoding="utf-8") as f:
        json.dump(conclusions, f, indent=2)

    print("\n" + "=" * 88)
    print("v2.19 FINAL STATISTICAL ROBUSTNESS COMPLETE")
    print("=" * 88)
    print(f"Pooled beta_H={beta_obs:+.6f}, p={p_obs:.6g}, ΔR²={delta_r2_obs:+.6f}")
    print(f"Time-block bootstrap 95% CI=[{boot_beta_ci[0]:+.6f}, {boot_beta_ci[1]:+.6f}]")
    print(f"Wild cluster bootstrap 95% CI=[{wild_beta_ci[0]:+.6f}, {wild_beta_ci[1]:+.6f}]")
    print(f"Leave-one-out all positive: {signs_positive}; max p={loo_p_max:.6g}")
    print(f"Joint asset-specific H p={joint['joint_p_cluster_timestamp']:.6g}")
    print(f"Variogram-controlled H p={vario['p_H_neural_cluster']:.6g}")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
