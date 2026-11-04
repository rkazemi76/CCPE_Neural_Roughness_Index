#!/usr/bin/env python3
"""v2.23: chronological OOS benchmark for Neural H with a fitted GARCH(1,1) baseline.

Changes from v2.21:
- Naive persistence baseline forecasts next-day RV with current-day RV directly.
- HAR models use expanding-window pooled OLS with asset fixed effects.
- QLIKE uses positive RV forecasts with a train-only Duan smearing correction for fitted log-RV models.
- Adds a univariate Gaussian-QMLE GARCH(1,1) model fitted on trailing minute returns for each asset and OOS timestamp.
- GARCH variance forecasts are aggregated over the next 1,440 one-minute periods to target next-day realized variance.
- Adds paired moving-block bootstrap CIs for timestamp-clustered loss differentials, including Neural-H vs GARCH.

No additional volatility package (e.g. `arch`) is required; estimation uses SciPy.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit, logit


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("v215_module_for_v223", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load v2.15 module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def contiguous_rv(returns, ts_ns, end_ns, minutes):
    """Past RV ending at end_ns; require exactly contiguous minute returns."""
    minute_ns = 60 * 10**9
    j = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
    if j < 0 or ts_ns[j] != end_ns:
        return np.nan
    n = int(minutes); i = j - n + 1
    if i < 0 or ts_ns[j] - ts_ns[i] != (n - 1) * minute_ns:
        return np.nan
    if np.any(np.diff(ts_ns[i:j+1]) != minute_ns):
        return np.nan
    return max(float(np.sum(returns[i:j+1] ** 2)), 1e-18)


def future_rv(returns, ts_ns, end_ns, minutes):
    """RV over the next `minutes` returns strictly after end_ns."""
    minute_ns = 60 * 10**9
    j = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
    if j < 0 or ts_ns[j] != end_ns:
        return np.nan
    i, k = j + 1, j + 1 + int(minutes)
    if k > len(returns):
        return np.nan
    if ts_ns[i] - ts_ns[j] != minute_ns or np.any(np.diff(ts_ns[i:k]) != minute_ns):
        return np.nan
    return max(float(np.sum(returns[i:k] ** 2)), 1e-18)


def make_design(frame, features, asset_categories):
    x = frame[list(features)].astype(float).reset_index(drop=True).copy()
    cats = pd.Categorical(frame["symbol"], categories=asset_categories)
    dummies = pd.get_dummies(cats, prefix="asset", drop_first=True, dtype=float)
    dummies = dummies.reset_index(drop=True)
    return sm.add_constant(pd.concat([x, dummies], axis=1), has_constant="add")


def fit_log_model(train, test, features, asset_categories):
    xtr = make_design(train, features, asset_categories)
    xte = make_design(test, features, asset_categories)
    ytr = train["target_logRV"].astype(float).to_numpy()
    fit = sm.OLS(ytr, xtr.to_numpy()).fit()
    pred_train = fit.predict(xtr.to_numpy())
    pred_test = fit.predict(xte.to_numpy())
    # Duan smearing, estimated on training residuals only (no OOS leakage).
    resid = np.clip(ytr - pred_train, -40, 30)
    smear = float(np.mean(np.exp(resid)))
    if not np.isfinite(smear) or smear <= 0:
        smear = 1.0
    return pred_test, smear


def qlike(actual_rv, forecast_rv):
    y = np.maximum(np.asarray(actual_rv, float), 1e-18)
    p = np.maximum(np.asarray(forecast_rv, float), 1e-18)
    ratio = y / p
    return ratio - np.log(ratio) - 1.0



def _garch_decode(theta):
    """Map unconstrained parameters to omega>0, alpha>=0, beta>=0, alpha+beta<0.999."""
    persistence_cap = 0.999
    omega = float(np.exp(np.clip(theta[0], -700, 100)))
    alpha = float(persistence_cap * expit(theta[1]))
    beta = float((persistence_cap - alpha) * expit(theta[2]))
    return omega, alpha, beta


def _garch_variances(y, omega, alpha, beta):
    """Conditional variance path for scaled returns, using a sample-variance start."""
    y = np.asarray(y, dtype=np.float64)
    v0 = max(float(np.var(y)), 1e-10)
    h = np.empty(y.size, dtype=np.float64)
    h[0] = v0
    if y.size > 1:
        # lfilter implements h[t] = omega + alpha*y[t-1]^2 + beta*h[t-1].
        innovations = omega + alpha * np.square(y[:-1])
        h[1:] = lfilter([1.0], [1.0, -beta], innovations, zi=[beta * h[0]])[0]
    return np.clip(h, 1e-10, 1e12)


def _garch_nll(theta, y):
    omega, alpha, beta = _garch_decode(theta)
    if not (np.isfinite(omega) and omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta < 0.999):
        return 1e100
    h = _garch_variances(y, omega, alpha, beta)
    val = 0.5 * float(np.sum(np.log(h) + np.square(y) / h))
    return val if np.isfinite(val) else 1e100


def _theta_from_garch(omega, alpha, beta):
    cap = 0.999
    aa = np.clip(alpha / cap, 1e-6, 1 - 1e-6)
    bb = np.clip(beta / max(cap - alpha, 1e-6), 1e-6, 1 - 1e-6)
    return np.array([np.log(max(omega, 1e-12)), logit(aa), logit(bb)], dtype=float)


def garch11_forecast_rv(returns, ts_ns, end_ns, horizon_minutes=1440,
                        lookback_days=90, min_obs=20000, maxiter=80):
    """Fit zero-mean Gaussian QMLE GARCH(1,1) on contiguous past 1m returns and forecast next-day RV.

    Only returns with timestamps <= end_ns enter parameter estimation. A contiguous suffix is
    used to respect the model's unit time step. If the optimizer fails, a documented fixed-
    parameter GARCH fallback (alpha=.05, beta=.94) ensures deterministic coverage; the status
    is retained in forecast output so fallback rows can be audited.
    """
    returns = np.asarray(returns, dtype=np.float64)
    ts_ns = np.asarray(ts_ns, dtype=np.int64)
    minute_ns = 60 * 10**9
    j = int(np.searchsorted(ts_ns, int(end_ns), side="right") - 1)
    if j < 0 or ts_ns[j] != int(end_ns):
        raise ValueError("No minute return exactly aligned to the forecast timestamp")

    # Limit estimation sample to trailing lookback_days and the latest gap-free suffix.
    desired_start_ns = int(end_ns) - int(lookback_days) * 1440 * minute_ns
    lo = int(np.searchsorted(ts_ns, desired_start_ns, side="left"))
    start = j
    while start > lo and ts_ns[start] - ts_ns[start - 1] == minute_ns:
        start -= 1
    yraw = returns[start:j + 1]
    if yraw.size < int(min_obs):
        raise ValueError(f"Insufficient contiguous GARCH history: {yraw.size} < {min_obs} returns")
    if not np.isfinite(yraw).all():
        raise ValueError("Non-finite return in GARCH estimation window")

    # Rescale tiny minute log returns to improve likelihood optimization conditioning.
    scale = 1000.0
    y = yraw * scale
    sample_var = max(float(np.var(y)), 1e-8)
    alpha0, beta0 = 0.05, 0.90
    omega0 = max(sample_var * (1.0 - alpha0 - beta0), sample_var * 1e-8)
    theta0 = _theta_from_garch(omega0, alpha0, beta0)
    bounds = [
        (np.log(sample_var * 1e-10), np.log(sample_var)),
        (-10.0, 10.0),
        (-10.0, 10.0),
    ]
    try:
        opt = minimize(_garch_nll, theta0, args=(y,), method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": int(maxiter), "ftol": 1e-8, "maxls": 20})
        if np.isfinite(opt.fun) and np.all(np.isfinite(opt.x)) and opt.fun < 1e90:
            omega, alpha, beta = _garch_decode(opt.x)
            status = "ok" if opt.success else "optimizer_nonconverged"
        else:
            raise RuntimeError("non-finite GARCH optimizer solution")
    except Exception:
        alpha, beta = 0.05, 0.94
        omega = max(sample_var * (1.0 - alpha - beta), 1e-10)
        status = "fallback_fixed_parameters"

    h = _garch_variances(y, omega, alpha, beta)
    # One-step-ahead conditional variance given the last observed return.
    h_next = omega + alpha * float(y[-1] ** 2) + beta * float(h[-1])
    h_next = max(float(h_next), 1e-10)
    persistence = min(max(alpha + beta, 0.0), 0.999999)
    unconditional = omega / max(1.0 - alpha - beta, 1e-8)
    k = int(horizon_minutes)
    # Sum h_{t+1},...,h_{t+k}; future expected variance follows the GARCH AR(1) recursion.
    if abs(1.0 - persistence) < 1e-8:
        forecast_scaled_rv = k * h_next
    else:
        forecast_scaled_rv = k * unconditional + (h_next - unconditional) * (1.0 - persistence ** k) / (1.0 - persistence)
    forecast_rv = max(float(forecast_scaled_rv / (scale ** 2)), 1e-18)
    return {
        "forecast_rv": forecast_rv,
        "nobs": int(y.size),
        "omega_scaled": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "persistence": float(alpha + beta),
        "status": status,
        "optimizer_success": bool(status == "ok"),
    }

def model_metrics(g):
    err = g["actual_logRV"].to_numpy() - g["predicted_logRV"].to_numpy()
    corr = np.corrcoef(g["actual_logRV"], g["predicted_logRV"])[0, 1]
    return {"model": g["model"].iloc[0], "n_oos": len(g),
            "n_timestamps": g["timestamp"].nunique(),
            "RMSE_logRV": float(np.sqrt(np.mean(err**2))),
            "MAE_logRV": float(np.mean(np.abs(err))),
            "QLIKE_RV": float(np.mean(qlike(g["actual_rv"], g["predicted_rv_qlike"]))),
            "Corr_logRV": float(corr) if np.isfinite(corr) else np.nan}


def moving_block_bootstrap_diff(pred, model_a, model_b, loss_name, block_len, reps, rng):
    """CI for loss(model_a)-loss(model_b), resampling consecutive timestamp blocks.

    All assets within a sampled timestamp travel together, preserving contemporaneous
    cross-asset dependence. The block bootstrap also retains short-range time dependence.
    """
    def get_loss(g, model):
        if loss_name == "MSE":
            return (g.actual_logRV.to_numpy() - g[f"predlog__{model}"].to_numpy()) ** 2
        if loss_name == "MAE":
            return np.abs(g.actual_logRV.to_numpy() - g[f"predlog__{model}"].to_numpy())
        if loss_name == "QLIKE":
            return qlike(g.actual_rv.to_numpy(), g[f"predrv__{model}"].to_numpy())
        raise ValueError(loss_name)

    # Each row has unique (timestamp,symbol); index and resample timestamp groups as clusters.
    time_values = sorted(pred["timestamp"].drop_duplicates().tolist())
    groups = [pred.index[pred.timestamp == t].to_numpy() for t in time_values]
    la = get_loss(pred, model_a); lb = get_loss(pred, model_b)
    diff = la - lb
    point = float(np.mean(diff))
    n = len(groups); L = max(1, min(int(block_len), n))
    boot = np.empty(int(reps), dtype=float)
    # Rows are laid out timestamp-major; concatenate whole timestamp groups per sampled block.
    for b in range(int(reps)):
        picked = []
        while len(picked) < n:
            start = int(rng.integers(0, n - L + 1))
            picked.extend(range(start, start + L))
        tids = [j for j in picked[:n]]
        rowidx = np.concatenate([groups[j] for j in tids])
        boot[b] = float(np.mean(diff[rowidx]))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"model_a": model_a, "model_b": model_b, "loss": loss_name,
            "delta_loss_a_minus_b": point, "ci95_lower": float(lo), "ci95_upper": float(hi),
            "block_length_timestamps": L, "n_oos_timestamps": n, "bootstrap_reps": int(reps)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_root", default="inputs/results_hurst_v2_15_empirical_crypto_multiasset")
    ap.add_argument("--v215_script", default="code/HurstNeuralEstimator_v2_15_empirical_crypto_multiasset.py")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"])
    ap.add_argument("--start_date", default="2024-01-01")
    ap.add_argument("--end_date", default="2026-09-01")
    ap.add_argument("--download_dir", default="binance_cache_v2_15")
    ap.add_argument("--initial_train_timestamps", type=int, default=30)
    ap.add_argument("--bootstrap_reps", type=int, default=10000)
    ap.add_argument("--bootstrap_block_length", type=int, default=3)
    ap.add_argument("--garch_lookback_days", type=int, default=90)
    ap.add_argument("--garch_min_obs", type=int, default=20000)
    ap.add_argument("--garch_maxiter", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--out_dir", default="results_hurst_v2_23_walkforward_GARCH_benchmark")
    args = ap.parse_args()

    root, out = Path(args.input_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    v215 = load_module(Path(args.v215_script))
    panel_frames=[]
    for sym in args.symbols:
        p = root / f"{sym}_rolling_estimates.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing H panel: {p}")
        d = pd.read_csv(p); d["symbol"] = sym; panel_frames.append(d)
    panel = pd.concat(panel_frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel = panel.sort_values(["timestamp","symbol"]).reset_index(drop=True)
    all_rows=[]
    minute_data = {}
    for sym in args.symbols:
        print(f"\nLoading minute data: {sym}")
        cfg = v215.Config(symbols=(sym,), start_date=args.start_date, end_date=args.end_date,
                          download_dir=args.download_dir, out_dir=str(out))
        df = v215.load_symbol_1m(sym, cfg)
        close = df["close"].to_numpy(np.float64)
        ret = np.diff(np.log(np.maximum(close, 1e-300)))
        rts = pd.to_datetime(df["timestamp"], utc=True).iloc[1:]
        ts_ns = rts.astype("int64").to_numpy()
        minute_data[sym] = (ret, ts_ns)
        sub = panel[panel.symbol == sym].copy()
        vals=[]
        for row in sub.itertuples(index=False):
            end_ns = int(pd.Timestamp(row.timestamp).value)
            vals.append((contiguous_rv(ret, ts_ns, end_ns, 1440),
                         contiguous_rv(ret, ts_ns, end_ns, 7*1440),
                         contiguous_rv(ret, ts_ns, end_ns, 30*1440),
                         future_rv(ret, ts_ns, end_ns, 1440)))
        sub[["rv_1d","rv_7d","rv_30d","target_rv"]] = np.asarray(vals, float)
        good = np.isfinite(sub[["rv_1d","rv_7d","rv_30d","target_rv"]]).all(axis=1).sum()
        print(f"  usable rows with complete windows: {int(good)}/{len(sub)}")
        all_rows.append(sub)

    data = pd.concat(all_rows, ignore_index=True)
    for c in ["rv_1d","rv_7d","rv_30d","target_rv"]:
        data[c] = np.maximum(data[c].astype(float), 1e-18)
    data["target_logRV"] = np.log(data["target_rv"])
    data["logRV_1d"] = np.log(data["rv_1d"])
    data["logRV_7d"] = np.log(data["rv_7d"])
    data["logRV_30d"] = np.log(data["rv_30d"])
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    valid_cols = ["target_logRV","logRV_1d","logRV_7d","logRV_30d","H_neural_mean","H_variogram"]
    data = data.loc[np.isfinite(data[valid_cols]).all(axis=1)].sort_values(["timestamp","symbol"]).reset_index(drop=True)
    timestamps = sorted(data.timestamp.drop_duplicates().tolist())
    n0 = int(args.initial_train_timestamps)
    if len(timestamps) <= n0 + 5:
        raise RuntimeError(f"Only {len(timestamps)} usable timestamps; need more than {n0+5}.")

    features = {
        "HAR_RV": ["logRV_1d","logRV_7d","logRV_30d"],
        "HAR_RV_plus_Neural_H": ["logRV_1d","logRV_7d","logRV_30d","H_neural_mean"],
        "HAR_RV_plus_Variogram_H": ["logRV_1d","logRV_7d","logRV_30d","H_variogram"],
    }
    categories = sorted(args.symbols)
    pred_rows=[]
    for ti in range(n0, len(timestamps)):
        t = timestamps[ti]
        train = data[data.timestamp < t].copy()
        test = data[data.timestamp == t].copy()
        if train.timestamp.nunique() < n0 or test.empty:
            continue
        # Direct persistence benchmark: forecast next-day RV by current-day RV (no fitted regression).
        for _, r in test.iterrows():
            pred_rows.append({"timestamp":t,"symbol":r.symbol,"model":"Naive_Current_RV",
                              "actual_logRV":r.target_logRV,"actual_rv":r.target_rv,
                              "predicted_logRV":r.logRV_1d,"predicted_rv_qlike":r.rv_1d,
                              "smearing_factor":1.0,"H_neural_mean":r.H_neural_mean,"H_variogram":r.H_variogram})
        for name, cols in features.items():
            if len(train) < max(25, 5*(len(cols)+1)):
                continue
            try:
                pred_log, smear = fit_log_model(train, test, cols, categories)
            except Exception as e:
                warnings.warn(f"Skipping {name} at {t}: {e}"); continue
            pred_rv = np.exp(np.clip(pred_log, -40, 30)) * smear
            pred_rv = np.maximum(pred_rv, 1e-18)
            for j, (_, r) in enumerate(test.iterrows()):
                pred_rows.append({"timestamp":t,"symbol":r.symbol,"model":name,
                                  "actual_logRV":r.target_logRV,"actual_rv":r.target_rv,
                                  "predicted_logRV":float(pred_log[j]),"predicted_rv_qlike":float(pred_rv[j]),
                                  "smearing_factor":smear,"H_neural_mean":r.H_neural_mean,"H_variogram":r.H_variogram})
        # Asset-specific GARCH(1,1), fitted only to minute returns available at this timestamp.
        for _, r in test.iterrows():
            ret, ts_ns = minute_data[r.symbol]
            try:
                gf = garch11_forecast_rv(
                    ret, ts_ns, int(pd.Timestamp(t).value), horizon_minutes=1440,
                    lookback_days=args.garch_lookback_days, min_obs=args.garch_min_obs,
                    maxiter=args.garch_maxiter)
                pred_rows.append({"timestamp":t,"symbol":r.symbol,"model":"GARCH11_RV",
                                  "actual_logRV":r.target_logRV,"actual_rv":r.target_rv,
                                  "predicted_logRV":float(np.log(gf["forecast_rv"])),
                                  "predicted_rv_qlike":float(gf["forecast_rv"]),
                                  "smearing_factor":1.0,"H_neural_mean":r.H_neural_mean,"H_variogram":r.H_variogram,
                                  "garch_nobs":gf["nobs"],"garch_omega_scaled":gf["omega_scaled"],
                                  "garch_alpha":gf["alpha"],"garch_beta":gf["beta"],
                                  "garch_persistence":gf["persistence"],"garch_status":gf["status"]})
            except Exception as e:
                warnings.warn(f"GARCH11 forecast failed for {r.symbol} at {t}: {e}")
    pred = pd.DataFrame(pred_rows)
    if pred.empty:
        raise RuntimeError("No OOS predictions produced.")
    pred.to_csv(out/"v223_walkforward_predictions.csv", index=False)
    metrics = pd.DataFrame([model_metrics(g) for _, g in pred.groupby("model", sort=False)])
    coverage = pred.groupby("model").agg(n_oos=("symbol", "size"), n_timestamps=("timestamp", "nunique")).reset_index()
    coverage.to_csv(out/"v223_oos_coverage.csv", index=False)
    if "garch_status" in pred.columns:
        pred.loc[pred.model == "GARCH11_RV", "garch_status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count").to_csv(out/"v223_garch_fit_status.csv", index=False)
    order = ["Naive_Current_RV","HAR_RV","GARCH11_RV","HAR_RV_plus_Neural_H","HAR_RV_plus_Variogram_H"]
    metrics["_ord"] = metrics.model.map({m:i for i,m in enumerate(order)})
    metrics = metrics.sort_values("_ord").drop(columns="_ord")
    metrics.to_csv(out/"v223_oos_metrics.csv", index=False)
    by_asset = pd.DataFrame([model_metrics(g) | {"symbol": g["symbol"].iloc[0]} for (_, _), g in pred.groupby(["symbol", "model"], sort=False)])
    by_asset.to_csv(out/"v223_oos_metrics_by_asset.csv", index=False)

    # Wide paired file for convenient inspection and downstream analysis.
    wide = pred.pivot_table(index=["timestamp","symbol"], columns="model",
                            values=["predicted_logRV","predicted_rv_qlike"], aggfunc="first")
    wide.columns = [f"{a}__{b}" for a,b in wide.columns]
    wide = wide.reset_index()
    actuals = pred.drop_duplicates(["timestamp","symbol"])[["timestamp","symbol","actual_logRV","actual_rv"]]
    wide = wide.merge(actuals, on=["timestamp","symbol"], how="left")
    wide.to_csv(out/"v223_paired_oos_predictions.csv", index=False)

    # For bootstrap arrays, arrange each row as one timestamp-symbol forecast and make wide loss columns.
    for m in order:
        if m not in pred.model.unique():
            raise RuntimeError(f"Expected model {m} has no predictions.")
    bootdf = actuals.copy()
    for m in order:
        g = pred[pred.model == m][["timestamp","symbol","predicted_logRV","predicted_rv_qlike"]].copy()
        g = g.rename(columns={"predicted_logRV":f"predlog__{m}","predicted_rv_qlike":f"predrv__{m}"})
        bootdf = bootdf.merge(g, on=["timestamp","symbol"], how="inner")
    rng = np.random.default_rng(args.seed)
    comparisons = [("HAR_RV_plus_Neural_H","Naive_Current_RV"),
                   ("HAR_RV_plus_Neural_H","HAR_RV"),
                   ("HAR_RV_plus_Neural_H","GARCH11_RV"),
                   ("HAR_RV_plus_Neural_H","HAR_RV_plus_Variogram_H"),
                   ("GARCH11_RV","HAR_RV")]
    boot_rows=[]
    for a,b in comparisons:
        for loss in ["MSE","MAE","QLIKE"]:
            boot_rows.append(moving_block_bootstrap_diff(bootdf, a, b, loss,
                args.bootstrap_block_length, args.bootstrap_reps, rng))
    pd.DataFrame(boot_rows).to_csv(out/"v223_paired_loss_bootstrap.csv", index=False)
    config = {"input_root":str(root),"v215_script":args.v215_script,"symbols":args.symbols,
              "start_date":args.start_date,"end_date":args.end_date,
              "initial_train_timestamps":n0,"n_usable_timestamps":len(timestamps),
              "n_oos_timestamps":len(timestamps)-n0,
              "models":order,"evaluation":"expanding-window chronological pooled OLS with asset fixed effects; asset-specific GARCH(1,1) fit on trailing contiguous 1m log returns",
              "naive_current_rv":"direct persistence forecast: next-day RV = current-day RV",
              "qlike":"RV forecast; fitted log-RV models use Duan smearing from each training window only",
              "garch_model":"zero-mean Gaussian-QMLE GARCH(1,1), returns scaled by 1000, 1440-step integrated variance forecast; fixed alpha=.05 beta=.94 fallback if numerical optimization fails",
              "garch_lookback_days":args.garch_lookback_days,"garch_min_obs":args.garch_min_obs,"garch_maxiter":args.garch_maxiter,
              "bootstrap":"paired moving-block bootstrap by timestamp; all assets within timestamp kept together",
              "bootstrap_reps":args.bootstrap_reps,"bootstrap_block_length":args.bootstrap_block_length,
              "seed":args.seed}
    (out/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nOOS SUMMARY")
    print(metrics.to_string(index=False))
    print("\nPAIRED LOSS DIFFERENTIALS (model_a minus model_b; negative favors model_a)")
    print(pd.DataFrame(boot_rows).to_string(index=False))
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
