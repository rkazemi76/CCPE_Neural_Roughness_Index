#!/usr/bin/env python3
"""v2.22: frozen neural-vs-classical benchmark on fresh stochastic-H synthetic paths.

Uses the ORIGINAL v2.7 stochastic-H simulator implementation in v2.13 memoryfix,
loads the frozen W384 checkpoint from seed_42, and evaluates untouched synthetic
paths for BSS and rBergomi-style generators across the noise levels used in v2.13.
Classical variogram and DFA estimates are reported raw and with an affine calibration
fitted only on a separate synthetic calibration sample (per generator/noise cell).
The context-mean oracle is included as a non-observable diagnostic, not a usable model.
--h_mode constant reconstructs constant-H paths with the same generator innovations,
providing a closer apples-to-apples target for window-global classical estimators.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.signal import fftconvolve


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _NumpyFastConvolveProxy:
    """Module-local NumPy proxy: use FFT convolution for long simulator convolutions."""
    def __init__(self, np_module):
        self._np = np_module
    def __getattr__(self, name):
        if name == "convolve":
            return lambda a, v, mode="full": fftconvolve(a, v, mode=mode)
        return getattr(self._np, name)


def install_observation_cache(sim):
    original = sim.observation_rv
    original_fine = sim.fine_returns
    cache = {}
    def cached_observation_rv(path, generator, cfg, M, noise):
        key = (id(path), generator, round(float(noise), 8))
        if key not in cache:
            r_true = original_fine(path, generator, cfg)
            if float(noise) != 0.0:
                eps = np.broadcast_to(path.obs_noise_driver[:, None], r_true.shape).astype(np.float32)
                obs_sd = float(noise) * cfg.return_scale / math.sqrt(float(max(cfg.resolutions)))
                r_obs = r_true + obs_sd * eps
            else:
                r_obs = r_true
            cache[key] = r_obs.astype(np.float32, copy=False)
        agg = sim.aggregate(cache[key], int(M))
        return np.maximum(np.sum(agg ** 2, axis=1), 1e-12).astype(np.float32)
    sim.observation_rv = cached_observation_rv
    def clear_path(path):
        pid = id(path)
        for k in [k for k in cache if k[0] == pid]:
            del cache[k]
    return original, clear_path


def generate_path(sim, generator, cfg, rng, h_mode="stochastic"):
    if generator == sim.GEN_BSS:
        path = sim.generate_bss_path(cfg, rng)
    elif generator == sim.GEN_RBERGOMI:
        path = sim.generate_rbergomi_path(cfg, rng)
    else:
        raise ValueError(generator)
    if h_mode == "stochastic":
        return path
    if h_mode == "constant":
        h0 = float(rng.uniform(cfg.h_min, cfg.h_max))
        h_new = np.full(cfg.seq_len, h0, dtype=np.float32)
        if generator == sim.GEN_BSS:
            return sim.make_bss_counterfactual(path, h_new, cfg)
        return sim.make_rb_counterfactual(path, h_new, cfg)
    raise ValueError(f"Unknown h_mode: {h_mode}")


def model_prediction(sim, model, mean, std, path, generator, cfg, noise, W, device):
    feat = sim.make_representation_context(path, generator, cfg, float(noise), int(W))
    feat = (np.asarray(feat, np.float32) - mean.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1e-8)
    with torch.no_grad():
        p = model(torch.from_numpy(feat[None, ...]).to(device)).cpu().numpy().reshape(-1)[0]
    return float(p)


def classical_estimates(sim_cls, sim, path, generator, cfg, noise, W):
    # Match the empirical estimator definition: estimate from finest (M=64) log-RV.
    finest = np.log(sim.observation_rv(path, generator, cfg, max(cfg.resolutions), float(noise))) [-int(W):]
    vh = sim_cls.variogram_h(finest, cfg.benchmark_scales)
    dh = sim_cls.dfa_h(finest, cfg.dfa_scales)
    return float(vh), float(dh)


def collect_sample(sim, sim_cls, model, mean, std, cfg, generator, noises,
                   n_samples, seed, W, device, label, clear_path, h_mode):
    rng = np.random.default_rng(int(seed))
    rows = []
    t0 = time.perf_counter()
    for i in range(int(n_samples)):
        path = generate_path(sim, generator, cfg, rng, h_mode)
        try:
            h_last = float(path.H_path[-1])
            h_context_mean = float(np.mean(path.H_path[-int(W):]))
            for noise in noises:
                hn = model_prediction(sim, model, mean, std, path, generator, cfg, noise, W, device)
                vh, dh = classical_estimates(sim_cls, sim, path, generator, cfg, noise, W)
                rows.append({
                    "sample_id": i, "sample": label, "generator": generator,
                    "noise": float(noise), "H_last": h_last,
                    "H_context_mean_oracle": h_context_mean,
                    "H_neural": hn, "H_variogram_raw": vh, "H_dfa_raw": dh,
                })
        finally:
            clear_path(path)
        if (i + 1) % max(1, int(n_samples) // 10) == 0 or i + 1 == n_samples:
            print(f"  {label} {generator}: {i+1}/{n_samples} paths ({time.perf_counter()-t0:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def affine_calibrate(cal_est, cal_y, test_est, lo, hi):
    x = np.asarray(cal_est, float); y = np.asarray(cal_y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or np.std(x[ok]) < 1e-10:
        return np.clip(np.full(len(test_est), np.nanmean(y[ok]) if ok.sum() else 0.275), lo, hi), (np.nan, np.nan)
    X = np.column_stack([np.ones(ok.sum()), x[ok]])
    beta = np.linalg.lstsq(X, y[ok], rcond=None)[0]
    pred = beta[0] + beta[1] * np.asarray(test_est, float)
    return np.clip(pred, lo, hi), (float(beta[0]), float(beta[1]))


def score(y, pred):
    y = np.asarray(y, float); p = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {"n": 0, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "Corr": np.nan, "Bias": np.nan}
    e = p - y
    sst = np.sum((y - y.mean()) ** 2)
    corr = np.corrcoef(y, p)[0, 1] if len(y) >= 2 and np.std(p) > 1e-12 and np.std(y) > 1e-12 else np.nan
    return {"n": int(len(y)), "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAE": float(np.mean(np.abs(e))), "R2": float(1 - np.sum(e**2) / sst) if sst > 0 else np.nan,
            "Corr": float(corr) if np.isfinite(corr) else np.nan, "Bias": float(np.mean(e))}


def paired_bootstrap(test_df, pred_a, pred_b, reps, seed):
    y = test_df["H_last"].to_numpy(float)
    a = test_df[pred_a].to_numpy(float); b = test_df[pred_b].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[ok], a[ok], b[ok]
    if len(y) < 10:
        return {"n": len(y), "delta_mse_a_minus_b": np.nan, "mse_ci95_lower": np.nan, "mse_ci95_upper": np.nan,
                "delta_mae_a_minus_b": np.nan, "mae_ci95_lower": np.nan, "mae_ci95_upper": np.nan}
    da = (a-y)**2 - (b-y)**2
    db = np.abs(a-y) - np.abs(b-y)
    rng = np.random.default_rng(seed)
    boot_m = np.empty(int(reps)); boot_a = np.empty(int(reps))
    for k in range(int(reps)):
        ix = rng.integers(0, len(y), size=len(y))
        boot_m[k] = np.mean(da[ix]); boot_a[k] = np.mean(db[ix])
    mlo, mhi = np.quantile(boot_m, [0.025, 0.975]); alo, ahi = np.quantile(boot_a, [0.025, 0.975])
    return {"n": int(len(y)), "delta_mse_a_minus_b": float(np.mean(da)),
            "mse_ci95_lower": float(mlo), "mse_ci95_upper": float(mhi),
            "delta_mae_a_minus_b": float(np.mean(db)), "mae_ci95_lower": float(alo), "mae_ci95_upper": float(ahi)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulator_script", default="code/HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix.py")
    ap.add_argument("--v215_script", default="code/HurstNeuralEstimator_v2_15_empirical_crypto_multiasset.py")
    ap.add_argument("--checkpoint_dir", default="seed_42")
    ap.add_argument("--context_length", type=int, default=384)
    ap.add_argument("--h_mode", choices=["stochastic", "constant"], default="stochastic",
                    help="stochastic: original blockwise-H target; constant: fresh paths with one fixed H per path")
    ap.add_argument("--n_calibration", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=1500)
    ap.add_argument("--noise_levels", nargs="+", type=float, default=[0.0, 0.0375, 0.0875, 0.1])
    ap.add_argument("--calibration_seed", type=int, default=314159)
    ap.add_argument("--test_seed", type=int, default=20261023)
    ap.add_argument("--bootstrap_reps", type=int, default=2000)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out_dir", default="results_hurst_v2_22_synthetic_classical_benchmark")
    args = ap.parse_args()

    sim = load_module(Path(args.simulator_script), "v213_sim_for_v222")
    sim_cls = load_module(Path(args.v215_script), "v215_classical_for_v222")
    # Use FFT convolution only for the simulation module; mathematically equivalent to np.convolve.
    sim.np = _NumpyFastConvolveProxy(np)
    W = int(args.context_length)
    if W != 384:
        raise ValueError("This frozen checkpoint is the W384 model; --context_length must be 384.")
    cfg = sim.Config(seq_len=768, h_block=32, h_phi=0.97, h_latent_sd=2.0,
                     bss_kernel_lambda=0.08, bss_kernel_size=512, bss_burn_in=512,
                     bss_xi=0.04, bss_eta=1.0, rb_xi0=0.04, rb_eta=1.0, rb_rho=-0.70,
                     T=3.0, resolutions=(8,16,32,64), context_lengths=(384,),
                     train_noise_levels=(0.0,0.025,0.05,0.075,0.1),
                     val_noise_levels=(0.0,0.025,0.05,0.075,0.1),
                     test_noise_levels=tuple(args.noise_levels))
    cfg.benchmark_scales = (1, 2, 4, 8, 16, 32, 64)
    cfg.dfa_scales = (8, 16, 32, 64)
    if cfg.seq_len % cfg.h_block or W % cfg.h_block:
        raise ValueError("seq_len and context length must align to H-blocks.")

    ckpt = Path(args.checkpoint_dir)
    model_path = ckpt / f"model_state_W{W}.pt"
    scaler_path = ckpt / f"standardizer_W{W}.npz"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Need {model_path} and {scaler_path}; extract seed_42.zip first.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = sim.HurstLSTM(input_size=27, cfg=cfg).to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state); model.eval()
    ss = np.load(scaler_path)
    mean = np.asarray(ss["mean"], np.float32).reshape(-1)
    std = np.asarray(ss["std"], np.float32).reshape(-1)
    if len(mean) != 27 or len(std) != 27:
        raise ValueError(f"Checkpoint standardizer must have 27 features, got {len(mean)} and {len(std)}")

    original_observation, clear_path = install_observation_cache(sim)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    all_cal=[]; all_test=[]; coefficients=[]
    generators = [sim.GEN_BSS, sim.GEN_RBERGOMI]
    for gi, gen in enumerate(generators):
        cal = collect_sample(sim, sim_cls, model, mean, std, cfg, gen, args.noise_levels,
                             args.n_calibration, args.calibration_seed + gi * 100003, W, device, "calibration", clear_path, args.h_mode)
        test = collect_sample(sim, sim_cls, model, mean, std, cfg, gen, args.noise_levels,
                              args.n_test, args.test_seed + gi * 100003, W, device, "test", clear_path, args.h_mode)
        all_cal.append(cal); all_test.append(test)
        for noise in args.noise_levels:
            c = cal[np.isclose(cal.noise, noise)].copy()
            t = test[np.isclose(test.noise, noise)].copy()
            for raw, calibrated in [("H_variogram_raw", "H_variogram_calibrated"), ("H_dfa_raw", "H_dfa_calibrated")]:
                pred, (intercept, slope) = affine_calibrate(c[raw], c.H_last, t[raw], cfg.h_min, cfg.h_max)
                t.loc[:, calibrated] = pred
                coefficients.append({"generator":gen,"noise":float(noise),"estimator":raw,
                                     "intercept":intercept,"slope":slope,"n_calibration":len(c)})
            # Add diagnostic oracle, not a feasible estimate.
            all_test[-1].loc[all_test[-1].index.isin(t.index), "H_variogram_calibrated"] = t["H_variogram_calibrated"]
            all_test[-1].loc[all_test[-1].index.isin(t.index), "H_dfa_calibrated"] = t["H_dfa_calibrated"]
    caldf = pd.concat(all_cal, ignore_index=True)
    testdf = pd.concat(all_test, ignore_index=True)
    # Fill calibrated columns in the combined test data using the test rows' stable sample_id/generator/noise keys.
    testdf = testdf.drop(columns=[c for c in ["H_variogram_calibrated","H_dfa_calibrated"] if c in testdf.columns], errors="ignore")
    # Refit calibration using the saved independent calibration panel and apply to matching test cells.
    for row in coefficients:
        gen, noise, est = row["generator"], row["noise"], row["estimator"]
        base = est.replace("_raw", "_calibrated")
        maskc = (caldf.generator == gen) & np.isclose(caldf.noise, noise)
        maskt = (testdf.generator == gen) & np.isclose(testdf.noise, noise)
        pred, _ = affine_calibrate(caldf.loc[maskc, est], caldf.loc[maskc, "H_last"], testdf.loc[maskt, est], cfg.h_min, cfg.h_max)
        testdf.loc[maskt, base] = pred

    testdf.to_csv(out / "v222_test_predictions.csv", index=False)
    caldf.to_csv(out / "v222_calibration_predictions.csv", index=False)
    pd.DataFrame(coefficients).to_csv(out / "v222_calibration_coefficients.csv", index=False)

    metric_rows=[]
    model_cols = ["H_neural","H_variogram_raw","H_variogram_calibrated","H_dfa_raw","H_dfa_calibrated","H_context_mean_oracle"]
    for (gen, noise), g in testdf.groupby(["generator","noise"], sort=False):
        for col in model_cols:
            s = score(g.H_last, g[col])
            metric_rows.append({"generator":gen,"noise":float(noise),"model":col,**s})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "v222_synthetic_metrics.csv", index=False)

    paired=[]
    rng_offset=0
    for (gen, noise), g in testdf.groupby(["generator","noise"], sort=False):
        for comparator in ["H_variogram_raw","H_variogram_calibrated","H_dfa_raw","H_dfa_calibrated"]:
            res = paired_bootstrap(g, "H_neural", comparator, args.bootstrap_reps,
                                   args.test_seed + 41 + rng_offset)
            paired.append({"generator":gen,"noise":float(noise),"model_a":"H_neural",
                           "model_b":comparator,**res,"bootstrap_reps":int(args.bootstrap_reps)})
            rng_offset += 1
    pd.DataFrame(paired).to_csv(out / "v222_paired_bootstrap.csv", index=False)
    config = {"basis":"ORIGINAL_v2.7 stochastic-H simulator via v2.13 memoryfix; v2.7-fix1 not used",
              "h_mode":args.h_mode,
              "checkpoint":str(model_path),"standardizer":str(scaler_path),"context_length":W,
              "sequence_length":cfg.seq_len,"h_block":cfg.h_block,"h_phi":cfg.h_phi,"h_latent_sd":cfg.h_latent_sd,
              "generators":generators,"noise_levels":[float(x) for x in args.noise_levels],
              "n_calibration_per_generator":int(args.n_calibration),"n_test_per_generator":int(args.n_test),
              "calibration":"separate synthetic sample; affine mapping H_last ~ classical_estimate fit per generator and noise; predictions clipped to known support [0.05, 0.50]",
              "target":"H_last; context-mean oracle uses true mean H over final W=384 observations and is not observable",
              "classical_series":"finest-resolution M=64 log realized variance over same final 384 observations",
              "bootstrap":"paired iid test-path bootstrap; calibration sample uncertainty not included",
              "bootstrap_reps":int(args.bootstrap_reps),"test_seed":int(args.test_seed),"calibration_seed":int(args.calibration_seed),
              "device":str(device),"fast_convolution":"scipy FFT convolution replacing numpy full convolution in simulator; same linear convolution up to floating-point tolerance"}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nSYNTHETIC TEST SUMMARY")
    print(metrics.to_string(index=False))
    print("\nPAIRED BOOTSTRAP (negative loss differential favors Neural H)")
    print(pd.DataFrame(paired).to_string(index=False))
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
