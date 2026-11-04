#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Evaluate v2.13 stochastic-H checkpoints on a COMMON held-out test set.

This script does not retrain models. It generates one fixed stochastic-H
BSS test set and one fixed stochastic-H rBergomi test set, then evaluates
seed 42/123/2024 checkpoints on exactly the same paths at each observation
noise level. This separates model-seed differences from test-path differences.

Required per seed files:
  checkpoints/seed_<SEED>/model_state_W384.pt
  checkpoints/seed_<SEED>/standardizer_W384.npz

Run from project root, e.g.:
  python .\code\evaluate_variable_H_three_seed_common_test.py `
      --base_script .\code\HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix.py `
      --checkpoint_root .\checkpoints `
      --out_dir .\results\v2_26_variable_H_three_seed_common_test
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

CONFIG_KEYS = [
    "seq_len", "h_min", "h_max", "h_block", "h_phi", "h_latent_sd",
    "bss_kernel_lambda", "bss_kernel_size", "bss_burn_in", "bss_xi", "bss_eta",
    "rb_xi0", "rb_eta", "rb_rho", "T", "return_scale", "resolutions",
    "test_noise_levels", "temporal_lag", "roughness_lag", "smooth_window",
    "hidden_size", "num_layers", "dropout",
]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("v213_source", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source script: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def as_tuple_if_needed(v):
    if isinstance(v, list):
        return tuple(v)
    return v


def load_models(mod, checkpoint_root: Path, seeds, device):
    models = {}
    cfgs = {}
    for seed in seeds:
        sd = checkpoint_root / f"seed_{seed}"
        model_path = sd / "model_state_W384.pt"
        std_path = sd / "standardizer_W384.npz"
        if not model_path.exists() or not std_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint/standardizer for seed {seed}. Expected:\n"
                f"  {model_path}\n  {std_path}"
            )
        ckpt = safe_torch_load(model_path, device)
        saved = dict(ckpt.get("config", {}))
        allowed = {x.name for x in __import__("dataclasses").fields(mod.Config)}
        saved = {k: as_tuple_if_needed(v) for k, v in saved.items() if k in allowed}
        cfg = mod.Config(**saved)
        cfg.context_lengths = (384,)
        cfg.seeds = tuple(seeds)
        # The first checkpoint defines the common data-generation/feature setup.
        std = np.load(std_path)
        mean = np.asarray(std["mean"], dtype=np.float32)
        scale = np.asarray(std["std"], dtype=np.float32)
        if mean.shape != scale.shape:
            raise ValueError(f"Mean/std shape mismatch for seed {seed}: {mean.shape} vs {scale.shape}")
        model = mod.HurstLSTM(int(mean.shape[-1]), cfg).to(device)
        state = ckpt.get("state_dict")
        if state is None:
            raise ValueError(f"Checkpoint {model_path} does not have a state_dict key")
        model.load_state_dict(state, strict=True)
        model.eval()
        models[seed] = {"model": model, "mean": mean, "std": scale, "cfg": cfg}
        cfgs[seed] = cfg
        print(f"Loaded seed {seed}: {model_path.name}; feature count={mean.shape[-1]}")

    base = cfgs[seeds[0]]
    for seed in seeds[1:]:
        cfg = cfgs[seed]
        mismatches = []
        for key in CONFIG_KEYS:
            va = as_tuple_if_needed(getattr(base, key))
            vb = as_tuple_if_needed(getattr(cfg, key))
            if va != vb:
                mismatches.append(f"{key}: seed {seeds[0]}={va!r}, seed {seed}={vb!r}")
        if mismatches:
            raise ValueError(
                "Checkpoint configurations are not comparable. Align/retrain before evaluation:\n"
                + "\n".join(mismatches)
            )
    return models, base


def generate_common_paths(mod, cfg, generator, n_test, test_seed):
    paths = []
    start = 300000 if generator == mod.GEN_BSS else 600000
    for i in range(n_test):
        rng = np.random.default_rng(test_seed + start + i)
        if generator == mod.GEN_BSS:
            paths.append(mod.generate_bss_path(cfg, rng))
        else:
            paths.append(mod.generate_rbergomi_path(cfg, rng))
        if (i + 1) % max(1, n_test // 10) == 0 or i + 1 == n_test:
            print(f"  generated {generator} common paths: {i+1}/{n_test}")
    return paths


def metric_row(mod, seed, generator, noise, y, pred, context_length):
    met = mod.regression_metrics(y, pred)
    return {
        "seed": int(seed), "generator": generator, "noise": float(noise),
        "context_length": int(context_length), "n_test": int(len(y)),
        **met,
    }


def block_bootstrap_ensemble_vs_seed(df_pred, metric, reps, rng):
    """Path-i.i.d. paired bootstrap over independent synthetic test paths. metric is MSE or MAE."""
    rows = []
    for (generator, noise), grp in df_pred.groupby(["generator", "noise"], sort=True):
        wide = grp.pivot(index="path_id", columns="seed", values="prediction").sort_index()
        y_w = grp.drop_duplicates("path_id").set_index("path_id")["H_last"].reindex(wide.index).to_numpy(float)
        p_ens = wide.mean(axis=1).to_numpy(float)
        err_ens = np.abs(p_ens - y_w) if metric == "MAE" else (p_ens - y_w) ** 2
        for seed in wide.columns:
            p_one = wide[seed].to_numpy(float)
            err_one = np.abs(p_one - y_w) if metric == "MAE" else (p_one - y_w) ** 2
            # Negative delta means ensemble loss is lower.
            delta = err_ens - err_one
            boot = np.empty(reps, dtype=float)
            n = len(delta)
            for b in range(reps):
                idx = rng.integers(0, n, size=n)
                boot[b] = delta[idx].mean()
            rows.append({
                "generator": generator, "noise": float(noise), "metric": metric,
                "ensemble_vs_seed": int(seed), "delta_loss_ensemble_minus_seed": float(delta.mean()),
                "ci95_lower": float(np.quantile(boot, .025)),
                "ci95_upper": float(np.quantile(boot, .975)), "bootstrap_reps": int(reps),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_script", required=True, help="Path to HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix.py")
    ap.add_argument("--checkpoint_root", required=True, help="Folder containing seed_42, seed_123, seed_2024")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--n_test", type=int, default=1500)
    ap.add_argument("--test_seed", type=int, default=230261026)
    ap.add_argument("--bootstrap_reps", type=int, default=2000)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    src = Path(args.base_script).resolve()
    ckroot = Path(args.checkpoint_root).resolve()
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    if not src.exists(): raise FileNotFoundError(src)

    mod = load_module(src)
    device = mod.get_device(args.device)
    print(f"Device: {device}; n_test={args.n_test}; fixed test_seed={args.test_seed}")
    models, cfg = load_models(mod, ckroot, args.seeds, device)
    cfg.n_test = args.n_test
    cfg.context_lengths = (384,)

    # Record the configuration governing the shared test data and inference.
    config_json = {
        "source_script": src.name, "checkpoint_root": str(ckroot),
        "seeds": args.seeds, "n_test": args.n_test, "test_seed": args.test_seed,
        "context_length": 384, "device": str(device),
        "shared_test_paths_across_models": True,
        "note": "All checkpoints are evaluated on identical generated paths within each generator/noise setting.",
    }
    (out / "common_test_config.json").write_text(json.dumps(config_json, indent=2), encoding="utf-8")

    metric_rows = []
    prediction_chunks = []
    all_noise = tuple(cfg.test_noise_levels)
    for generator in (mod.GEN_BSS, mod.GEN_RBERGOMI):
        print(f"\nGenerating one common {generator} test set...")
        t0 = time.perf_counter()
        paths = generate_common_paths(mod, cfg, generator, args.n_test, args.test_seed)
        print(f"Generated {generator} test set in {time.perf_counter()-t0:.1f}s")
        for noise in all_noise:
            print(f"  Evaluating generator={generator}, noise={noise:g}")
            ref_y = None
            for seed in args.seeds:
                item = models[seed]
                y, hm, pred, met = mod.context_test_predictions(
                    paths, generator, float(noise), cfg, item["model"], item["mean"], item["std"],
                    device, 384
                )
                if ref_y is None:
                    ref_y = y.copy()
                elif not np.allclose(ref_y, y):
                    raise RuntimeError("Target H_last differed across models; common-path matching failed")
                metric_rows.append({
                    **metric_row(mod, seed, generator, noise, y, pred, 384),
                    "oracle_Hmean_R2": met["oracle_Hmean_R2"],
                    "gain_vs_Hmean_oracle": met["gain_vs_Hmean_oracle"],
                    "localDev_R2": met["localDev_R2"],
                    "partialCorr_Hlast_given_Hmean": met["partialCorr_Hlast_given_Hmean"],
                })
                prediction_chunks.append(pd.DataFrame({
                    "path_id": np.arange(len(y), dtype=int), "seed": int(seed),
                    "generator": generator, "noise": float(noise),
                    "H_last": y.astype(float), "H_mean": hm.astype(float), "prediction": pred.astype(float),
                }))
                print(f"    seed={seed}: R2={met['R2']:.4f}, RMSE={met['RMSE']:.5f}, Corr={met['Corr']:.4f}")
        del paths

    predictions = pd.concat(prediction_chunks, ignore_index=True)
    predictions.to_csv(out / "variable_H_common_test_predictions.csv", index=False)
    per_seed = pd.DataFrame(metric_rows)
    per_seed.to_csv(out / "variable_H_common_test_metrics_by_seed.csv", index=False)

    # Ensemble metrics: average each seed's prediction for the very same path.
    ensemble = predictions.groupby(["path_id", "generator", "noise"], as_index=False).agg(
        H_last=("H_last", "first"), H_mean=("H_mean", "first"), prediction=("prediction", "mean")
    )
    ens_rows = []
    for (gen, noise), g in ensemble.groupby(["generator", "noise"], sort=True):
        met = mod.regression_metrics(g["H_last"].to_numpy(), g["prediction"].to_numpy())
        ens_rows.append({"generator": gen, "noise": float(noise), "context_length": 384,
                         "n_test": len(g), **met,
                         "oracle_Hmean_R2": mod.regression_metrics(g["H_last"].to_numpy(), g["H_mean"].to_numpy())["R2"]})
    ensemble_metrics = pd.DataFrame(ens_rows)
    ensemble_metrics.to_csv(out / "variable_H_common_test_ensemble_metrics.csv", index=False)

    rng = np.random.default_rng(args.test_seed + 888)
    boot_rows = []
    boot_rows += block_bootstrap_ensemble_vs_seed(predictions, "MSE", args.bootstrap_reps, rng)
    boot_rows += block_bootstrap_ensemble_vs_seed(predictions, "MAE", args.bootstrap_reps, rng)
    pd.DataFrame(boot_rows).to_csv(out / "variable_H_common_test_ensemble_paired_bootstrap.csv", index=False)

    # Summary across independently trained checkpoints on identical paths.
    across_seed = per_seed.groupby(["generator", "noise"], as_index=False).agg(
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        Corr_mean=("Corr", "mean"), Corr_sd=("Corr", "std"),
        localDevR2_mean=("localDev_R2", "mean"),
        partialCorr_mean=("partialCorr_Hlast_given_Hmean", "mean"),
        gain_vs_Hmean_mean=("gain_vs_Hmean_oracle", "mean"),
    )
    across_seed.to_csv(out / "variable_H_common_test_across_seed_summary.csv", index=False)
    print("\nCompleted common-test variable-H evaluation.")
    print("Outputs:", out)
    print("Primary files: variable_H_common_test_across_seed_summary.csv, variable_H_common_test_ensemble_metrics.csv, variable_H_common_test_ensemble_paired_bootstrap.csv")

if __name__ == "__main__":
    main()
