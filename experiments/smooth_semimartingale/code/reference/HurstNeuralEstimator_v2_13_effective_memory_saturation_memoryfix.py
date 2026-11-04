#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix

Scientific basis
----------------
Built from the ORIGINAL v2.7 stochastic-H implementation used by the user,
NOT v2.7-fix1. Preserves the v2.6/v2.8 pooled BSS + rough Bergomi training
recipe, multiscale_rv_robust representation, and frozen v2.1-fix2 BiLSTM.

Main purpose
------------
Identify the effective temporal memory of the stochastic-H estimator by
independently training a model for each retained tail context. The main
comparison is deliberately extended beyond the previous v2.12 maximum of
256 observations to 256, 384, 512, and 768 observations (8, 12, 16, and 24
stochastic-H blocks). This is a saturation test: if performance plateaus, the
plateau provides an empirical estimate of the effective memory scale; if it
continues to rise at 768, the experiment only establishes a lower bound.

Each context-specific representation is recomputed entirely inside the retained
tail, so centering, differences, moving averages, and normalized differences do
not import information from observations older than the declared context.

Core counterfactual design
--------------------------
For each held-out test path:
  1) keep the SAME Brownian/innovation realizations;
  2) for each distance d, identify the donor H block d positions before the endpoint;
  3) choose one common, path-specific perturbation amplitude A_i that is feasible
     for ALL tested distances;
  4) move the final H block toward the donor by exactly A_i;
  5) move the donor block in the opposite direction by exactly A_i, preserving
     H_mean exactly;
  6) use the SAME A_i for every distance on that path;
  7) regenerate the observed path using the same innovations and compare predictions.

This is a mean-preserving matched-amplitude perturbation, not a literal swap of
two H values. The absolute change |Delta H_last| is exactly matched across
distances within each path, so differences in prediction sensitivity cannot be
explained by different perturbation magnitudes. A mean-only predictor has zero
within-pair prediction change by construction.

Important interpretation
------------------------
This is a controlled counterfactual diagnostic, not proof of causality. The
counterfactual changes the temporal arrangement of H blocks and therefore can
change the generated observed path. The scientific question is whether those
changes in the observed path are reflected in predictions in the direction of
H_last.

Memory
------
Training/validation augmented features use disk-backed float16 memmaps and are
streamed batch-by-batch. Test/counterfactual features are built in small RAM
batches. Cache files are removed after each seed unless --keep_cache is used.

Suggested main run
------------------
python HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix.py ^
  --n_train 5000 --n_val 1000 --n_test 1500 --seq_len 256 ^
  --epochs 40 --resolutions 8 16 32 64 ^
  --train_noise_levels 0 0.025 0.05 0.075 0.10 ^
  --test_noise_levels 0 0.0375 0.0875 0.10 ^
  --seeds 42 123 2024 --h_block 32 --h_phi 0.97 --h_latent_sd 2.0 ^
  --recent_windows 64 128 ^
  --out_dir results_hurst_v2_9_matched_mean_local_identification
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn


GEN_BSS = "bss"
GEN_RBERGOMI = "rbergomi"
REP_ROBUST = "multiscale_rv_robust"


@dataclass
class Config:
    n_train: int = 5000
    n_val: int = 1000
    n_test: int = 1500
    seq_len: int = 768

    h_min: float = 0.05
    h_max: float = 0.50
    h_block: int = 32
    h_phi: float = 0.97
    h_latent_sd: float = 2.0

    bss_kernel_lambda: float = 0.08
    bss_kernel_size: int = 512
    bss_burn_in: int = 512
    bss_xi: float = 0.04
    bss_eta: float = 1.0

    rb_xi0: float = 0.04
    rb_eta: float = 1.0
    rb_rho: float = -0.70
    T: float = 3.0

    return_scale: float = 1.0
    resolutions: Tuple[int, ...] = (8, 16, 32, 64)

    train_noise_levels: Tuple[float, ...] = (0.0, 0.025, 0.05, 0.075, 0.10)
    val_noise_levels: Tuple[float, ...] = (0.0, 0.025, 0.05, 0.075, 0.10)
    test_noise_levels: Tuple[float, ...] = (0.0, 0.0375, 0.0875, 0.10)

    temporal_lag: int = 1
    roughness_lag: int = 4
    smooth_window: int = 4

    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.15
    batch_size: int = 16
    validation_batch_size: int = 16
    cache_build_batch: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-5
    epochs: int = 40
    patience: int = 8
    grad_clip: float = 1.0

    bootstrap: int = 500
    recent_windows: Tuple[int, ...] = (64, 128)
    context_lengths: Tuple[int, ...] = (256, 384, 512, 768)
    local_swap_distances: Tuple[int, ...] = (1, 2, 4)
    matched_sample_count: int = 1000
    matched_amplitude_fraction: float = 0.75

    cache_dtype: str = "float16"
    keep_cache: bool = False
    seeds: Tuple[int, ...] = (42, 123, 2024)
    out_dir: str = "results_hurst_v2_13_effective_memory_saturation"
    device: str = "auto"


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    corr = np.nan
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "Bias": float(np.mean(err)),
        "Corr": corr,
    }


def bootstrap_metrics(y_true, y_pred, reps, seed):
    if reps <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rows = []
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        rows.append(regression_metrics(y_true[idx], y_pred[idx]))
    b = pd.DataFrame(rows)
    p = regression_metrics(y_true, y_pred)
    out = {}
    for m in ("RMSE", "MAE", "R2", "Corr"):
        vals = b[m].dropna().to_numpy()
        out[f"{m}_estimate"] = p[m]
        out[f"{m}_ci_low"] = float(np.percentile(vals, 2.5)) if len(vals) else np.nan
        out[f"{m}_ci_high"] = float(np.percentile(vals, 97.5)) if len(vals) else np.nan
    return out


def partial_corr(y, p, z):
    y = np.asarray(y, float); p = np.asarray(p, float); z = np.asarray(z, float)
    A = np.column_stack([np.ones(len(z)), z])
    by = np.linalg.lstsq(A, y, rcond=None)[0]
    bp = np.linalg.lstsq(A, p, rcond=None)[0]
    ry = y - A @ by
    rp = p - A @ bp
    if np.std(ry) == 0 or np.std(rp) == 0:
        return np.nan
    return float(np.corrcoef(ry, rp)[0, 1])


def local_residual_r2(y_last, pred, h_mean):
    y = np.asarray(y_last, float) - np.asarray(h_mean, float)
    p = np.asarray(pred, float) - np.asarray(h_mean, float)
    ssr = float(np.sum((y - p) ** 2))
    sst = float(np.sum(y ** 2))
    return float(1.0 - ssr / sst) if sst > 0 else np.nan


def mask_temporal_region(X, keep_window=None, mode="recent"):
    Y = np.array(X, dtype=np.float32, copy=True)
    if keep_window is None or keep_window >= Y.shape[1]:
        return Y
    w = int(max(1, keep_window))
    if mode == "recent":
        Y[:, :Y.shape[1] - w, :] = 0.0
    elif mode == "old":
        Y[:, Y.shape[1] - w:, :] = 0.0
    else:
        raise ValueError("mode must be recent or old")
    return Y


# -----------------------------------------------------------------------------
# Stochastic H_t process
# -----------------------------------------------------------------------------

def generate_stochastic_H_path(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    if cfg.seq_len % cfg.h_block != 0:
        raise ValueError("seq_len must be divisible by h_block")
    B = cfg.seq_len // cfg.h_block
    z = np.zeros(B, dtype=np.float64)
    z[0] = rng.normal(0.0, cfg.h_latent_sd)
    innov_sd = cfg.h_latent_sd * math.sqrt(max(1.0 - cfg.h_phi ** 2, 1e-12))
    for j in range(1, B):
        z[j] = cfg.h_phi * z[j - 1] + innov_sd * rng.standard_normal()
    hb = cfg.h_min + (cfg.h_max - cfg.h_min) / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return np.repeat(hb, cfg.h_block).astype(np.float32)


def block_values(H: np.ndarray, cfg: Config) -> np.ndarray:
    return np.asarray(H).reshape(-1, cfg.h_block)[:, 0].astype(np.float32)


def swap_last_block(H: np.ndarray, cfg: Config, distance: int) -> np.ndarray:
    """Swap the final H block with a block `distance` positions earlier."""
    hb = block_values(H, cfg).copy()
    B = len(hb)
    distance = int(distance)
    if distance < 1 or distance >= B:
        raise ValueError(f"distance must be in [1, {B-1}] for {B} blocks")
    j = B - 1 - distance
    hb[-1], hb[j] = hb[j], hb[-1]
    return np.repeat(hb, cfg.h_block).astype(np.float32)


# -----------------------------------------------------------------------------
# BSS generator with reusable innovations
# -----------------------------------------------------------------------------

@dataclass
class BSSPath:
    H_path: np.ndarray
    latent_logvol: np.ndarray
    latent_variance: np.ndarray
    fine_eps: np.ndarray
    obs_noise_driver: np.ndarray
    innovation_z: np.ndarray


def bss_kernel(H, K, lam):
    k = np.arange(1, K + 1, dtype=np.float64)
    g = k ** (H - 0.5) * np.exp(-lam * k)
    g /= max(float(np.sqrt(np.sum(g * g))), 1e-15)
    return g


def bss_factor_from_H(H_path, cfg: Config, z: np.ndarray) -> np.ndarray:
    n = len(H_path); K = cfg.bss_kernel_size; burn = cfg.bss_burn_in; block = cfg.h_block
    total = n + burn + K + 8
    x = np.zeros(n, dtype=np.float64)
    for b0 in range(0, n, block):
        b1 = min(n, b0 + block)
        h_eff = float(np.mean(H_path[b0:b1]))
        g = bss_kernel(h_eff, K, cfg.bss_kernel_lambda)
        full = np.convolve(z, g, mode="full")[:total]
        start = burn + K // 2 + b0
        x[b0:b1] = full[start:start + (b1 - b0)]
    x -= np.mean(x)
    x /= max(float(np.std(x)), 1e-8)
    return x.astype(np.float32)


def generate_bss_path(cfg, rng):
    H = generate_stochastic_H_path(cfg, rng)
    z = rng.standard_normal(cfg.seq_len + cfg.bss_burn_in + cfg.bss_kernel_size + 8).astype(np.float64)
    x = bss_factor_from_H(H, cfg, z)
    logv = np.log(cfg.bss_xi) + cfg.bss_eta * x - 0.5 * cfg.bss_eta ** 2
    logv = np.clip(logv, -15, 8).astype(np.float32)
    v = np.exp(logv).astype(np.float32)
    M = max(cfg.resolutions)
    fine_eps = rng.standard_normal((cfg.seq_len, M)).astype(np.float32)
    obs = rng.standard_normal(cfg.seq_len).astype(np.float32)
    return BSSPath(H, logv, v, fine_eps, obs, z)


def make_bss_counterfactual(path: BSSPath, H_new: np.ndarray, cfg: Config) -> BSSPath:
    x = bss_factor_from_H(H_new, cfg, path.innovation_z)
    logv = np.log(cfg.bss_xi) + cfg.bss_eta * x - 0.5 * cfg.bss_eta ** 2
    logv = np.clip(logv, -15, 8).astype(np.float32)
    v = np.exp(logv).astype(np.float32)
    return BSSPath(H_new.copy(), logv, v, path.fine_eps.copy(), path.obs_noise_driver.copy(), path.innovation_z.copy())


# -----------------------------------------------------------------------------
# rBergomi-style generator with reusable innovations
# -----------------------------------------------------------------------------

@dataclass
class RBergomiPath:
    H_path: np.ndarray
    time: np.ndarray
    W_H: np.ndarray
    variance: np.ndarray
    fine_spot_dB: np.ndarray
    obs_noise_driver: np.ndarray
    log_variance: np.ndarray
    dW_fine: np.ndarray
    dWperp: np.ndarray


def rb_factor_from_H(H_path, cfg: Config, dW_fine: np.ndarray):
    n = cfg.seq_len; M = max(cfg.resolutions)
    dtp = cfg.T / float(n)
    dW_period = np.sum(dW_fine, axis=1).astype(np.float64)
    WH = np.zeros(n, dtype=np.float64)
    var_disc = np.zeros(n, dtype=np.float64)
    k = np.arange(1, n + 1, dtype=np.float64)
    for b0 in range(0, n, cfg.h_block):
        b1 = min(n, b0 + cfg.h_block)
        h_eff = float(np.mean(H_path[b0:b1]))
        g = np.zeros(n + 1, dtype=np.float64)
        g[1:] = np.sqrt(2.0 * h_eff) * (k * dtp) ** (h_eff - 0.5)
        conv = np.convolve(dW_period, g, mode="full")[:n]
        j = np.arange(n, dtype=np.int64)
        t = j * dtp
        pos = t > 0
        vv = np.zeros(n, dtype=np.float64)
        if np.any(pos):
            jj = j[pos]
            weights = (k * dtp) ** (2.0 * h_eff - 1.0)
            csum = np.cumsum(weights)
            vv[pos] = 2.0 * h_eff * dtp * csum[np.clip(jj - 1, 0, n - 2)]
        scale = np.zeros(n, dtype=np.float64)
        scale[pos] = np.sqrt(np.maximum(vv[pos], 1e-14))
        local = np.zeros(n, dtype=np.float64)
        local[pos] = conv[pos] / scale[pos] * (t[pos] ** h_eff)
        WH[b0:b1] = local[b0:b1]
        var_disc[b0:b1] = vv[b0:b1]
    return np.arange(n, dtype=np.float32) * np.float32(dtp), WH.astype(np.float32), var_disc.astype(np.float32)


def generate_rbergomi_path(cfg, rng):
    H = generate_stochastic_H_path(cfg, rng)
    n = cfg.seq_len; M = max(cfg.resolutions)
    dtf = cfg.T / float(n * M)
    dW = rng.standard_normal((n, M)).astype(np.float32) * np.float32(math.sqrt(dtf))
    t, WH, var_disc = rb_factor_from_H(H, cfg, dW)
    logv = np.log(cfg.rb_xi0) + cfg.rb_eta * WH.astype(np.float64) - 0.5 * cfg.rb_eta ** 2 * np.maximum(t.astype(np.float64), 0) ** (2.0 * H.astype(np.float64))
    logv[0] = np.log(cfg.rb_xi0)
    logv = np.clip(logv, -15, 8).astype(np.float32)
    v = np.exp(logv).astype(np.float32)
    dWperp = rng.standard_normal((n, M)).astype(np.float32) * np.float32(math.sqrt(dtf))
    dB = np.float32(cfg.rb_rho) * dW + np.float32(math.sqrt(max(1 - cfg.rb_rho ** 2, 0))) * dWperp
    obs = rng.standard_normal(n).astype(np.float32)
    return RBergomiPath(H, t, WH, v, dB.astype(np.float32), obs, logv, dW, dWperp)


def make_rb_counterfactual(path: RBergomiPath, H_new: np.ndarray, cfg: Config) -> RBergomiPath:
    t, WH, var_disc = rb_factor_from_H(H_new, cfg, path.dW_fine)
    logv = np.log(cfg.rb_xi0) + cfg.rb_eta * WH.astype(np.float64) - 0.5 * cfg.rb_eta ** 2 * np.maximum(t.astype(np.float64), 0) ** (2.0 * H_new.astype(np.float64))
    logv[0] = np.log(cfg.rb_xi0)
    logv = np.clip(logv, -15, 8).astype(np.float32)
    v = np.exp(logv).astype(np.float32)
    dB = np.float32(cfg.rb_rho) * path.dW_fine + np.float32(math.sqrt(max(1 - cfg.rb_rho ** 2, 0))) * path.dWperp
    return RBergomiPath(H_new.copy(), t, WH, v, dB.astype(np.float32), path.obs_noise_driver.copy(), logv, path.dW_fine.copy(), path.dWperp.copy())


# -----------------------------------------------------------------------------
# Observation and representation
# -----------------------------------------------------------------------------

def fine_returns(path, generator, cfg):
    if generator == GEN_BSS:
        return cfg.return_scale * np.sqrt(np.maximum(path.latent_variance, 1e-12))[:, None] / math.sqrt(max(cfg.resolutions)) * path.fine_eps
    return cfg.return_scale * np.sqrt(np.maximum(path.variance, 1e-12))[:, None] * path.fine_spot_dB


def aggregate(r, M):
    n, Mmax = r.shape
    if Mmax % M:
        raise ValueError("resolution must divide max resolution")
    block = Mmax // M
    return r.reshape(n, M, block).sum(axis=2)


def observation_rv(path, generator, cfg, M, noise):
    r_true = fine_returns(path, generator, cfg)
    driver = path.obs_noise_driver
    if noise != 0:
        eps = np.broadcast_to(driver[:, None], r_true.shape).astype(np.float32)
        obs_sd = noise * cfg.return_scale / math.sqrt(float(max(cfg.resolutions)))
        r = r_true + obs_sd * eps
    else:
        r = r_true
    agg = aggregate(r, M)
    return np.maximum(np.sum(agg ** 2, axis=1), 1e-12).astype(np.float32)


def causal_ma(x, w):
    x = np.asarray(x, np.float32)
    if w <= 1:
        return x.copy()
    out = np.empty_like(x); cs = np.cumsum(x, dtype=np.float64)
    for t in range(len(x)):
        lo = max(0, t - w + 1)
        out[t] = (cs[t] - (cs[lo - 1] if lo else 0)) / float(t - lo + 1)
    return out


def diff(x, lag):
    out = np.zeros_like(x); lag = max(1, int(lag))
    if lag < len(x): out[lag:] = x[lag:] - x[:-lag]
    return out


def robust_representation(levels, cfg):
    K = levels.shape[1]
    centered = levels - np.mean(levels, axis=0, keepdims=True)
    local_scale = np.maximum(np.std(centered, axis=1), 1e-6)[:, None]
    d1 = np.stack([diff(centered[:, j], cfg.temporal_lag) for j in range(K)], axis=1)
    d4 = np.stack([diff(centered[:, j], cfg.roughness_lag) for j in range(K)], axis=1)
    ma = np.stack([causal_ma(centered[:, j], cfg.smooth_window) for j in range(K)], axis=1)
    nd1 = d1 / local_scale; nd4 = d4 / local_scale
    cols = []
    cols.extend(centered[:, j] for j in range(K))
    cols.extend(levels[:, j] - levels[:, j + 1] for j in range(K - 1))
    cols.extend(d1[:, j] for j in range(K))
    cols.extend(d4[:, j] for j in range(K))
    cols.extend(ma[:, j] for j in range(K))
    cols.extend(nd1[:, j] for j in range(K))
    cols.extend(nd4[:, j] for j in range(K))
    return np.stack(cols, axis=1).astype(np.float32)


def make_representation(path, generator, cfg, noise):
    levels = np.stack([
        np.log(observation_rv(path, generator, cfg, M, noise)) for M in cfg.resolutions
    ], axis=1).astype(np.float32)
    return robust_representation(levels, cfg)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class HurstLSTM(nn.Module):
    def __init__(self, input_size, cfg):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=True, batch_first=True,
        )
        h = cfg.hidden_size * 2
        self.head = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(h, 1))
        self.h_min = cfg.h_min; self.h_max = cfg.h_max

    def encode(self, x):
        h, _ = self.lstm(x)
        return torch.cat([h.mean(dim=1), h[:, -1, :]], dim=1)

    def forward(self, x):
        raw = self.head(self.encode(x)).squeeze(-1)
        return self.h_min + (self.h_max - self.h_min) * torch.sigmoid(raw)


def fit_standardizer_streaming(mm, batch=32):
    C = mm.shape[2]
    s1 = np.zeros(C, np.float64); s2 = np.zeros(C, np.float64); count = 0
    for s in range(0, mm.shape[0], batch):
        e = min(mm.shape[0], s + batch)
        x = np.array(mm[s:e], dtype=np.float32, copy=True).reshape(-1, C).astype(np.float64, copy=False)
        s1 += x.sum(axis=0); s2 += np.square(x).sum(axis=0); count += len(x)
        del x
    mean = s1 / count; var = np.maximum(s2 / count - mean * mean, 1e-12)
    return mean[None, None, :].astype(np.float32), np.sqrt(var)[None, None, :].astype(np.float32)


def create_augmented_cache(paths_by_gen, cfg, cache_dir, name):
    total = sum(len(paths) for paths in paths_by_gen.values()) * len(cfg.train_noise_levels)
    C = len(cfg.resolutions) + (len(cfg.resolutions)-1) + 5*len(cfg.resolutions)
    shape = (total, cfg.seq_len, C)
    path = cache_dir / f"{name}.dat"; ypath = cache_dir / f"{name}_y.npy"
    dtype = np.float16 if cfg.cache_dtype == "float16" else np.float32
    mm = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    y = np.empty(total, np.float32)
    cursor = 0
    per = cfg.cache_build_batch
    for gen, paths in paths_by_gen.items():
        for noise in cfg.train_noise_levels:
            print(f"      cache | generator={gen} noise={noise:g}")
            for s in range(0, len(paths), per):
                e = min(len(paths), s+per)
                feats = np.stack([make_representation(paths[i], gen, cfg, float(noise)) for i in range(s,e)], axis=0).astype(dtype, copy=False)
                mm[cursor+s:cursor+e] = feats
                del feats
            y[cursor:cursor+len(paths)] = np.asarray([p.H_path[-1] for p in paths], dtype=np.float32)
            cursor += len(paths)
            mm.flush()
    np.save(ypath, y)
    return mm, y, path, ypath


def _effective_batch_size(context_length: int, requested: int) -> int:
    """Conservatively shrink the batch as sequence length grows.

    The original v2.13 batch of 128 is unnecessarily aggressive for long CPU
    sequences. This keeps the scientific setup unchanged while controlling peak
    RAM during memmap -> float32 conversion and BiLSTM activations.
    """
    W = int(context_length)
    if W >= 768:
        return min(requested, 8)
    if W >= 512:
        return min(requested, 12)
    if W >= 384:
        return min(requested, 16)
    return min(requested, 32)


def _normalize_memmap_batch(mm, ids, mean, std):
    """Load one batch with low peak memory; normalize in-place on the copy."""
    xb = np.array(mm[ids], dtype=np.float32, copy=True)
    np.subtract(xb, mean, out=xb)
    np.divide(xb, std, out=xb)
    return xb


def train_model_streaming(mm_tr, y_tr, mm_va, y_va, mean, std, cfg, device, seed, context_length):
    model = HurstLSTM(mm_tr.shape[-1], cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss(); rng = np.random.default_rng(seed+1000)
    best = float("inf"); best_state = None; wait = 0; hist=[]
    train_bs = _effective_batch_size(context_length, cfg.batch_size)
    val_bs = min(cfg.validation_batch_size, max(4, train_bs))
    print(f"    memory-safe batches: train={train_bs}, val={val_bs}")
    for epoch in range(1, cfg.epochs+1):
        model.train(); perm = rng.permutation(len(y_tr)); losses=[]
        for s in range(0, len(perm), train_bs):
            ids = perm[s:s+train_bs]
            xb = _normalize_memmap_batch(mm_tr, ids, mean, std)
            yb = np.array(y_tr[ids], dtype=np.float32, copy=True)
            xb_t = torch.from_numpy(xb).to(device); yb_t = torch.from_numpy(yb).to(device)
            opt.zero_grad(set_to_none=True); pred=model(xb_t); loss=loss_fn(pred,yb_t); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step(); losses.append(float(loss.detach().cpu()))
            del xb, yb, xb_t, yb_t, pred, loss
        model.eval(); v=[]
        with torch.no_grad():
            for s in range(0,len(y_va),val_bs):
                e=min(len(y_va),s+val_bs)
                xb = _normalize_memmap_batch(mm_va, slice(s,e), mean, std)
                yb=torch.from_numpy(np.array(y_va[s:e],np.float32,copy=True)).to(device)
                v.append(float(loss_fn(model(torch.from_numpy(xb).to(device)),yb).cpu()))
                del xb, yb
        tr=float(np.mean(losses)); va=float(np.mean(v)); hist.append({"epoch":epoch,"train_mse":tr,"val_mse":va})
        print(f"    epoch={epoch:03d} train={tr:.6f} val={va:.6f}")
        if va < best - 1e-7:
            best=va; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=cfg.patience:
            print("    early stopping"); break
    if best_state is None:
        raise RuntimeError("No validation epoch completed; best_state is empty.")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)


def predict_batch(model, X, mean, std, device):
    model.eval()
    Xn = np.array(X, dtype=np.float32, copy=True)
    np.subtract(Xn, mean, out=Xn)
    np.divide(Xn, std, out=Xn)
    with torch.no_grad():
        out = model(torch.from_numpy(Xn).to(device)).cpu().numpy().astype(np.float64)
    del Xn
    return out




def save_scatter(y, p, path, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, p, s=12, alpha=.4)
    lo=min(float(np.min(y)),float(np.min(p))); hi=max(float(np.max(y)),float(np.max(p)))
    ax.plot([lo,hi],[lo,hi],linestyle="--")
    ax.set_xlabel("True H_last"); ax.set_ylabel("Predicted H_last"); ax.set_title(title); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


# -----------------------------------------------------------------------------
# v2.13 diagnostics: temporal receptive field
# -----------------------------------------------------------------------------


def make_representation_context(path, generator, cfg, noise, context_length):
    """Build robust features using ONLY the requested tail context.

    This is important: centering, temporal differences, moving averages, and
    normalized differences are all recomputed inside the retained context, so
    older observations do not leak into the context-specific representation.
    """
    W = int(context_length)
    if W < 2 or W > cfg.seq_len:
        raise ValueError(f"context_length must be in [2,{cfg.seq_len}].")
    if W % cfg.h_block != 0:
        raise ValueError("context_length must be divisible by h_block for block-aligned interpretation.")

    levels = np.stack([
        np.log(observation_rv(path, generator, cfg, M, noise))[-W:]
        for M in cfg.resolutions
    ], axis=1).astype(np.float32)
    return robust_representation(levels, cfg)


def create_context_cache(paths_by_gen, cfg, cache_dir, name, context_length):
    """Create disk-backed float16 cache for one tail context length."""
    total = sum(len(paths) for paths in paths_by_gen.values()) * len(cfg.train_noise_levels)
    C = len(cfg.resolutions) + (len(cfg.resolutions)-1) + 5*len(cfg.resolutions)
    shape = (total, int(context_length), C)
    dtype = np.float16 if cfg.cache_dtype == "float16" else np.float32
    data_path = cache_dir / f"{name}_W{int(context_length)}.dat"
    ypath = cache_dir / f"{name}_W{int(context_length)}_y.npy"
    mm = np.memmap(data_path, mode="w+", dtype=dtype, shape=shape)
    y = np.empty(total, np.float32)
    cursor = 0
    per = cfg.cache_build_batch
    for gen, paths in paths_by_gen.items():
        for noise in cfg.train_noise_levels:
            print(f"      cache | generator={gen} noise={noise:g} context={int(context_length)}")
            for s in range(0, len(paths), per):
                e = min(len(paths), s + per)
                feats = np.stack([
                    make_representation_context(paths[i], gen, cfg, float(noise), context_length)
                    for i in range(s, e)
                ], axis=0).astype(dtype, copy=False)
                mm[cursor:cursor + (e-s)] = feats
                del feats
                cursor += e-s
            y[cursor-len(paths):cursor] = np.asarray(
                [p.H_path[-1] for p in paths], dtype=np.float32
            )
            mm.flush()
    np.save(ypath, y)
    return mm, y, data_path, ypath


def context_test_predictions(paths, gen, noise, cfg, model, mean, std, device, context_length):
    """Evaluate a model trained on exactly the same tail context length."""
    y = np.asarray([p.H_path[-1] for p in paths], np.float32)
    hm = np.asarray([np.mean(p.H_path) for p in paths], np.float32)
    pred_parts = []
    test_bs = min(cfg.cache_build_batch, 16 if context_length >= 512 else 32)
    for s in range(0, len(paths), test_bs):
        e = min(len(paths), s + test_bs)
        X = np.stack([
            make_representation_context(paths[i], gen, cfg, float(noise), context_length)
            for i in range(s, e)
        ], axis=0)
        pred_parts.append(predict_batch(model, X, mean, std, device))
    pred = np.concatenate(pred_parts)
    met = regression_metrics(y, pred)
    oracle = regression_metrics(y, hm)
    met.update({
        "context_length": int(context_length),
        "h_block_count": int(context_length // cfg.h_block),
        "oracle_Hmean_R2": oracle["R2"],
        "gain_vs_Hmean_oracle": met["R2"] - oracle["R2"],
        "localDev_R2": local_residual_r2(y, pred, hm),
        "partialCorr_Hlast_given_Hmean": partial_corr(y, pred, hm),
    })
    return y, hm, pred, met


def save_context_curve(summary_df, path):
    """Plot R2 against temporal context for each generator/noise."""
    fig, ax = plt.subplots(figsize=(9, 6))
    if summary_df.empty:
        plt.close(fig)
        return
    for (gen, noise), g in summary_df.groupby(["generator", "noise"]):
        g = g.sort_values("context_length")
        ax.plot(g["context_length"], g["R2_mean"], marker="o", label=f"{gen} | noise={noise:g}")
    ax.set_xlabel("Tail context length")
    ax.set_ylabel("R² for H_last")
    ax.set_title("Effective temporal memory saturation curve")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_completed_context_rows(seed_dir, context_length, cfg):
    """Reuse a fully completed context from a previous interrupted run."""
    expected=[]
    for gen in (GEN_BSS, GEN_RBERGOMI):
        for noise in cfg.test_noise_levels:
            expected.append(seed_dir / f"{gen}_noise_{noise:g}_W{int(context_length)}_metrics.json")
    model_path=seed_dir / f"model_state_W{int(context_length)}.pt"
    if not model_path.exists() or not all(p.exists() for p in expected):
        return None
    rows=[]
    for p in expected:
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload=json.load(f)
            row={k:payload[k] for k in (
                "seed","generator","noise","target","experiment","RMSE","MAE","R2","Corr","Bias",
                "context_length","h_block_count","oracle_Hmean_R2","gain_vs_Hmean_oracle",
                "localDev_R2","partialCorr_Hlast_given_Hmean"
            )}
            rows.append(row)
        except Exception:
            return None
    return pd.DataFrame(rows)

def run_seed(seed, cfg, device, out_root):
    set_seed(seed)
    seed_dir = out_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    train_bss = [generate_bss_path(cfg, np.random.default_rng(seed + 100000 + i)) for i in range(cfg.n_train)]
    val_bss = [generate_bss_path(cfg, np.random.default_rng(seed + 200000 + i)) for i in range(cfg.n_val)]
    test_bss = [generate_bss_path(cfg, np.random.default_rng(seed + 300000 + i)) for i in range(cfg.n_test)]
    train_rb = [generate_rbergomi_path(cfg, np.random.default_rng(seed + 400000 + i)) for i in range(cfg.n_train)]
    val_rb = [generate_rbergomi_path(cfg, np.random.default_rng(seed + 500000 + i)) for i in range(cfg.n_val)]
    test_rb = [generate_rbergomi_path(cfg, np.random.default_rng(seed + 600000 + i)) for i in range(cfg.n_test)]
    print(f"  Generated stochastic-H domains in {time.perf_counter()-t0:.1f}s")

    all_rows = []
    cache_root = Path(tempfile.mkdtemp(prefix=f"v212_seed_{seed}_", dir=str(out_root)))
    try:
        for context_length in cfg.context_lengths:
            print(f"\n  CONTEXT LENGTH = {int(context_length)} ({int(context_length//cfg.h_block)} H-blocks)")
            completed = load_completed_context_rows(seed_dir, context_length, cfg)
            if completed is not None:
                print(f"    RESUME: reusing completed context={int(context_length)} from {seed_dir}")
                all_rows.extend(completed.to_dict(orient="records"))
                continue
            context_dir = Path(tempfile.mkdtemp(prefix=f"W{int(context_length)}_", dir=str(cache_root)))
            try:
                mm_tr, y_tr, _, _ = create_context_cache(
                    {GEN_BSS: train_bss, GEN_RBERGOMI: train_rb},
                    cfg, context_dir, "train", context_length
                )
                mm_va, y_va, _, _ = create_context_cache(
                    {GEN_BSS: val_bss, GEN_RBERGOMI: val_rb},
                    cfg, context_dir, "val", context_length
                )
                mean, std = fit_standardizer_streaming(mm_tr)
                np.savez(
                    seed_dir / f"standardizer_W{int(context_length)}.npz",
                    mean=mean, std=std
                )
                model, hist = train_model_streaming(
                    mm_tr, y_tr, mm_va, y_va, mean, std, cfg, device,
                    seed + int(context_length), context_length
                )
                hist.to_csv(seed_dir / f"training_history_W{int(context_length)}.csv", index=False)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": asdict(cfg),
                        "basis": "ORIGINAL_v2.7_stochastic_H",
                        "representation": REP_ROBUST,
                        "context_length": int(context_length),
                        "h_block": int(cfg.h_block),
                    },
                    seed_dir / f"model_state_W{int(context_length)}.pt",
                )

                for noise in cfg.test_noise_levels:
                    print(f"    TEST context={int(context_length)} noise={noise:g}")
                    for gen, paths in ((GEN_BSS, test_bss), (GEN_RBERGOMI, test_rb)):
                        y, hm, pred, met = context_test_predictions(
                            paths, gen, noise, cfg, model, mean, std, device, context_length
                        )
                        row = {
                            "seed": int(seed),
                            "generator": gen,
                            "noise": float(noise),
                            "target": "H_last",
                            "experiment": "temporal_receptive_field",
                            "RMSE": met["RMSE"],
                            "MAE": met["MAE"],
                            "R2": met["R2"],
                            "Corr": met["Corr"],
                            "Bias": met["Bias"],
                            "context_length": int(context_length),
                            "h_block_count": int(context_length // cfg.h_block),
                            "oracle_Hmean_R2": met["oracle_Hmean_R2"],
                            "gain_vs_Hmean_oracle": met["gain_vs_Hmean_oracle"],
                            "localDev_R2": met["localDev_R2"],
                            "partialCorr_Hlast_given_Hmean": met["partialCorr_Hlast_given_Hmean"],
                        }
                        all_rows.append(row)
                        save_scatter(
                            y, pred,
                            seed_dir / f"{gen}_noise_{noise:g}_W{int(context_length)}_scatter.png",
                            f"v2.13 {gen} | context={int(context_length)} | noise={noise:g}"
                        )
                        boot = bootstrap_metrics(
                            y, pred, cfg.bootstrap,
                            seed + 900000 + int(noise * 10000) + int(context_length) + (0 if gen == GEN_BSS else 1)
                        )
                        with open(
                            seed_dir / f"{gen}_noise_{noise:g}_W{int(context_length)}_metrics.json",
                            "w", encoding="utf-8"
                        ) as f:
                            json.dump({**row, "bootstrap": boot}, f, indent=2)
            finally:
                shutil.rmtree(context_dir, ignore_errors=True)
        return pd.DataFrame(all_rows)
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)


def parse_args():
    p = argparse.ArgumentParser(description="HurstNeuralEstimator_v2_13_effective_memory_saturation")
    p.add_argument("--n_train", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=1500)
    p.add_argument("--seq_len", type=int, default=768)
    p.add_argument("--h_min", type=float, default=.05)
    p.add_argument("--h_max", type=float, default=.50)
    p.add_argument("--h_block", type=int, default=32)
    p.add_argument("--h_phi", type=float, default=.97)
    p.add_argument("--h_latent_sd", type=float, default=2.0)
    p.add_argument("--bss_kernel_lambda", type=float, default=.08)
    p.add_argument("--bss_kernel_size", type=int, default=512)
    p.add_argument("--bss_burn_in", type=int, default=512)
    p.add_argument("--bss_xi", type=float, default=.04)
    p.add_argument("--bss_eta", type=float, default=1.0)
    p.add_argument("--rb_xi0", type=float, default=.04)
    p.add_argument("--rb_eta", type=float, default=1.0)
    p.add_argument("--rb_rho", type=float, default=-.70)
    p.add_argument("--T", type=float, default=3.0)
    p.add_argument("--return_scale", type=float, default=1.0)
    p.add_argument("--resolutions", type=int, nargs="+", default=[8,16,32,64])
    p.add_argument("--train_noise_levels", type=float, nargs="+", default=[0,.025,.05,.075,.10])
    p.add_argument("--val_noise_levels", type=float, nargs="+", default=None)
    p.add_argument("--test_noise_levels", type=float, nargs="+", default=[0,.0375,.0875,.10])
    p.add_argument("--temporal_lag", type=int, default=1)
    p.add_argument("--roughness_lag", type=int, default=4)
    p.add_argument("--smooth_window", type=int, default=4)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=.15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--validation_batch_size", type=int, default=16)
    p.add_argument("--cache_build_batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=.002)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--context_lengths", type=int, nargs="+", default=[256,384,512,768])
    p.add_argument("--cache_dtype", choices=["float16","float32"], default="float16")
    p.add_argument("--keep_cache", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[42,123,2024])
    p.add_argument("--out_dir", type=str, default="results_hurst_v2_13_effective_memory_saturation")
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    return p.parse_args()


def main():
    a = parse_args()
    val_noise = a.train_noise_levels if a.val_noise_levels is None else a.val_noise_levels
    context_lengths = tuple(sorted(set(int(x) for x in a.context_lengths)))
    cfg = Config(
        n_train=a.n_train, n_val=a.n_val, n_test=a.n_test,
        seq_len=a.seq_len, h_min=a.h_min, h_max=a.h_max,
        h_block=a.h_block, h_phi=a.h_phi, h_latent_sd=a.h_latent_sd,
        bss_kernel_lambda=a.bss_kernel_lambda, bss_kernel_size=a.bss_kernel_size,
        bss_burn_in=a.bss_burn_in, bss_xi=a.bss_xi, bss_eta=a.bss_eta,
        rb_xi0=a.rb_xi0, rb_eta=a.rb_eta, rb_rho=a.rb_rho, T=a.T,
        return_scale=a.return_scale, resolutions=tuple(a.resolutions),
        train_noise_levels=tuple(a.train_noise_levels),
        val_noise_levels=tuple(val_noise), test_noise_levels=tuple(a.test_noise_levels),
        temporal_lag=a.temporal_lag, roughness_lag=a.roughness_lag,
        smooth_window=a.smooth_window, hidden_size=a.hidden_size,
        num_layers=a.num_layers, dropout=a.dropout, batch_size=a.batch_size,
        validation_batch_size=a.validation_batch_size, cache_build_batch=a.cache_build_batch,
        lr=a.lr, weight_decay=a.weight_decay, epochs=a.epochs, patience=a.patience,
        grad_clip=a.grad_clip, bootstrap=a.bootstrap, context_lengths=context_lengths,
        cache_dtype=a.cache_dtype, keep_cache=a.keep_cache, seeds=tuple(a.seeds),
        out_dir=a.out_dir, device=a.device,
    )
    if cfg.seq_len % cfg.h_block:
        raise ValueError("seq_len must be divisible by h_block")
    if any(W < cfg.h_block or W > cfg.seq_len or W % cfg.h_block for W in cfg.context_lengths):
        raise ValueError("All context_lengths must be multiples of h_block and lie in [h_block, seq_len].")
    Mmax = max(cfg.resolutions)
    if any(Mmax % M for M in cfg.resolutions):
        raise ValueError("All resolutions must divide max resolution")

    device = get_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("="*88)
    print("HurstNeuralEstimator_v2_13_effective_memory_saturation")
    print("="*88)
    print(f"Device: {device}")
    print("Basis: ORIGINAL v2.7 stochastic-H (v2.7-fix1 was NOT used)")
    print("Training: pooled BSS + rBergomi with noise augmentation")
    print("Target: H_last")
    print("Estimator: v2.1-fix2 BiLSTM + mean/last pooling + bounded H")
    print("Representation: multiscale_rv_robust")
    print(f"H_t: blockwise stochastic, block={cfg.h_block}, phi={cfg.h_phi}, latent_sd={cfg.h_latent_sd}")
    print(f"Train noise: {list(cfg.train_noise_levels)}")
    print(f"Test noise: {list(cfg.test_noise_levels)}")
    print(f"Context lengths: {list(cfg.context_lengths)}")
    print(f"Context H-block counts: {[int(W//cfg.h_block) for W in cfg.context_lengths]}")
    print(f"T={cfg.T:g} chosen so dt matches v2.12 at seq_len=256; longer contexts add history rather than changing sampling interval.")
    print("CORE v2.13 TEST: effective-memory saturation via independently trained tail-context models")
    print("Each context recomputes its representation using ONLY that retained tail; no older observations enter centering/lag/smoothing features.")
    print("Interpretation: plateau -> empirical effective memory; continued rise at 768 -> only a lower bound.")

    all_df = []
    for seed in cfg.seeds:
        print("\n" + "-"*88)
        print(f"SEED {seed}")
        all_df.append(run_seed(seed, cfg, device, out))

    all_df = pd.concat(all_df, ignore_index=True)
    all_df.to_csv(out / "temporal_receptive_field_all_results.csv", index=False)

    summary = all_df.groupby(
        ["generator", "noise", "context_length", "h_block_count"], as_index=False
    ).agg(
        RMSE_mean=("RMSE","mean"), RMSE_sd=("RMSE","std"),
        R2_mean=("R2","mean"), R2_sd=("R2","std"),
        Corr_mean=("Corr","mean"), Corr_sd=("Corr","std"),
        localDevR2_mean=("localDev_R2","mean"),
        partialCorr_mean=("partialCorr_Hlast_given_Hmean","mean"),
        gain_vs_Hmean_mean=("gain_vs_Hmean_oracle","mean"),
    )
    summary.to_csv(out / "effective_memory_summary_mean_sd.csv", index=False)
    save_context_curve(summary, out / "effective_memory_saturation_curve.png")

    # Adjacent-context gains make the saturation pattern explicit.
    delta = summary.copy().sort_values(["generator", "noise", "context_length"])
    delta["delta_R2_vs_previous_context"] = delta.groupby(["generator", "noise"])["R2_mean"].diff()
    delta["delta_R2_per_observation"] = delta["delta_R2_vs_previous_context"] / delta.groupby(["generator", "noise"])["context_length"].diff()
    delta.to_csv(out / "effective_memory_adjacent_gains.csv", index=False)

    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("\n" + "="*88)
    print("v2.13 TEMPORAL RECEPTIVE-FIELD COMPLETE")
    print("="*88)
    for _, r in summary.sort_values(["generator","noise","context_length"]).iterrows():
        print(
            f"{r['generator']} | noise={r['noise']:g} | "
            f"context={int(r['context_length'])} ({int(r['h_block_count'])} blocks) | "
            f"R2={r['R2_mean']:.6f}±{r['R2_sd']:.6f} | "
            f"RMSE={r['RMSE_mean']:.6f}±{r['RMSE_sd']:.6f} | "
            f"Corr={r['Corr_mean']:.6f}±{r['Corr_sd']:.6f}"
        )
    print("\nAdjacent-context R² gains:")
    for _, r in delta.dropna(subset=["delta_R2_vs_previous_context"]).iterrows():
        print(
            f"{r['generator']} | noise={r['noise']:g} | context={int(r['context_length'])} | "
            f"ΔR2={r['delta_R2_vs_previous_context']:+.6f}"
        )
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
