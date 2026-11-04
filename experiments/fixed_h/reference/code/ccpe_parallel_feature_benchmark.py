#!/usr/bin/env python3
"""Reference CPU benchmark for parallel synthetic-path and feature construction.

The benchmark operates on independent, deterministically seeded synthetic paths.
Each task constructs one 27-channel, multiscale realized-variance representation.
A process pool is used only for this data-generation/feature stage; neural-network
training and empirical forecasting are not distributed by this script.

Example (PowerShell, from package root):
  python code/ccpe_parallel_feature_benchmark.py `
    --simulator_script code/HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix.py `
    --out_dir results/ccpe_computational_profile --n_paths 384 --seq_len 384 `
    --noise_level 0.05 --workers 1 2 3 --repetitions 3

The program writes a CSV of wall-clock summaries and a JSON environment manifest.
On Windows, process-based execution is protected by the __main__ guard and a
worker initializer. Deterministic per-task seeds make the output independent of
worker count and scheduling order.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SIM: Any = None


def load_simulator(path: Path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Simulator script not found: {path}")
    spec = importlib.util.spec_from_file_location("ccpe_parallel_simulator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import simulator at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def init_worker(simulator_path: str) -> None:
    global _SIM
    _SIM = load_simulator(Path(simulator_path))


def construct_features(task: tuple[str, int, dict[str, Any], float]):
    """Generate one seeded synthetic path and return its features and seed."""
    global _SIM
    generator, seed, cfg_kwargs, noise = task
    if _SIM is None:
        raise RuntimeError("Simulator was not initialized in this process")
    cfg = _SIM.Config(**cfg_kwargs)
    rng = np.random.default_rng(int(seed))
    if generator == "bss":
        path = _SIM.generate_bss_path(cfg, rng)
    elif generator == "rbergomi":
        path = _SIM.generate_rbergomi_path(cfg, rng)
    else:
        raise ValueError(f"Unknown generator: {generator}")
    features = _SIM.make_representation(path, generator, cfg, float(noise))
    return int(seed), np.asarray(features, dtype=np.float32)


def run_batch(tasks, workers: int, simulator_path: Path, simulator_module):
    if workers == 1:
        global _SIM
        _SIM = simulator_module
        return [construct_features(task) for task in tasks]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(str(simulator_path.resolve()),),
    ) as pool:
        return list(pool.map(construct_features, tasks, chunksize=4))


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulator_script",
        default=str(package_root / "code" / "HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix.py"),
    )
    parser.add_argument("--out_dir", default=str(package_root / "results" / "ccpe_computational_profile"))
    parser.add_argument("--n_paths", type=int, default=384)
    parser.add_argument("--seq_len", type=int, default=384)
    parser.add_argument("--noise_level", type=float, default=0.05)
    parser.add_argument("--base_seed", type=int, default=81000)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    if args.n_paths < 1 or args.seq_len < 32 or args.repetitions < 1:
        parser.error("n_paths and repetitions must be positive; seq_len must be at least 32")
    if any(w < 1 for w in args.workers):
        parser.error("worker counts must be positive integers")
    workers_to_test = sorted(set(args.workers))
    if 1 not in workers_to_test:
        parser.error("include worker count 1 so speedup can be normalized to the serial baseline")
    if args.seq_len % 32:
        parser.error("seq_len must be divisible by the default H block length (32)")

    sim_path = Path(args.simulator_script).expanduser().resolve()
    sim = load_simulator(sim_path)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = sim.Config(seq_len=args.seq_len)
    cfg_kwargs = {
        key: getattr(cfg, key)
        for key in sim.Config.__dataclass_fields__
        if key in {
            "seq_len", "h_min", "h_max", "h_block", "h_phi", "h_latent_sd",
            "bss_kernel_lambda", "bss_kernel_size", "bss_burn_in", "bss_xi", "bss_eta",
            "rb_xi0", "rb_eta", "rb_rho", "T", "return_scale", "resolutions",
            "temporal_lag", "roughness_lag", "smooth_window",
        }
    }

    rows = []
    for generator in ("bss", "rbergomi"):
        tasks = [
            (generator, args.base_seed + i, cfg_kwargs, args.noise_level)
            for i in range(args.n_paths)
        ]
        reference = None
        baseline_seconds = None
        for workers in workers_to_test:
            run_seconds = []
            current_result = None
            for _ in range(args.repetitions):
                t0 = time.perf_counter()
                current_result = run_batch(tasks, workers, sim_path, sim)
                run_seconds.append(time.perf_counter() - t0)
            elapsed = float(statistics.median(run_seconds))
            current_result = sorted(current_result, key=lambda item: item[0])

            if reference is None:
                reference = current_result
            max_abs_diff = max(
                float(np.max(np.abs(ref[1] - got[1])))
                for ref, got in zip(reference, current_result)
            )
            if workers == 1:
                baseline_seconds = elapsed
            rows.append({
                "generator": generator,
                "n_paths": args.n_paths,
                "seq_len": args.seq_len,
                "feature_channels": int(current_result[0][1].shape[1]),
                "noise_level": args.noise_level,
                "workers": workers,
                "median_wall_seconds": elapsed,
                "paths_per_second": args.n_paths / elapsed,
                "speedup_vs_serial": (baseline_seconds / elapsed) if baseline_seconds else np.nan,
                "max_abs_feature_diff_vs_first_worker_count": max_abs_diff,
                "repetitions": args.repetitions,
                "base_seed": args.base_seed,
            })

    result = pd.DataFrame(rows)
    csv_path = out_dir / "parallel_feature_benchmark.csv"
    result.to_csv(csv_path, index=False)

    try:
        import psutil
        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
        ram_total = round(psutil.virtual_memory().total / 1024**3, 2)
        ram_available = round(psutil.virtual_memory().available / 1024**3, 2)
    except Exception:
        logical = os.cpu_count()
        physical = None
        ram_total = None
        ram_available = None
    manifest = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "logical_cpu_count_reported": logical,
        "physical_cpu_count_reported": physical,
        "total_ram_gb_reported": ram_total,
        "available_ram_gb_at_start_gb": ram_available,
        "simulator_script": sim_path.name,
        "simulator_basis": "ORIGINAL_v2.7_stochastic_H",
        "task": "independent synthetic path generation plus 27-channel feature construction",
        "n_paths_per_generator": args.n_paths,
        "sequence_length": args.seq_len,
        "noise_level": args.noise_level,
        "base_seed": args.base_seed,
        "workers_tested": workers_to_test,
        "repetitions_per_configuration": args.repetitions,
        "neural_network_training_parallelized": False,
        "empirical_forecasting_parallelized": False,
    }
    manifest_path = out_dir / "parallel_feature_benchmark_environment.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(result.round(4).to_string(index=False))
    print(f"\nWrote: {csv_path}\nWrote: {manifest_path}")


if __name__ == "__main__":
    main()
