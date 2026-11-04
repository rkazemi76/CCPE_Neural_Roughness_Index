#!/usr/bin/env python3
"""New paired synthetic diagnostic. Requires real torch only for --neural."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import reference_numeric as rn
from protocol import config, features as reference_features, verify_sources as verify_reference_sources

SIM, BENCH, CLASSICAL, CLEAR = rn.load_numeric()

CONDITIONS = ('baseline', 'level_0.25', 'level_4', 'eta_0.5', 'eta_1.5', 'smooth_level')
HS = (0.10, 0.25, 0.40)
ROOT_SEED = 48261917


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def verify_sources():
    return verify_reference_sources(ROOT)


def varied_path(base, generator, cfg, condition):
    # Exact reference arithmetic/compensators, including rBergomi t=0.
    eta = float(condition.split('_')[1]) if condition.startswith('eta_') else 1.0
    if generator == 'bss':
        factor = SIM.bss_factor_from_H(base.H_path, cfg, base.innovation_z)
        raw = np.log(cfg.bss_xi) + eta * factor - 0.5 * eta**2
    else:
        raw = (np.log(cfg.rb_xi0) + eta * base.W_H.astype(np.float64)
               - 0.5 * eta**2 * np.maximum(base.time.astype(np.float64), 0)**(2 * base.H_path.astype(np.float64)))
        raw[0] = np.log(cfg.rb_xi0)
    if condition.startswith('level_'):
        raw = raw + np.log(float(condition.split('_')[1]))
    elif condition == 'smooth_level':
        raw = raw + np.log(4.) * np.sin(2 * np.pi * np.arange(1, cfg.seq_len + 1) / cfg.seq_len)
    clips = int(np.count_nonzero((raw < -15.) | (raw > 8.)))
    logv = np.clip(raw, -15., 8.).astype(np.float32)
    variance = np.exp(logv).astype(np.float32)
    if generator == 'bss':
        path = dataclasses.replace(base, latent_variance=variance, latent_logvol=logv)
        drivers = ('innovation_z', 'fine_eps', 'obs_noise_driver', 'H_path')
    else:
        path = dataclasses.replace(base, variance=variance, log_variance=logv)
        drivers = ('W_H', 'dW_fine', 'dWperp', 'fine_spot_dB', 'obs_noise_driver', 'H_path')
    for key in drivers:
        np.testing.assert_array_equal(getattr(path, key), getattr(base, key))
    if condition == 'baseline':
        np.testing.assert_array_equal(variance, base.latent_variance if generator == 'bss' else base.variance)
    return path, clips, variance, logv


def build_arrays(n, cfg):
    meta = []
    arrays = {rep: np.empty((2*3*n*6*2, cfg.context_length, channels), dtype=np.float32)
              for rep, channels in [('single_raw',1), ('single_derived',6), ('multi_original',27)]}
    invariance = []
    count = 0
    for gi, generator in enumerate(('bss', 'rbergomi')):
        for hi, h in enumerate(HS):
            for pi in range(n):
                rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, gi, hi, pi]))
                base = BENCH.generate_path(SIM, generator, cfg, rng, 'constant')
                constant_h = np.full(cfg.seq_len, h, dtype=np.float32)
                base = (SIM.make_bss_counterfactual(base, constant_h, cfg) if generator == 'bss'
                        else SIM.make_rb_counterfactual(base, constant_h, cfg))
                np.testing.assert_array_equal(base.H_path, constant_h)
                references = {}
                for condition in CONDITIONS:
                    path, clips, variance, logv = varied_path(base, generator, cfg, condition)
                    for noise in (0., 0.1):
                        logrv = np.stack([np.log(SIM.observation_rv(path, generator, cfg, M, noise))
                                          for M in cfg.resolutions], axis=1)[-cfg.context_length:]
                        assert logrv.shape == (cfg.context_length, len(cfg.resolutions))
                        f = {rep: reference_features(logrv, rep, SIM, cfg)[None] for rep in arrays}
                        if condition == 'baseline':
                            references[noise] = (logrv.copy(), {k: v.copy() for k, v in f.items()}, clips)
                        if noise == 0. and condition.startswith('level_'):
                            ref, rf, rc = references[noise]
                            expected = np.log(float(condition.split('_')[1]))
                            err = float(np.max(np.abs(logrv - ref - expected)))
                            if not clips and not rc:
                                assert err < 2e-5, (generator, h, condition, err)
                            invariance.append(dict(generator=generator, H=h, path_index=pi,
                                condition=condition, clipped=bool(clips or rc), logrv_shift_max_error=err,
                                **{k + '_feature_max_change': float(np.max(np.abs(f[k]-rf[k])))
                                   for k in ('single_raw','single_derived','multi_original')}))
                        meta.append(dict(row_id=count, pair_id=f'{generator}_H{h:.2f}_p{pi:04d}',
                            generator=generator, H=h, path_index=pi, noise=noise,
                            condition=condition, clipped_points=clips,
                            mean_variance=float(np.mean(variance)),
                            latent_logvariance_sd=float(np.std(logv))))
                        for rep in arrays:
                            arrays[rep][count] = f[rep][0]
                        count += 1
                    CLEAR(path)
                CLEAR(base)
            print(f'Generated {generator} H={h}: {count} rows', flush=True)
    assert all(np.isfinite(a).all() for a in arrays.values())
    return pd.DataFrame(meta), arrays, pd.DataFrame(invariance)


def summarize(predictions, out, nboot):
    keys = ['generator','H','noise','method']
    metrics, contrasts = [], []
    rng = np.random.default_rng(7301921)
    for labels, frame in predictions.groupby(keys, sort=True):
        common = dict(zip(keys, labels))
        base = frame[frame.condition == 'baseline'].set_index('pair_id').prediction
        for condition, cell in frame.groupby('condition', sort=True):
            assert not cell.pair_id.duplicated().any()
            error = cell.prediction.to_numpy() - labels[1]
            metrics.append(dict(**common, condition=condition, n=len(cell),
                mean_prediction=float(cell.prediction.mean()), bias=float(error.mean()),
                mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error**2)))))
            if condition == 'baseline':
                continue
            assert set(cell.pair_id) == set(base.index)
            delta = cell.prediction.to_numpy() - base.loc[cell.pair_id].to_numpy()
            row = dict(**common, condition=condition, n=len(delta),
                mean_paired_change=float(delta.mean()), mean_absolute_paired_change=float(np.abs(delta).mean()),
                max_absolute_paired_change=float(np.abs(delta).max()))
            if nboot:
                draws = delta[rng.integers(0, len(delta), size=(nboot, len(delta)))].mean(axis=1)
                row.update(zip(('pointwise_ci_low','pointwise_ci_high'), map(float, np.quantile(draws,[.025,.975]))))
            contrasts.append(row)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', choices=['smoke','full'], default='smoke')
    ap.add_argument('--neural', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()
    if sys.flags.optimize:
        ap.error('Do not run with -O or PYTHONOPTIMIZE; assertions are required.')
    if args.threads < 1:
        ap.error('--threads must be positive')
    if args.out.exists():
        ap.error('output directory already exists; use a fresh directory')
    args.out.mkdir(parents=True)
    out = args.out
    start = time.perf_counter()
    try:
        checked = verify_sources()
        if args.neural:
            import torch
            if not hasattr(torch, '__version__') or not hasattr(torch, 'nn'):
                raise RuntimeError('Real PyTorch is required')
            torch.set_num_threads(args.threads)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            from fixed_h_neural import FrozenEnsemble
        n = 3 if args.profile == 'smoke' else 200
        cfg = config(SIM, args.profile)
        cfg.test_noise_levels = [0., 0.1]  # This diagnostic's levels, for both profiles.
        dump(out/'protocol.json', dict(profile=args.profile, neural_requested=args.neural,
            n_per_generator_H=n, root_seed=ROOT_SEED, config=vars(cfg),
            conditions=CONDITIONS, H=HS, bootstrap_replicates=0 if n==3 else 2000,
            protocol_sha256=digest(ROOT/'PROTOCOL.md'), source_files_verified=checked))
        dump(out/'environment.json', dict(python=sys.version, platform=platform.platform(),
            numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__ if args.neural else None,
            device='cpu', threads=args.threads, neural_executed=False))
        meta, arrays, inv = build_arrays(n, cfg)
        meta.to_csv(out/'design.csv', index=False)
        inv.to_csv(out/'noiseless_scale_controls.csv', index=False)
        np.savez_compressed(out/'features.npz', **arrays)
        predictions = []
        for method in ('variogram','dfa'):
            fn, scales = ((CLASSICAL.variogram_h, cfg.benchmark_scales) if method == 'variogram'
                          else (CLASSICAL.dfa_h, cfg.dfa_scales))
            values = [float(fn(x[:,0], scales)) for x in arrays['single_raw']]
            assert np.isfinite(values).all()
            frame = meta.copy()
            frame['method'] = 'uncalibrated_' + method
            frame['prediction'] = values
            predictions.append(frame)
        tests = dict(source_hashes=True, paired_drivers_and_H=True, finite_features=True,
            noiseless_logrv_scale_control=True, baseline_generator_parity=True,
            unique_design_rows=not meta.duplicated(['pair_id','noise','condition']).any(),
            expected_row_count=len(meta)==2*3*n*6*2)
        if args.neural:
            checkpoints = {}
            for variant in ('single_raw','single_derived','multi_original'):
                X = arrays[variant]
                ensemble = FrozenEnsemble(variant, cfg, device='cpu')
                checkpoints[variant] = ensemble.checkpoint_hashes
                print(f'Inference {variant}', flush=True)
                members = ensemble.predict_individual(X, batch_size=32)
                assert members.shape == (len(meta),3)
                assert np.isfinite(members).all() and (members>=cfg.h_min).all() and (members<=cfg.h_max).all()
                rows = np.linspace(0,len(meta)-1, min(24,len(meta)), dtype=int)
                serial = ensemble.predict_individual(X[rows], batch_size=1)
                np.testing.assert_allclose(members[rows], serial, atol=2e-6, rtol=2e-5)
                np.testing.assert_allclose(ensemble.predict(X[rows]), members[rows].mean(axis=1), atol=2e-6, rtol=2e-5)
                for j, seed in enumerate((42,123,2024)):
                    frame = meta.copy(); frame['method'] = f'{variant}_seed_{seed}'
                    frame['prediction'] = members[:,j]; predictions.append(frame)
                frame = meta.copy(); frame['method'] = variant + '_ensemble'
                frame['prediction'] = members.mean(axis=1); predictions.append(frame)
            dump(out/'checkpoints.json', checkpoints)
            tests.update(real_torch_inference=True, batch_serial_parity=True, ensemble_parity=True,
                         finite_bounded_neural_predictions=True)
        pred = pd.concat(predictions, ignore_index=True)
        pred.to_csv(out/'predictions.csv', index=False)
        summarize(pred, out, 0 if n==3 else 2000)
        dump(out/'tests.json', tests)
        assert all(tests.values())
        env = json.loads((out/'environment.json').read_text()); env['neural_executed']=args.neural
        dump(out/'environment.json',env)
        dump(out/'status.json',dict(status='PASS', profile=args.profile, neural_executed=args.neural,
            interpretation='Engineering test only' if n==3 else 'New controlled synthetic diagnostic',
            rows=len(meta), prediction_rows=len(pred), elapsed_seconds=time.perf_counter()-start))
        dump(out/'manifest.json', {p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()})
        print((out/'status.json').read_text())
    except Exception as exc:
        dump(out/'status.json',dict(status='FAILED', neural_results_complete=False,
            error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter()-start))
        raise


if __name__ == '__main__':
    main()
