"""Frozen-checkpoint evaluation and synchronized inference cost. Requires real torch."""
import time
import json
import numpy as np
import pandas as pd
import psutil
import torch
from reference_numeric import ROOT, SIM, selected, load_numeric
from protocol import REPS, SEEDS, config, features, load_frames, evaluate, write_json


def model_class():
    return selected(SIM, ['HurstLSTM'], 'compute_frozen_architecture',
                    {'torch': torch, 'nn': torch.nn}).HurstLSTM


def load_models(rep, cfg, device):
    cls = model_class()
    models, records = [], []
    channels = dict(single_raw=1, single_derived=6, multi_original=27)[rep]
    with np.load(ROOT / 'reference' / (rep + '_scaler.npz'), allow_pickle=False) as scaler:
        mean, std = scaler['mean'].reshape(-1), scaler['std'].reshape(-1)
    if mean.shape != (channels,) or std.shape != (channels,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid scaler for ' + rep)
    for seed in SEEDS:
        directory = ROOT / 'reference/models' / rep / f'seed_{seed}'
        start = time.perf_counter()
        saved = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
        if saved['representation'] != rep or saved['seed'] != seed:
            raise ValueError('Checkpoint identity mismatch')
        for key in ('hidden_size', 'num_layers', 'dropout', 'h_min', 'h_max', 'context_length'):
            if saved['config'][key] != getattr(cfg, key):
                raise ValueError('Checkpoint configuration mismatch: ' + key)
        model = cls(channels, cfg)
        model.load_state_dict(saved['state_dict'], strict=True)
        model = model.to(device).eval()
        if device == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        arch = json.loads((directory / 'architecture.json').read_text())
        count = sum(p.numel() for p in model.parameters())
        if arch['parameter_count'] != count or arch['input_channels'] != channels:
            raise ValueError('Architecture metadata mismatch')
        records.append(dict(representation=rep, seed=seed, parameters=count,
            parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            checkpoint_bytes=(directory / 'model.pt').stat().st_size, load_seconds=seconds))
        models.append(model)
    return models, mean, std, records


def tensor_batch(levels, rep, sim, cfg, mean, std, device):
    x = np.stack([features(a, rep, sim, cfg) for a in levels])
    x = ((x - mean) / std).astype(np.float32)
    return torch.from_numpy(x).to(device)


def forward(models, x):
    if len(models) == 1:
        return models[0](x)
    return torch.stack([m(x) for m in models]).mean(dim=0)


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def time_call(fn, device, repeats, label, batch, scope):
    # Warm calls are excluded; device is synchronized before and after each timed call.
    for _ in range(3):
        fn()
    synchronize(device)
    rows = []
    process = psutil.Process()
    for i in range(repeats):
        synchronize(device)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        rss_before = process.memory_info().rss
        start = time.perf_counter()
        result = fn()
        synchronize(device)
        seconds = time.perf_counter() - start
        rows.append(dict(method=label, scope=scope, batch_size=batch, repeat=i,
            seconds=seconds, milliseconds_per_window=1000*seconds/batch,
            windows_per_second=batch/seconds, rss_before_bytes=rss_before,
            rss_after_bytes=process.memory_info().rss,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None))
        del result
    return rows


def run(out, mode, device, threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(98301)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    sim, _, classical, _ = load_numeric()
    cfg = config(sim, mode)
    results = load_frames(out)
    sets = {(split, hm): np.load(out / 'data' / f'{split}_{hm}' / 'levels.npy', mmap_mode='r')
            for split in ('calibration', 'test') for hm in ('stochastic', 'constant')}
    sample = sets[('test', 'stochastic')]
    # Repeat available windows only for timing if a smoke set has <64 examples.
    sample = np.array(sample[np.arange(64) % len(sample)], copy=True)
    methods = ['variogram_raw', 'dfa_raw']
    timings, model_records, parity = [], [], []
    repeats = 3 if mode == 'smoke' else 15
    with torch.inference_mode():
        for rep in REPS:
            models, mean, std, records = load_models(rep, cfg, device)
            model_records.extend(records)
            for j, seed in enumerate(SEEDS):
                method = f'{rep}_seed{seed}'
                methods.append(method)
                for split in ('calibration', 'test'):
                    predicted = []
                    for hm in ('stochastic', 'constant'):
                        levels = sets[(split, hm)]
                        for start in range(0, len(levels), 16):
                            x = tensor_batch(levels[start:start+16], rep, sim, cfg, mean, std, device)
                            predicted.extend(models[j](x).cpu().numpy().reshape(-1))
                    results[split][method] = predicted
            ensemble = rep + '_ensemble'
            methods.append(ensemble)
            for frame in results.values():
                frame[ensemble] = frame[[f'{rep}_seed{s}' for s in SEEDS]].mean(axis=1)
            # Actual, not mocked, neural checks: serial/batch, repeatability and ensemble parity.
            probe = tensor_batch(sample[:4], rep, sim, cfg, mean, std, device)
            for j, model in enumerate(models):
                batch_pred = model(probe).cpu().numpy()
                serial_pred = np.concatenate([model(x[None]).cpu().numpy() for x in probe])
                np.testing.assert_allclose(batch_pred, serial_pred, rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(batch_pred, model(probe).cpu().numpy(), rtol=2e-5, atol=2e-6)
                if not np.isfinite(batch_pred).all() or np.any(batch_pred < .05) or np.any(batch_pred > .5):
                    raise ValueError('Invalid model predictions')
                parity.append(dict(representation=rep, seed=SEEDS[j], batch_serial_max_abs=float(abs(batch_pred-serial_pred).max())))
            np.testing.assert_allclose(forward(models, probe).cpu().numpy(),
                np.mean([m(probe).cpu().numpy() for m in models], axis=0), rtol=2e-5, atol=2e-6)
            for label, group in [(f'{rep}_seed{s}', [m]) for s, m in zip(SEEDS, models)] + [(ensemble, models)]:
                for batch in (1, 16, 64):
                    levels = sample[:batch]
                    x = tensor_batch(levels, rep, sim, cfg, mean, std, device)
                    def cached():
                        return forward(group, x)
                    def pipeline():
                        z = tensor_batch(levels, rep, sim, cfg, mean, std, device)
                        return forward(group, z).cpu().numpy()
                    np.testing.assert_allclose(cached().cpu().numpy(), pipeline(), rtol=2e-5, atol=2e-6)
                    timings.extend(time_call(cached, device, repeats, label, batch, 'device_resident_forward'))
                    timings.extend(time_call(pipeline, device, repeats, label, batch, 'logrv_to_prediction'))
            del models, model, group, probe, x
            if device == 'cuda':
                torch.cuda.empty_cache()
        for label, fn, scales in [('variogram_raw', classical.variogram_h, cfg.benchmark_scales),
                                  ('dfa_raw', classical.dfa_h, cfg.dfa_scales)]:
            for batch in (1, 16, 64):
                timings.extend(time_call(lambda: np.array([fn(a[:, -1], scales) for a in sample[:batch]]),
                    'cpu', repeats, label, batch, 'logrv_to_prediction'))
    scores = evaluate(out, results['calibration'], results['test'], methods, mode, 'neural')
    raw = pd.DataFrame(timings)
    raw.to_csv(out / 'latency_raw.csv', index=False)
    summary = raw.groupby(['method', 'scope', 'batch_size']).agg(
        median_ms_per_window=('milliseconds_per_window', 'median'),
        min_ms_per_window=('milliseconds_per_window', 'min'),
        max_ms_per_window=('milliseconds_per_window', 'max'),
        median_windows_per_second=('windows_per_second', 'median')).reset_index()
    summary.to_csv(out / 'latency_summary.csv', index=False)
    # Only raw metrics join: calibration arithmetic is NOT included in timed inference.
    joined = scores.merge(summary.loc[summary.scope == 'logrv_to_prediction'], on='method', how='inner')
    joined.to_csv(out / 'accuracy_cost.csv', index=False)
    pd.DataFrame(model_records).to_csv(out / 'models.csv', index=False)
    write_json(out / 'neural_correctness.json', dict(batch_serial_parity=parity,
        pipeline_cached_parity=True, ensemble_parity=True, all_predictions_finite=True,
        rtol=2e-5, atol=2e-6, torch_version=torch.__version__, device=device,
        device_name=torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
        memory_scope='RSS before/after are not peaks. Stage RSS includes data and all live models. CUDA peak includes three resident models per representation; not isolated per-model memory.',
        timing_scope='warm model, 384-point log-RV to prediction; excludes simulation, RV construction, I/O, calibration and training',
        ensemble='three models serially, not concurrent', threads=threads))
