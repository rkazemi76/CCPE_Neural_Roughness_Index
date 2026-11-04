"""Shared numerical protocol; no torch dependency."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from reference_numeric import ROOT, load_numeric, load_evaluation

sys.path.insert(0, str(ROOT / 'reference/code'))
from matched_features import single_features, split_seed

REPS = ('single_raw', 'single_derived', 'multi_original')
SEEDS = (42, 123, 2024)
DATA_SEED = 960831  # new evaluation, not the historical 190317 data seed


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def verify_sources(root=ROOT):
    provenance = json.loads((root / 'SOURCE_PROVENANCE.json').read_text())
    entries = provenance['sources']
    destinations = [x['destination'] for x in entries]
    if len(set(destinations)) != len(destinations):
        raise ValueError('Duplicate provenance destinations')
    for entry in entries:
        path = root / entry['destination']
        if not path.is_file() or sha_file(path) != entry['sha256']:
            raise ValueError('Source hash mismatch: ' + entry['destination'])
    return len(entries)


def config(sim, mode):
    cfg = sim.Config()
    source = json.loads((ROOT / 'reference/config.json').read_text())['config']
    for k, v in source.items():
        setattr(cfg, k, v)
    cfg.test_noise_levels = [0., .1] if mode == 'smoke' else source['test_noise_levels']
    cfg.cache_dtype = 'float32'
    cfg.out_dir = '(unused reference setting)'
    cfg.device = 'cpu'
    return cfg


def features(levels, rep, sim, cfg):
    levels = np.asarray(levels, dtype=np.float32)
    if levels.shape != (384, 4) or not np.isfinite(levels).all():
        raise ValueError('Expected finite final 384 x 4 log-RV window')
    if rep == 'multi_original':
        return sim.robust_representation(levels, cfg).astype(np.float32)
    if rep == 'single_raw':
        return levels[:, -1:].copy()  # do not charge for unused derived channels
    if rep == 'single_derived':
        return single_features(levels[:, -1])[rep]
    raise ValueError(rep)


def rng_seed(split, h_mode, data_seed=DATA_SEED):
    if h_mode not in ('constant', 'stochastic'):
        raise ValueError(h_mode)
    return data_seed + split_seed(split, 0) + (10000000 if h_mode == 'constant' else 0)


def check_split_keys(frames):
    sets = [set(map(tuple, f[['rng_seed', 'rng_generator', 'rng_path']].drop_duplicates().to_numpy()))
            for f in frames]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError('Overlapping RNG keys across evaluation splits')
    historic = {rng_seed(s, m, 190317) for s in ('train', 'validation', 'calibration', 'test')
                for m in ('stochastic', 'constant')}
    if any(k[0] in historic for keys in sets for k in keys):
        raise ValueError('RNG base overlaps documented historical protocol')
    return True


def make_data(out, mode):
    sim, bench, classical, clear = load_numeric()
    cfg = config(sim, mode)
    frames = []
    ncal, ntest = (4, 6) if mode == 'smoke' else (100, 400)
    for split, n in [('calibration', ncal), ('test', ntest)]:
        for hm in ('stochastic', 'constant'):
            dest = out / 'data' / f'{split}_{hm}'
            dest.mkdir(parents=True)
            count = n * 2 * len(cfg.test_noise_levels)
            levels = np.lib.format.open_memmap(dest / 'levels.npy', mode='w+', dtype='float32',
                                               shape=(count, 384, 4))
            rows = []
            seed = rng_seed(split, hm)
            for gi, gen in enumerate(cfg.generators):
                for i in range(n):
                    path = bench.generate_path(sim, gen, cfg,
                        np.random.default_rng(np.random.SeedSequence([seed, gi, i])), hm)
                    try:
                        for noise in cfg.test_noise_levels:
                            window = np.stack([np.log(sim.observation_rv(path, gen, cfg, M, noise))[-384:]
                                               for M in cfg.resolutions], axis=1).astype(np.float32)
                            if not np.isfinite(window).all():
                                raise ValueError('Nonfinite log-RV')
                            levels[len(rows)] = window
                            rows.append(dict(split=split, h_mode=hm, generator=gen, noise=noise,
                                path_id=f'{split}:{hm}:{gen}:{i}', rng_seed=seed,
                                rng_generator=gi, rng_path=i, target=float(path.H_path[-1]),
                                variogram_raw=float(classical.variogram_h(window[:, -1], cfg.benchmark_scales)),
                                dfa_raw=float(classical.dfa_h(window[:, -1], cfg.dfa_scales))))
                    finally:
                        clear(path)
            levels.flush()
            frame = pd.DataFrame(rows)
            frame.to_csv(dest / 'rows.csv', index=False)
            frames.append(frame)
            write_json(dest / 'manifest.json', dict(shape=list(levels.shape),
                levels_sha256=sha_file(dest / 'levels.npy'), rows_sha256=sha_file(dest / 'rows.csv')))
            del levels
    check_split_keys(frames)
    write_json(out / 'protocol.json', dict(mode=mode, data_seed=DATA_SEED,
        calibration_paths_per_generator_and_H_mode=ncal, test_paths_per_generator_and_H_mode=ntest,
        noise_levels=cfg.test_noise_levels, sequence_length=768, context_length=384,
        target='H_last', evaluation_rng_disjoint=True,
        historical_rng_comparison='distinct bases from documented 190317 protocol only; not proof of all training membership',
        new_analysis=True, historical_reproduction=False, retraining=False,
        calibration='held-out affine per H-mode/generator/noise; clip [0.05,0.50]',
        bootstrap='path within cell; conditional on fixed calibration; pointwise, unadjusted',
        config=cfg.__dict__))


def load_frames(out):
    result = {}
    for split in ('calibration', 'test'):
        result[split] = pd.concat([pd.read_csv(out / 'data' / f'{split}_{hm}' / 'rows.csv')
                                  for hm in ('stochastic', 'constant')], ignore_index=True)
    return result


def add_baseline(cal, test):
    for key, group in cal.groupby(['h_mode', 'generator', 'noise']):
        mask = np.ones(len(test), bool)
        for column, value in zip(['h_mode', 'generator', 'noise'], key):
            mask &= test[column].to_numpy() == value
        cal.loc[group.index, 'calibration_mean'] = float(group.target.mean())
        test.loc[mask, 'calibration_mean'] = float(group.target.mean())


def evaluate(out, cal, test, methods, mode, prefix):
    keys = ['h_mode', 'generator', 'noise']
    if not methods or len(methods) != len(set(methods)) or 'calibration_mean' in methods:
        raise ValueError('Expected unique estimator methods without reserved baseline name')
    cells = []
    for frame in (cal, test):
        if frame.empty or frame[keys + ['path_id']].isna().any().any():
            raise ValueError('Empty data or missing cell/path labels')
        if frame.duplicated(keys + ['path_id']).any():
            raise ValueError('Duplicate latent path within evaluation cell')
        if not np.isfinite(frame[['target'] + list(methods)].to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite evaluation values')
        cells.append(set(map(tuple, frame[keys].drop_duplicates().to_numpy())))
    if cells[0] != cells[1]:
        raise ValueError('Calibration/test cells differ')
    if set(cal.path_id) & set(test.path_id):
        raise ValueError('Calibration/test path overlap')
    if all({'rng_seed', 'rng_generator', 'rng_path'} <= set(f.columns) for f in (cal, test)):
        check_split_keys([cal, test])
    add_baseline(cal, test)
    methods = list(methods) + ['calibration_mean']
    scores, pairs, coefficients, predictions = load_evaluation().evaluate(
        cal, test, methods, 100 if mode == 'smoke' else 2000, 98301)
    cal.to_csv(out / f'{prefix}_calibration_predictions.csv', index=False)
    predictions.to_csv(out / f'{prefix}_test_predictions.csv', index=False)
    scores.to_csv(out / f'{prefix}_metrics.csv', index=False)
    pairs.to_csv(out / f'{prefix}_paired_bootstrap.csv', index=False)
    coefficients.to_csv(out / f'{prefix}_calibration_coefficients.csv', index=False)
    return scores
