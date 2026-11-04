#!/usr/bin/env python3
"""Exploratory constant-H sensitivity check for empirical temporal H dispersion.

This is a diagnostic, not a formal test that market H is stochastic. It compares
observed within-asset SDs of the three-seed empirical ensemble with a simple
simulation-based measurement-error reference built from constant-H synthetic
paths predicted by the available frozen seed-42 W384 estimator. The synthetic
null does not reproduce serial dependence between market windows and uses a
single-seed model rather than a three-seed ensemble; interpretations must remain
exploratory.

Run from package root:
  python code/ccpe_constant_H_temporal_variability_sensitivity.py
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel_dir', default=str(root / 'inputs' / 'results_hurst_v2_15_empirical_crypto_multiasset'))
    parser.add_argument('--constant_h_predictions', default=str(root / 'results' / 'v2_22_constant_H' / 'v222_test_predictions.csv'))
    parser.add_argument('--out_dir', default=str(root / 'results' / 'v2_24_empirical_variability_null'))
    parser.add_argument('--bootstrap_reps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=987654)
    parser.add_argument('--local_h_band', type=float, default=0.025)
    parser.add_argument('--noise_levels', nargs='+', type=float, default=[0.0375, 0.0875])
    args = parser.parse_args()
    if args.bootstrap_reps < 1000 or args.local_h_band <= 0:
        parser.error('bootstrap_reps must be >= 1000 and local_h_band must be positive')

    panel_dir = Path(args.panel_dir)
    panel_frames = []
    for path in sorted(panel_dir.glob('*_rolling_estimates.csv')):
        frame = pd.read_csv(path)
        frame['asset'] = path.name.split('_')[0]
        panel_frames.append(frame)
    if not panel_frames:
        raise FileNotFoundError(f'No *_rolling_estimates.csv files found in {panel_dir}')
    panel = pd.concat(panel_frames, ignore_index=True)
    required = {'asset', 'H_neural_mean'}
    if not required.issubset(panel.columns):
        raise ValueError(f'Panel missing required columns: {sorted(required - set(panel.columns))}')

    const = pd.read_csv(args.constant_h_predictions)
    required_const = {'generator', 'noise', 'H_last', 'H_neural'}
    if not required_const.issubset(const.columns):
        raise ValueError(f'Constant-H predictions missing columns: {sorted(required_const - set(const.columns))}')
    rng = np.random.default_rng(args.seed)
    results = []
    for asset, asset_df in panel.groupby('asset', sort=True):
        target_h = float(asset_df['H_neural_mean'].mean())
        observed_sd = float(asset_df['H_neural_mean'].std(ddof=1))
        n_contexts = int(asset_df['H_neural_mean'].notna().sum())
        for generator in ('bss', 'rbergomi'):
            for noise in args.noise_levels:
                sample = const[(const['generator'] == generator) & np.isclose(const['noise'], noise)].copy()
                if len(sample) < 100:
                    raise ValueError(f'Not enough constant-H rows for {generator=} {noise=}: {len(sample)}')
                # Estimate and remove the linear response of the one-seed estimator
                # to the known constant H; resampled residuals approximate error
                # variation around a fixed H in the local neighborhood.
                slope, intercept = np.polyfit(sample['H_last'].to_numpy(float), sample['H_neural'].to_numpy(float), 1)
                sample['residual'] = sample['H_neural'] - (intercept + slope * sample['H_last'])
                local = sample[np.abs(sample['H_last'] - target_h) <= args.local_h_band]
                errors = local['residual'].to_numpy(float)
                if len(errors) < n_contexts:
                    raise ValueError(f'Insufficient local paths for {asset=} {generator=} {noise=}: {len(errors)}')
                draw_idx = rng.integers(0, len(errors), size=(args.bootstrap_reps, n_contexts))
                sd_draws = np.std(errors[draw_idx], axis=1, ddof=1)
                results.append({
                    'asset': asset,
                    'target_empirical_mean_Hhat': target_h,
                    'observed_temporal_sd_ensemble': observed_sd,
                    'n_contexts': n_contexts,
                    'generator': generator,
                    'noise': float(noise),
                    'local_h_band': float(args.local_h_band),
                    'local_constant_H_paths': int(len(errors)),
                    'null_sd_median': float(np.median(sd_draws)),
                    'null_sd_q95': float(np.quantile(sd_draws, .95)),
                    'null_sd_q99': float(np.quantile(sd_draws, .99)),
                    'observed_sd_percentile': float(np.mean(sd_draws <= observed_sd)),
                    'sd_ratio_obs_over_null_median': float(observed_sd / np.median(sd_draws)),
                    'linear_response_slope': float(slope),
                    'linear_response_intercept': float(intercept),
                    'bootstrap_reps': int(args.bootstrap_reps),
                    'seed': int(args.seed),
                })
    detail = pd.DataFrame(results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / 'constant_H_temporal_variability_sensitivity.csv', index=False)
    summary = detail.groupby('asset', sort=True).agg(
        observed_temporal_sd=('observed_temporal_sd_ensemble', 'first'),
        null_median_sd_min=('null_sd_median', 'min'),
        null_median_sd_max=('null_sd_median', 'max'),
        null_q95_min=('null_sd_q95', 'min'),
        null_q95_max=('null_sd_q95', 'max'),
        max_observed_sd_percentile=('observed_sd_percentile', 'max'),
        min_observed_sd_percentile=('observed_sd_percentile', 'min'),
    ).reset_index()
    summary.to_csv(out_dir / 'constant_H_temporal_variability_summary.csv', index=False)
    config = {
        'purpose': 'Exploratory check whether empirical temporal SD exceeds an approximate constant-H estimator-error reference',
        'empirical_statistic': 'sample SD of H_neural_mean across 57 windows per asset; ensemble average of three frozen seeds',
        'synthetic_null': 'constant-H synthetic predictions from v2.22; one frozen seed-42 W384 estimator; fit linear Hhat~Htrue per generator/noise, resample residuals in sets matching empirical n_contexts',
        'local_H_band': args.local_h_band,
        'noise_levels': args.noise_levels,
        'generators': ['bss', 'rbergomi'],
        'bootstrap_reps': args.bootstrap_reps,
        'seed': args.seed,
        'limitations': [
            'Not a formal p-value or a calibrated state-space test.',
            'Synthetic contexts are resampled independently and do not preserve temporal dependence across empirical windows.',
            'Synthetic reference uses a single model seed, whereas the empirical statistic averages three seeds.',
            'Synthetic generators/noise are only approximations to exchange-market data; the observation noise parameter is not directly identified with market microstructure noise.'
        ]
    }
    import json
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(summary.round(4).to_string(index=False))
    print(f'\nWrote diagnostic outputs to {out_dir}')

if __name__ == '__main__':
    main()
