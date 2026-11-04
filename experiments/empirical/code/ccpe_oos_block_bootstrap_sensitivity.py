#!/usr/bin/env python3
"""Sensitivity of the existing timestamp-clustered OOS GARCH benchmark to bootstrap block length.

Reads the v2.23 walk-forward prediction CSV from the reproducibility package. It does not
refit any model or change forecasts. All five assets at a timestamp remain in the same
resampled cluster. Output is inference sensitivity only, not a new OOS sample.
"""
from __future__ import annotations
import argparse, zipfile
from pathlib import Path
import numpy as np
import pandas as pd


def read_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            options = [n for n in z.namelist() if n.endswith('v223_walkforward_predictions.csv')]
            if len(options) != 1:
                raise ValueError(f'Expected exactly one v223 prediction CSV in {path}; found {options}')
            with z.open(options[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def loss_series(df: pd.DataFrame, model: str, loss: str) -> pd.Series:
    if loss == 'MSE':
        return (df['actual_logRV'].to_numpy(float) - df.loc[:, 'predicted_logRV'].to_numpy(float)) ** 2
    if loss == 'MAE':
        return np.abs(df['actual_logRV'].to_numpy(float) - df.loc[:, 'predicted_logRV'].to_numpy(float))
    if loss == 'QLIKE':
        y = np.maximum(df['actual_rv'].to_numpy(float), 1e-18)
        p = np.maximum(df['predicted_rv_qlike'].to_numpy(float), 1e-18)
        r = y / p
        return r - np.log(r) - 1.0
    raise ValueError(loss)


def ci_moving_block(values: np.ndarray, block_len: int, reps: int, rng: np.random.Generator):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    n = len(values); L = max(1, min(int(block_len), n))
    if n < 5:
        raise ValueError(f'Only {n} timestamps available')
    out = np.empty(reps, float)
    for b in range(reps):
        picked = []
        while len(picked) < n:
            start = int(rng.integers(0, n - L + 1))
            picked.extend(range(start, start + L))
        out[b] = values[np.asarray(picked[:n], int)].mean()
    return float(values.mean()), float(np.quantile(out, .025)), float(np.quantile(out, .975))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions', default='results/v2_23_garch/v223_walkforward_predictions.csv')
    ap.add_argument('--out_dir', default='results/v2_25_oos_block_length_sensitivity_recomputed')
    ap.add_argument('--reps', type=int, default=10000)
    ap.add_argument('--block_lengths', nargs='+', type=int, default=[1,2,3,4,5,6])
    ap.add_argument('--seed', type=int, default=20261024)
    args = ap.parse_args()
    df = read_predictions(Path(args.predictions))
    needed = {'timestamp','symbol','model','actual_logRV','actual_rv','predicted_logRV','predicted_rv_qlike'}
    if not needed.issubset(df.columns): raise ValueError(f'Missing columns: {sorted(needed-set(df.columns))}')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    models = ['Naive_Current_RV','HAR_RV','GARCH11_RV','HAR_RV_plus_Neural_H','HAR_RV_plus_Variogram_H']
    comparisons = [
        ('HAR_RV_plus_Neural_H','Naive_Current_RV'),
        ('HAR_RV_plus_Neural_H','HAR_RV'),
        ('HAR_RV_plus_Neural_H','GARCH11_RV'),
        ('HAR_RV_plus_Neural_H','HAR_RV_plus_Variogram_H'),
        ('GARCH11_RV','HAR_RV'),
    ]
    long = []
    for (ts, sym), g in df.groupby(['timestamp','symbol'], sort=True):
        actuals = g[['actual_logRV','actual_rv']].drop_duplicates()
        if len(actuals) != 1: raise ValueError(f'Inconsistent actuals at {ts} {sym}')
    # Pair model prediction rows before calculating losses.
    for loss in ['MSE','MAE','QLIKE']:
        pivot_col = 'predicted_logRV' if loss in ('MSE','MAE') else 'predicted_rv_qlike'
        wide = df.pivot_table(index=['timestamp','symbol'], columns='model', values=pivot_col, aggfunc='first').reset_index()
        actual = df.drop_duplicates(['timestamp','symbol']).set_index(['timestamp','symbol'])
        wide = wide.merge(actual[['actual_logRV','actual_rv']], left_on=['timestamp','symbol'], right_index=True, how='inner')
        for a,b in comparisons:
            if a not in wide or b not in wide: continue
            if loss == 'MSE':
                la = (wide['actual_logRV'].to_numpy(float)-wide[a].to_numpy(float))**2
                lb = (wide['actual_logRV'].to_numpy(float)-wide[b].to_numpy(float))**2
            elif loss == 'MAE':
                la = np.abs(wide['actual_logRV'].to_numpy(float)-wide[a].to_numpy(float))
                lb = np.abs(wide['actual_logRV'].to_numpy(float)-wide[b].to_numpy(float))
            else:
                y = np.maximum(wide['actual_rv'].to_numpy(float),1e-18)
                pa = np.maximum(wide[a].to_numpy(float),1e-18); pb=np.maximum(wide[b].to_numpy(float),1e-18)
                ra=y/pa; rb=y/pb
                la=ra-np.log(ra)-1; lb=rb-np.log(rb)-1
            d = pd.DataFrame({'timestamp':wide['timestamp'], 'diff':la-lb})
            # Equal five-asset clusters: average loss differential within timestamp.
            bytime = d.groupby('timestamp',sort=True)['diff'].mean().to_numpy(float)
            for L in args.block_lengths:
                rng=np.random.default_rng(args.seed + 1009*L + 17*len(long))
                est,lo,hi=ci_moving_block(bytime,L,args.reps,rng)
                long.append({'model_a':a,'model_b':b,'loss':loss,'delta_loss_a_minus_b':est,
                             'ci95_lower':lo,'ci95_upper':hi,'block_length_timestamps':L,
                             'n_oos_timestamps':len(bytime),'n_asset_forecasts':int(len(wide)),
                             'bootstrap_reps':args.reps,'ci_excludes_zero':bool(lo>0 or hi<0),
                             'direction':'negative favors model_a; positive favors model_b'})
    result=pd.DataFrame(long)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/'oos_block_bootstrap_sensitivity.csv',index=False)
    grouped=result.groupby(['model_a','model_b','loss']).agg(
        min_ci_lower=('ci95_lower','min'),max_ci_lower=('ci95_lower','max'),
        min_ci_upper=('ci95_upper','min'),max_ci_upper=('ci95_upper','max'),
        lengths_excluding_zero=('ci_excludes_zero','sum'),n_block_lengths=('block_length_timestamps','count'),
        n_oos_timestamps=('n_oos_timestamps','first')).reset_index()
    grouped['conclusion_stable_across_lengths']=grouped['lengths_excluding_zero'].eq(grouped['n_block_lengths'])
    grouped.to_csv(out/'oos_block_bootstrap_sensitivity_summary.csv',index=False)
    print('OOS COVERAGE:',int(df.timestamp.nunique()),'timestamps;',len(df.drop_duplicates(['timestamp','symbol'])),'asset-time rows')
    print(result.pivot_table(index=['model_a','model_b','loss'],columns='block_length_timestamps',values=['delta_loss_a_minus_b','ci95_lower','ci95_upper']).round(5).to_string())
    print('\nWrote',out)
if __name__=='__main__': main()
