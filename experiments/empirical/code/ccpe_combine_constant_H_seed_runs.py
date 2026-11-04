#!/usr/bin/env python3
"""Combine matched v2.22 constant-H test predictions from independent model seeds.

All seed runs must use identical simulator code, h_mode=constant, calibration_seed,
test_seed, sample counts and noise levels. The script verifies row-key / target alignment,
then averages H_neural over the supplied checkpoints. It keeps per-seed columns so that
between-seed variability can also be inspected.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEYS=['sample_id','sample','generator','noise']

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',nargs='+',required=True,help='Prediction CSV paths, ordered by model seed')
    ap.add_argument('--seed_ids',nargs='+',type=int,required=True,help='Seed IDs matching --inputs order')
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    if len(args.inputs)!=len(args.seed_ids) or len(args.inputs)<2:
        ap.error('Provide at least two --inputs and the same number of --seed_ids')
    base=None; pred_cols=[]
    for path,seed in zip(args.inputs,args.seed_ids):
        d=pd.read_csv(path)
        required=set(KEYS+['H_last','H_context_mean_oracle','H_neural'])
        if not required.issubset(d.columns): raise ValueError(f'{path} missing {sorted(required-set(d.columns))}')
        d=d[d['sample']=='test'].copy()
        keycols=KEYS+['H_last','H_context_mean_oracle']
        d=d[keycols+['H_neural']].rename(columns={'H_neural':f'H_neural_seed_{seed}'})
        col=f'H_neural_seed_{seed}'; pred_cols.append(col)
        if base is None: base=d
        else:
            base=base.merge(d,on=keycols,how='outer',validate='one_to_one',indicator=True)
            if not (base['_merge']=='both').all(): raise ValueError(f'Rows do not align for seed {seed}')
            base=base.drop(columns='_merge')
    # Model seed outputs should agree on target/keys. Duplicate seed IDs are not allowed.
    if len(set(args.seed_ids))!=len(args.seed_ids): ap.error('Duplicate seed IDs')
    arr=base[pred_cols].to_numpy(float)
    base['H_neural']=np.mean(arr,axis=1)
    base['H_neural_seed_sd']=np.std(arr,axis=1,ddof=1)
    base['H_neural_seed_min']=np.min(arr,axis=1)
    base['H_neural_seed_max']=np.max(arr,axis=1)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    base.to_csv(out,index=False)
    print(f'Combined {len(args.seed_ids)} seeds across {len(base):,} matched test rows -> {out}')
    print('Mean within-context seed SD:',float(base.H_neural_seed_sd.mean()))
if __name__=='__main__': main()
