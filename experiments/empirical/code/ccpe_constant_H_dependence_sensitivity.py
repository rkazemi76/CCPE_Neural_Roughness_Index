#!/usr/bin/env python3
"""Dependence-aware sensitivity analysis for temporal dispersion of the neural H index.

This is a conservative diagnostic, not a formal test that latent market H is stochastic.
It uses the available single-seed W384 constant-H synthetic predictions, and therefore
cannot claim seed-matched ensemble calibration. It (i) block-bootstraps the empirical
57-window series to retain local serial dependence in uncertainty around observed SD,
and (ii) adds AR(1) dependence to the synthetic constant-H estimation residuals, with
rho set to the observed lag-1 autocorrelation and its approximate 95% Fisher-z upper bound.
"""
from __future__ import annotations
import argparse, zipfile, json
from pathlib import Path
import numpy as np
import pandas as pd


def read_panel(panel_dir: Path) -> pd.DataFrame:
    fs=[]
    for p in sorted(panel_dir.glob('*_rolling_estimates.csv')):
        d=pd.read_csv(p); d['asset']=p.name.split('_')[0]; fs.append(d)
    if not fs: raise FileNotFoundError(f'No panel CSVs in {panel_dir}')
    out=pd.concat(fs,ignore_index=True)
    out['timestamp']=pd.to_datetime(out['timestamp'],utc=True)
    return out.sort_values(['asset','timestamp']).reset_index(drop=True)


def read_csv_from_zip_or_file(path: Path, suffix: str) -> pd.DataFrame:
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            candidates=[n for n in z.namelist() if n.endswith(suffix)]
            if len(candidates)!=1: raise ValueError(f'Need one {suffix} in zip; found {candidates}')
            with z.open(candidates[0]) as f: return pd.read_csv(f)
    return pd.read_csv(path)


def lag1_rho(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<4 or np.std(x[:-1])<1e-12 or np.std(x[1:])<1e-12: return 0.0
    return float(np.corrcoef(x[:-1],x[1:])[0,1])


def fisher_upper(rho,n):
    rho=float(np.clip(rho,-.98,.98))
    if n<=4: return min(.8,max(0,rho))
    z=np.arctanh(rho); upper=np.tanh(z+1.96/np.sqrt(n-3))
    return float(np.clip(upper,0,.85))


def moving_block_sd_bootstrap(x, block, reps, rng):
    x=np.asarray(x,float); n=len(x); L=max(1,min(block,n))
    nblocks=int(np.ceil(n/L)); starts=rng.integers(0,n-L+1,size=(reps,nblocks))
    offsets=np.arange(L)[None,None,:]
    idx=(starts[:,:,None]+offsets).reshape(reps,-1)[:,:n]
    draws=x[idx]
    out=np.std(draws,axis=1,ddof=1)
    return float(np.quantile(out,.025)),float(np.quantile(out,.975))


def residual_pool(const, h_target, local_band):
    x=const['H_last'].to_numpy(float); y=const['H_neural'].to_numpy(float)
    ok=np.isfinite(x)&np.isfinite(y)
    X=np.column_stack([np.ones(ok.sum()),x[ok]])
    beta=np.linalg.lstsq(X,y[ok],rcond=None)[0]
    resid=y[ok]-(beta[0]+beta[1]*x[ok])
    local=np.abs(x[ok]-h_target)<=local_band
    e=resid[local]
    if len(e)<100: e=resid
    e=e-np.mean(e)
    return e, float(beta[0]),float(beta[1]),int(len(e))


def draw_iid_sd(errors,n,reps,rng):
    idx=rng.integers(0,len(errors),size=(reps,n))
    return np.std(errors[idx],axis=1,ddof=1)


def draw_ar1_sd(errors,n,reps,rho,rng):
    """Stationary-marginal AR(1) bootstrap driven by resampled synthetic residuals."""
    errors=np.asarray(errors,float); errors=errors-errors.mean()
    sigma=float(np.std(errors,ddof=1)); rho=float(np.clip(rho,0,.85))
    e=np.empty((reps,n),float); e[:,0]=errors[rng.integers(0,len(errors),size=reps)]
    innovation=errors[rng.integers(0,len(errors),size=(reps,n-1))]
    scale=np.sqrt(max(1-rho*rho,1e-8))
    for t in range(1,n): e[:,t]=rho*e[:,t-1]+scale*innovation[:,t-1]
    # Do not renormalize each replicate: the finite-sample SD distribution is the statistic of interest.
    return np.std(e,axis=1,ddof=1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel_dir',default='inputs/results_hurst_v2_15_empirical_crypto_multiasset')
    ap.add_argument('--constant_h_predictions',default='results/v2_22_constant_H/v222_test_predictions.csv')
    ap.add_argument('--out_dir',default='results/v2_24_constant_H_dependence_sensitivity_recomputed')
    ap.add_argument('--reps',type=int,default=10000)
    ap.add_argument('--local_h_band',type=float,default=.025)
    ap.add_argument('--noise_levels',nargs='+',type=float,default=[.0375,.0875])
    ap.add_argument('--seed',type=int,default=20261024)
    ap.add_argument('--null_model_description',default='single frozen seed-42 W384 checkpoint; independent synthetic contexts with AR(1)-residual sensitivity')
    args=ap.parse_args()
    panel=read_panel(Path(args.panel_dir))
    const=read_csv_from_zip_or_file(Path(args.constant_h_predictions),'v222_test_predictions.csv')
    const=const[(const['sample']=='test') & const['generator'].isin(['bss','rbergomi'])].copy()
    rng=np.random.default_rng(args.seed)
    details=[]; obs_summary=[]
    for asset,g in panel.groupby('asset',sort=True):
        g=g.sort_values('timestamp'); h=g['H_neural_mean'].to_numpy(float); h=h[np.isfinite(h)]
        n=len(h); sd=float(np.std(h,ddof=1)); rho=lag1_rho(h); rho_u=fisher_upper(rho,n)
        obs_ci={}
        for L in [2,3,4,5,6]:
            lo,hi=moving_block_sd_bootstrap(h,L,args.reps,np.random.default_rng(args.seed+L+sum(map(ord,asset))))
            obs_ci[L]=(lo,hi)
        obs_summary.append({'asset':asset,'n_windows':n,'observed_temporal_sd':sd,'lag1_autocorrelation':rho,
                            'lag1_rho_fisher95_upper':rho_u,
                            **{f'block{L}_sd_ci95_lower':obs_ci[L][0] for L in obs_ci},
                            **{f'block{L}_sd_ci95_upper':obs_ci[L][1] for L in obs_ci}})
        target_h=float(np.mean(h))
        for gen in ['bss','rbergomi']:
            for noise in args.noise_levels:
                s=const[(const.generator==gen)&np.isclose(const.noise,noise)]
                if len(s)<100: continue
                err,intercept,slope,nlocal=residual_pool(s,target_h,args.local_h_band)
                for dep_name,r in [('iid',0.0),('ar1_empirical_rho',max(0.0,rho)),('ar1_fisher_upper',rho_u)]:
                    local_rng=np.random.default_rng(args.seed + 31*len(details)+int(round(noise*10000)))
                    null=draw_iid_sd(err,n,args.reps,local_rng) if dep_name=='iid' else draw_ar1_sd(err,n,args.reps,r,local_rng)
                    perc=float(np.mean(null<=sd)); q95=float(np.quantile(null,.95)); q99=float(np.quantile(null,.99))
                    details.append({'asset':asset,'n_windows':n,'observed_mean_Hhat':target_h,'observed_temporal_sd':sd,
                        'empirical_lag1_autocorrelation':rho,'assumed_null_ar1_rho':float(r), 'dependence_scenario':dep_name,
                        'generator':gen,'noise':float(noise),'local_Hhat_band':args.local_h_band,'n_synthetic_local_residuals':nlocal,
                        'residual_sd':float(np.std(err,ddof=1)),'null_sd_median':float(np.median(null)),'null_sd_q95':q95,'null_sd_q99':q99,
                        'observed_sd_null_percentile':perc,'upper_tail_exceedance_rate':float(np.mean(null>=sd)),
                        'exceeds_null_q95':bool(sd>q95),'linear_response_intercept':intercept,'linear_response_slope':slope,
                        'bootstrap_reps':args.reps,'seed':args.seed,'synthetic_model_seed':args.null_model_description})
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(details).to_csv(out/'constant_H_dependence_sensitivity.csv',index=False)
    pd.DataFrame(obs_summary).to_csv(out/'empirical_H_series_block_bootstrap.csv',index=False)
    config={'purpose':'Dependence-aware sensitivity analysis of temporal dispersion of empirical H estimates under an approximate constant-H measurement-error reference',
      'empirical series':'57 non-overlapping windows per asset; H_neural_mean averages seeds 42,123,2024',
      'synthetic reference':args.null_model_description+'; residuals from local linear Hhat~Htrue response',
      'dependence handling':'moving-block bootstrap of empirical H series for SD uncertainty; synthetic residual AR(1) sensitivity uses empirical lag-1 rho and Fisher-z upper bound',
      'not a formal test':'the AR(1) residual sensitivity is an approximation, not a continuous-long-path parametric bootstrap; interpret results descriptively',
      'reps':args.reps,'local_h_band':args.local_h_band,'noise_levels':args.noise_levels,'seed':args.seed}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(pd.DataFrame(obs_summary)[['asset','observed_temporal_sd','lag1_autocorrelation','lag1_rho_fisher95_upper','block3_sd_ci95_lower','block3_sd_ci95_upper']].round(4).to_string(index=False))
    print('\nNull summary by dependence scenario:')
    d=pd.DataFrame(details)
    print(d.groupby(['dependence_scenario']).agg(n=('asset','size'),exceed_q95=('exceeds_null_q95','sum'),min_obs_percentile=('observed_sd_null_percentile','min'),max_obs_percentile=('observed_sd_null_percentile','max'),min_q95=('null_sd_q95','min'),max_q95=('null_sd_q95','max')).round(4).to_string())
    print('\nWrote',out)
if __name__=='__main__': main()
