#!/usr/bin/env python3
"""Smooth semimartingale out-of-domain control for frozen W384 checkpoints.

Uses continuous-time OU log-variance, exact OU transitions on the coarse grid,
64 fine Gaussian returns per coarse interval, the paper's observation-noise
mechanism, and a vectorized implementation checked against its official
feature builder. No retraining is performed.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / 'reference'
sys.path.insert(0, str(REFERENCE))
import HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix as core
BASE = REFERENCE / 'checkpoints'


def robust_representation_batch(levels, cfg):
    # levels: N x T x K; exactly mirrors core.robust_representation for each path.
    centered = levels - np.mean(levels, axis=1, keepdims=True)
    local_scale = np.maximum(np.std(centered, axis=2, keepdims=True), 1e-6)
    d1 = np.zeros_like(centered); d1[:, 1:, :] = centered[:, 1:, :] - centered[:, :-1, :]
    lag = int(cfg.roughness_lag)
    d4 = np.zeros_like(centered); d4[:, lag:, :] = centered[:, lag:, :] - centered[:, :-lag, :]
    cs = np.cumsum(centered, axis=1, dtype=np.float64)
    ma = np.empty_like(centered)
    w = int(cfg.smooth_window)
    for t in range(centered.shape[1]):
        lo = max(0, t - w + 1)
        prev = cs[:, lo-1, :] if lo else 0.0
        ma[:, t, :] = (cs[:, t, :] - prev) / float(t - lo + 1)
    nd1 = d1 / local_scale
    nd4 = d4 / local_scale
    cols = []
    K = levels.shape[2]
    cols.extend(centered[:, :, j] for j in range(K))
    cols.extend(levels[:, :, j] - levels[:, :, j+1] for j in range(K-1))
    cols.extend(d1[:, :, j] for j in range(K))
    cols.extend(d4[:, :, j] for j in range(K))
    cols.extend(ma[:, :, j] for j in range(K))
    cols.extend(nd1[:, :, j] for j in range(K))
    cols.extend(nd4[:, :, j] for j in range(K))
    return np.stack(cols, axis=2).astype(np.float32)


def make_smooth_batch(cfg, n_paths, seed, kappa, stationary_sd=1.0):
    rng = np.random.default_rng(seed)
    n = cfg.seq_len; mmax = max(cfg.resolutions)
    dt = cfg.T / float(n)
    rho = math.exp(-kappa * dt)
    innov_sd = stationary_sd * math.sqrt(1.0-rho*rho)
    x = np.empty((n_paths,n), dtype=np.float64)
    x[:,0] = rng.normal(0.0, stationary_sd, size=n_paths)
    shocks = rng.standard_normal((n_paths,n-1))
    for t in range(1,n):
        x[:,t] = rho*x[:,t-1] + innov_sd*shocks[:,t-1]
    logv = math.log(cfg.bss_xi) + x - 0.5*stationary_sd**2
    logv = np.clip(logv,-15.0,8.0)
    v = np.exp(logv).astype(np.float32)
    eps = rng.standard_normal((n_paths,n,mmax)).astype(np.float32)
    obs = rng.standard_normal((n_paths,n)).astype(np.float32)
    # Match the paper's BSS-style fine-return scaling (Mmax=64).
    base_returns = np.sqrt(v / float(mmax))[:,:,None] * eps
    return base_returns, obs, logv.astype(np.float32)


def build_features(base_returns, obs, cfg, noise):
    n_paths, n, mmax = base_returns.shape
    if noise:
        r = base_returns + np.float32(noise / math.sqrt(float(mmax))) * obs[:,:,None]
    else:
        r = base_returns
    levels = np.empty((n_paths,n,len(cfg.resolutions)),dtype=np.float32)
    for j,M in enumerate(cfg.resolutions):
        block = mmax // M
        agg = r.reshape(n_paths,n,M,block).sum(axis=3)
        rv = np.maximum(np.sum(agg*agg,axis=2),1e-12)
        levels[:,:,j] = np.log(rv)
    X = robust_representation_batch(levels[:,-384:,:],cfg)
    return X


def metric_summary(pred, seed):
    pred=np.asarray(pred,float); rng=np.random.default_rng(9173+int(seed) if isinstance(seed,int) else 9183)
    b=rng.choice(pred,size=(3000,len(pred)),replace=True).mean(axis=1)
    q=np.quantile(pred,[.05,.25,.5,.75,.95])
    return {
      'n':len(pred),'mean_prediction':float(pred.mean()),
      'mean_ci95_low':float(np.quantile(b,.025)),'mean_ci95_high':float(np.quantile(b,.975)),
      'sd_prediction':float(pred.std(ddof=1)), 'q05':float(q[0]), 'q25':float(q[1]),
      'median':float(q[2]),'q75':float(q[3]),'q95':float(q[4]),
      'fraction_ge_0_45':float(np.mean(pred>=.45)), 'fraction_ge_0_475':float(np.mean(pred>=.475)),
      'fraction_le_0_40':float(np.mean(pred<=.40)),
      'mean_bias_from_boundary_0_5':float(pred.mean()-.5)
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n-paths',type=int,default=200)
    ap.add_argument('--noise-levels',type=float,nargs='+',default=[0.0,0.05,0.10])
    ap.add_argument('--kappas',type=float,nargs='+',default=[0.5,3.0])
    ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--out',type=Path,default=HERE.parent/'evidence/smooth_semimartingale_reproduced')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    ckpt0=torch.load(BASE/'seed_42/model_state_W384.pt',map_location='cpu',weights_only=False)
    cfg=core.Config(**ckpt0['config']); device=torch.device('cpu')
    models={}; means={}; stds={}
    for s in (42,123,2024):
        c=torch.load(BASE/f'seed_{s}/model_state_W384.pt',map_location='cpu',weights_only=False)
        model=core.HurstLSTM(input_size=27,cfg=cfg); model.load_state_dict(c['state_dict']); model.eval()
        z=np.load(BASE/f'seed_{s}/standardizer_W384.npz')
        models[s]=model; means[s]=z['mean']; stds[s]=z['std']
    rows=[]
    for kappa in args.kappas:
        base_returns, obs, _ = make_smooth_batch(cfg,args.n_paths,args.seed+int(round(kappa*1000)),float(kappa))
        for noise in args.noise_levels:
            X=build_features(base_returns,obs,cfg,float(noise))
            # Cross-check vectorized feature builder against the original function on one path.
            if noise == args.noise_levels[0] and kappa == args.kappas[0]:
                # Reconstruct BSSPath only for an independent parity check of one path.
                p=core.BSSPath(H_path=np.full(cfg.seq_len,.5,np.float32),latent_logvol=np.zeros(cfg.seq_len,np.float32),
                   latent_variance=np.ones(cfg.seq_len,np.float32),fine_eps=np.zeros((cfg.seq_len,64),np.float32),
                   obs_noise_driver=np.zeros(cfg.seq_len,np.float32),innovation_z=np.zeros(cfg.seq_len+cfg.bss_burn_in+cfg.bss_kernel_size+8))
                # Direct parity check on actual arrays, bypassing generator path type.
                # official builder recomputes returns as sqrt(V/64)*eps; inject base returns as eps with V=1.
                p.fine_eps = (base_returns[0] * math.sqrt(64.0)).astype(np.float32)
                p.latent_variance = np.ones(cfg.seq_len,np.float32)
                p.obs_noise_driver = obs[0].copy()
                x0=core.make_representation_context(p,core.GEN_BSS,cfg,float(noise),384)
                err=float(np.max(np.abs(x0-X[0])))
                if err > 2e-5: raise RuntimeError(f'feature parity check failed: max_abs_err={err}')
                (args.out/'feature_parity.json').write_text(json.dumps({'max_abs_error':err,'tolerance':2e-5},indent=2))
            for s,model in models.items():
                pr=core.predict_batch(model,X,means[s],stds[s],device)
                for i,v in enumerate(pr):
                    rows.append({'kappa':float(kappa),'noise':float(noise),'seed':int(s),'path_id':int(i),'prediction':float(v)})
        del base_returns, obs
        print(f'completed kappa={kappa}',flush=True)
    df=pd.DataFrame(rows); df.to_csv(args.out/'smooth_semimartingale_predictions.csv',index=False)
    summary=[]
    for (kappa,noise,seed),g in df.groupby(['kappa','noise','seed']):
        row={'process':'OU_semimartingale','kappa':float(kappa),'stationary_logvariance_sd':1.0,'noise':float(noise),'checkpoint_seed':int(seed)}
        row.update(metric_summary(g.prediction.to_numpy(),int(seed))); summary.append(row)
    ens=df.groupby(['kappa','noise','path_id'],as_index=False)['prediction'].mean()
    for (kappa,noise),g in ens.groupby(['kappa','noise']):
        row={'process':'OU_semimartingale','kappa':float(kappa),'stationary_logvariance_sd':1.0,'noise':float(noise),'checkpoint_seed':'ensemble_mean'}
        row.update(metric_summary(g.prediction.to_numpy(),'ensemble')); summary.append(row)
    sdf=pd.DataFrame(summary); sdf.to_csv(args.out/'smooth_semimartingale_summary.csv',index=False)
    meta={
      'experiment':'Frozen-checkpoint out-of-domain smooth semimartingale control',
      'latent_process':'dX_t = -kappa X_t dt + sigma dW_t, with stationary_sd(X)=1 and sigma=sqrt(2*kappa); log V_t=log(xi)+X_t-0.5.',
      'discretization':'Exact OU transition on n=768 coarse points over T=3, piecewise-constant spot variance within each coarse interval; 64 independent Gaussian fine-return increments per coarse interval.',
      'observation_model':'Same 64-fine-return BSS-style scaling and additive observation-noise driver/scaling as the frozen checkpoint experiment; four-resolution RV and 27-channel feature code parity-checked.',
      'n_paths_per_kappa':args.n_paths,'kappas':args.kappas,'noise_levels':args.noise_levels,
      'stationary_logvariance_sd':1.0,'T':cfg.T,'seq_len':cfg.seq_len,'fine_steps_per_coarse':max(cfg.resolutions),
      'context_length':384,'seeds':[42,123,2024],'training_noise_levels':list(cfg.train_noise_levels),
      'training_target_range':[cfg.h_min,cfg.h_max],'brownian_semimartingale_boundary_reference_H':0.5,
      'interpretation':'The OU controls are out-of-domain semimartingale paths. H=0.5 is a Brownian-regularity boundary reference, not a supervised target observed for these paths. Predictions are diagnostic outputs from frozen rough-trained models, not calibrated estimates of OU roughness.',
      'random_seed':args.seed,'path_seed_rule':'For each kappa, the NumPy path seed equals base_seed + round(1000*kappa); the same latent paths are reused across the three observation-noise amplitudes.','checkpoint_package':'Reproducibility_Package_Roughness_Crypto_v0_6 W384 checkpoints and standardizers, seeds 42, 123, 2024.'
    }
    (args.out/'protocol.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(sdf.to_string(index=False),flush=True)

if __name__=='__main__': main()
