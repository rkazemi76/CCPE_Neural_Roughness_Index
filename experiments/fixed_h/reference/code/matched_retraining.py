"""Fresh retraining on split-isolated matched inputs; no precomputed model used.
All feature builders use only the final context. Calibration is independent of
training/validation/test. Timings are end-to-end serial wall times, not scaling.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from matched_features import single_features, split_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference/code'))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha_array(a):
    h = hashlib.sha256()
    for start in range(0, len(a), 128): h.update(np.asarray(a[start:start+128]).tobytes())
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def build(sim, bench, classical, clear_path, cfg, directory, split, h_mode, n, data_seed):
    directory.mkdir(parents=True, exist_ok=True)
    noises = cfg.train_noise_levels if split == 'train' else (cfg.val_noise_levels if split == 'validation' else cfg.test_noise_levels)
    count = n * len(cfg.generators) * len(noises)
    dims = {'single_raw': 1, 'single_derived': 6, 'multi_original': 6*len(cfg.resolutions)+len(cfg.resolutions)-1}
    arrays = {name: np.lib.format.open_memmap(directory / (name+'.npy'), mode='w+',
               dtype='float32', shape=(count, cfg.context_length, dim)) for name, dim in dims.items()}
    rows, k = [], 0
    seed = data_seed + split_seed(split, 0) + (10000000 if h_mode == 'constant' else 0)
    for gi, gen in enumerate(cfg.generators):
        for i in range(n):
            rng = np.random.default_rng(np.random.SeedSequence([seed, gi, i]))
            path = bench.generate_path(sim, gen, cfg, rng, h_mode)
            H = path.H_path
            # v2.13 trains on H_last, NOT the full-path or context mean.
            target = float(H[-1])
            try:
                for noise in noises:
                    levels = np.stack([np.log(sim.observation_rv(path, gen, cfg, M, noise))[-cfg.context_length:]
                                       for M in cfg.resolutions], axis=1).astype(np.float32)
                    single = levels[:, list(cfg.resolutions).index(max(cfg.resolutions))]
                    features = single_features(single)
                    arrays['single_raw'][k] = features['single_raw']
                    arrays['single_derived'][k] = features['single_derived']
                    arrays['multi_original'][k] = sim.robust_representation(levels, cfg)
                    row = dict(row=k, split=split, h_mode=h_mode, generator=gen,
                               noise=float(noise), path_id=f'{split}:{h_mode}:{gen}:{i}',
                               rng_seed=seed, rng_generator=gi, rng_path=i,
                               target=target, full_path_H_mean=float(np.mean(H)),
                               context_H_mean=float(np.mean(H[-cfg.context_length:])))
                    if split in ('calibration', 'test'):
                        row['variogram_raw'] = float(classical.variogram_h(single, cfg.benchmark_scales))
                        row['dfa_raw'] = float(classical.dfa_h(single, cfg.dfa_scales))
                    rows.append(row); k += 1
            finally:
                clear_path(path)
            if i % 100 == 0: print(split, h_mode, gen, i, '/', n, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / 'rows.csv', index=False)
    hashes = {}
    for name, array in arrays.items():
        array.flush()
        for start in range(0,len(array),128):
            if not np.isfinite(array[start:start+128]).all(): raise ValueError('Nonfinite features: '+name)
        hashes[name] = sha_array(array)
    if not np.isfinite(frame.target).all(): raise ValueError('Nonfinite targets')
    write_json(directory / 'manifest.json', dict(split=split, h_mode=h_mode,
        n_per_generator=n, n_rows=count, rng_seed=seed, dimensions=dims,
        content_sha256=hashes, target_sha256=sha_array(frame.target.to_numpy())))
    return arrays, frame


class Samples(torch.utils.data.Dataset):
    def __init__(self, x, y, mean, std): self.x, self.y, self.mean, self.std = x, y, mean, std
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        a = ((np.asarray(self.x[i], dtype=np.float32)-self.mean)/self.std).astype(np.float32)
        return torch.from_numpy(a), torch.tensor(self.y[i], dtype=torch.float32)


def scaler(x):
    total = np.zeros(x.shape[-1], np.float64); squares = total.copy(); n = 0
    for start in range(0, len(x), 128):
        a = np.asarray(x[start:start+128], dtype=np.float64)
        total += a.sum(axis=(0,1)); squares += (a*a).sum(axis=(0,1)); n += a.shape[0]*a.shape[1]
    mean = total/n; std = np.sqrt(np.maximum(squares/n - mean*mean, 0))
    std[std < 1e-6] = 1
    return mean.astype(np.float32), std.astype(np.float32)


def predict(model, data, device):
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in torch.utils.data.DataLoader(data, batch_size=16, shuffle=False):
            out.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(out).reshape(-1)


def metric(y, p):
    e = p-y
    return dict(n=len(y), bias=float(e.mean()), mae=float(abs(e).mean()), rmse=float(np.sqrt((e*e).mean())))


def evaluate(cal, test, methods, boot, seed):
    if boot < 1: raise ValueError('bootstrap count must be positive')
    test = test.reset_index(drop=True)
    score, comparisons, coefficients = [], [], []
    calibrated = {}
    keys = ['h_mode','generator','noise']
    for key, te in test.groupby(keys, sort=True):
        ca = cal
        for column, value in zip(keys, key): ca = ca.loc[ca[column] == value]
        y = te.target.to_numpy(); yc = ca.target.to_numpy()
        preds = {}
        for method in methods:
            raw_cal = ca[method].to_numpy(); raw_test = te[method].to_numpy()
            if not np.isfinite(raw_cal).all() or not np.isfinite(raw_test).all():
                raise ValueError('Nonfinite estimator output: '+method)
            # Linear calibration is fitted only on this held-out calibration cell.
            if len(yc) < 2 or not np.isfinite(yc).all() or not np.isfinite(y).all():
                raise ValueError('Invalid or insufficient calibration/test targets')
            if np.std(raw_cal) < 1e-10:
                a, b = 0.0, float(yc.mean())
            else:
                a, b = np.linalg.lstsq(np.column_stack([raw_cal, np.ones(len(raw_cal))]), yc, rcond=None)[0]
            for label, pred in [(method, raw_test), (method+'_cal', np.clip(a*raw_test+b, .05, .50))]:
                preds[label] = pred
                score.append(dict(zip(keys,key), method=label, **metric(y,pred)))
            coefficients.append(dict(zip(keys,key), method=method, a=float(a), b=float(b), n=len(ca)))
            col = method+'_cal'
            if col not in calibrated: calibrated[col] = np.empty(len(test))
            calibrated[col][te.index.to_numpy()] = preds[col]
        pairs = []
        for name in methods:
            if name.startswith(('single_', 'multi_')):
                pairs += [(name, 'variogram_raw_cal'), (name, 'dfa_raw_cal'),
                          (name+'_cal', 'variogram_raw_cal'), (name+'_cal', 'dfa_raw_cal')]
                pairs += [(name, base) for base in ('train_mean', 'calibration_mean') if base in preds]
        for name in methods:
            if name.startswith('multi_original_'):
                suffix = name.removeprefix('multi_original_')
                for rep in ['single_raw', 'single_derived']:
                    other = rep+'_'+suffix
                    if other in preds: pairs.append((name,other))
        # Each cell has one row per independent latent path. Noise cells are NOT pooled.
        assert te.path_id.is_unique
        rng = np.random.default_rng(seed)
        for left,right in pairs:
            e1 = (preds[left]-y)**2; e2 = (preds[right]-y)**2
            deltas = np.empty(boot)
            for i in range(boot):
                ids = rng.integers(0,len(y),len(y))
                deltas[i] = np.sqrt(e1[ids].mean())-np.sqrt(e2[ids].mean())
            comparisons.append(dict(zip(keys,key), left=left, right=right, n=len(y),
                delta_rmse=float(np.sqrt(e1.mean())-np.sqrt(e2.mean())),
                ci_low=float(np.quantile(deltas,.025)), ci_high=float(np.quantile(deltas,.975)),
                bootstrap=boot, interpretation='negative favors left; descriptive unadjusted CI'))
    for col, values in calibrated.items(): test[col] = values
    return pd.DataFrame(score),pd.DataFrame(comparisons),pd.DataFrame(coefficients),test


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',choices=['smoke','full'],required=True)
    p.add_argument('--output',required=True); p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=1)
    args = p.parse_args()
    if args.threads < 1: p.error('threads must be positive')
    if args.device not in ('cpu','cuda'): p.error('device must be cpu or cuda')
    if args.device == 'cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable; no silent fallback')
    out = Path(args.output); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads)
    sim = load('matched_reference_sim',ROOT/'reference/code/HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix.py')
    bench = load('matched_reference_bench',ROOT/'reference/code/HurstNeuralEstimator_v2_22_synthetic_classical_benchmark.py')
    classical = load('matched_classical',ROOT/'reference/code/HurstNeuralEstimator_v2_15_empirical_crypto_multiasset.py')
    sim.np = bench._NumpyFastConvolveProxy(np)
    _, clear_path = bench.install_observation_cache(sim)
    cfg = sim.Config(); cfg.device=args.device
    cfg.context_length=384; cfg.context_lengths=(384,)
    cfg.generators=(sim.GEN_BSS,sim.GEN_RBERGOMI)
    cfg.benchmark_scales=(1,2,4,8,16,32,64); cfg.dfa_scales=(8,16,32,64)
    cfg.batch_size=16
    cfg.epochs=2 if args.mode=='smoke' else 40
    cfg.patience=2 if args.mode=='smoke' else 8
    cfg.n_train=8 if args.mode=='smoke' else 4000
    cfg.n_val=4 if args.mode=='smoke' else 1000
    ncal=6 if args.mode=='smoke' else 400
    ntest=8 if args.mode=='smoke' else 1500
    if args.mode=='smoke':
        cfg.train_noise_levels=(0.0,0.1); cfg.val_noise_levels=(0.0,0.1); cfg.test_noise_levels=(0.0,0.1)
    seeds=[42] if args.mode=='smoke' else [42,123,2024]
    cfg.out_dir=str(out)
    write_json(out/'config.json',dict(mode=args.mode, model_seeds=seeds,data_seed=190317,
        n_calibration_per_generator=ncal,n_test_per_generator=ntest, config=cfg.__dict__,
        representation_definitions={'single_raw':'final 384 points of finest-resolution (M=64) logRV channel only',
          'single_derived':'six features recomputed solely from the same finest-channel window',
          'multi_original':'original multi-resolution multichannel final-window features'},
        calibration='independent, per H-mode/generator/noise cell, affine then clip to [0.05,0.50]',
        simulation_backend='v2.22 FFT convolution proxy and float32 observation cache',
        cache_dtype='float32 (reference default was float16); no claim of exact training replication',
        capacity='same hidden architecture, differing input-layer parameter counts; not capacity-matched',
        calibration_caveat='H-mode and generator labels condition calibration; simulation-only, not deployable without labels',
        scope='fresh stochastic-H training; independent stochastic-H and constant-H evaluation; fixed datasets across model seeds'))
    stages=[]; start=time.perf_counter()
    train, tr = build(sim,bench,classical,clear_path,cfg,out/'cache/train','train','stochastic',cfg.n_train,190317)
    val, va = build(sim,bench,classical,clear_path,cfg,out/'cache/validation','validation','stochastic',cfg.n_val,190317)
    sets={}
    for split,n in [('calibration',ncal),('test',ntest)]:
        for mode in ['stochastic','constant']:
            sets[(split,mode)] = build(sim,bench,classical,clear_path,cfg,out/'cache'/f'{split}_{mode}',split,mode,n,190317)
    stages.append(dict(stage='data_generation_all_representations',seconds=time.perf_counter()-start))
    ids = [set(tr.path_id),set(va.path_id)] + [set(frame.path_id) for _,frame in sets.values()]
    assert all(not ids[i].intersection(ids[j]) for i in range(len(ids)) for j in range(i))
    # RNG-key separation is checked independently of human-readable split labels.
    frames=[tr,va]+[frame for _,frame in sets.values()]
    rng_keys=[set(map(tuple,f[['rng_seed','rng_generator','rng_path']].drop_duplicates().to_numpy())) for f in frames]
    assert all(not rng_keys[i].intersection(rng_keys[j]) for i in range(len(rng_keys)) for j in range(i))
    results={key:frame.copy() for key,(_,frame) in sets.items()}
    methods=['variogram_raw','dfa_raw']
    for rep in train:
        mean,std=scaler(train[rep]); np.savez(out/(rep+'_scaler.npz'),mean=mean,std=std)
        for seed in seeds:
            start=time.perf_counter(); sim.set_seed(seed)
            checkpoint=out/'models'/rep/f'seed_{seed}'; checkpoint.mkdir(parents=True)
            model, history=sim.train_model_streaming(train[rep],tr.target.to_numpy(),val[rep],va.target.to_numpy(),
                mean,std,cfg,torch.device(args.device),seed,cfg.context_length)
            if not np.isfinite(history[['train_mse','val_mse']].to_numpy()).all():
                raise ValueError('Nonfinite training/validation loss')
            history.to_csv(checkpoint/'training_history.csv',index=False)
            write_json(checkpoint/'architecture.json', dict(input_channels=int(train[rep].shape[-1]),
                parameter_count=sum(p.numel() for p in model.parameters())))
            torch.save(dict(state_dict=model.state_dict(),config=cfg.__dict__,representation=rep,seed=seed),checkpoint/'model.pt')
            stages.append(dict(stage='training',representation=rep,seed=seed,seconds=time.perf_counter()-start))
            method=f'{rep}_seed{seed}'; methods.append(method)
            start=time.perf_counter()
            for key,(arrays,frame) in sets.items():
                results[key][method]=predict(model,Samples(arrays[rep],frame.target.to_numpy(),mean,std),args.device)
            stages.append(dict(stage='prediction_calibration_and_test',representation=rep,seed=seed,seconds=time.perf_counter()-start))
        ensemble=rep+'_ensemble'; methods.append(ensemble)
        for frame in results.values(): frame[ensemble]=frame[[f'{rep}_seed{s}' for s in seeds]].mean(axis=1)
    cal=pd.concat([results[('calibration',m)] for m in ['stochastic','constant']],ignore_index=True)
    test=pd.concat([results[('test',m)] for m in ['stochastic','constant']],ignore_index=True)
    cal['train_mean']=float(tr.target.mean()); test['train_mean']=float(tr.target.mean()); methods.append('train_mean')
    for _, group in cal.groupby(['h_mode','generator','noise']):
        mask=np.ones(len(test),bool)
        for key in ['h_mode','generator','noise']: mask &= test[key].to_numpy()==group[key].iloc[0]
        test.loc[mask,'calibration_mean']=float(group.target.mean())
        cal.loc[group.index,'calibration_mean']=float(group.target.mean())
    methods.append('calibration_mean')
    scores,pairs,coef,test=evaluate(cal,test,methods,100 if args.mode=='smoke' else 2000,98301)
    cal.to_csv(out/'calibration_predictions.csv',index=False); test.to_csv(out/'test_predictions.csv',index=False)
    scores.to_csv(out/'metrics.csv',index=False); pairs.to_csv(out/'paired_bootstrap.csv',index=False)
    coef.to_csv(out/'calibration_coefficients.csv',index=False); pd.DataFrame(stages).to_csv(out/'stage_wall_times.csv',index=False)
    # Do not duplicate large feature caches into the return ZIP. Keep metadata and rows.
    for array in train.values(): array.flush()
    write_json(out/'completed.json',dict(success=True,mode=args.mode,split_rng_disjoint=True,
        warnings=['smoke results are not scientific evidence'] if args.mode=='smoke' else [],
        training_seeds_share_data=True,bootstrap_unit='latent path within one generator/noise/H-mode cell'))

if __name__=='__main__': main()
