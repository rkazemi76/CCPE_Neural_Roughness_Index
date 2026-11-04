"""CPU-only stage profile; raw per-path timings, no calibration or training."""
import argparse,time,json,sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'reference/code'))
import numpy as np
import pandas as pd
import torch
import HurstNeuralEstimator_v2_22_synthetic_classical_benchmark as b

def main():
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=32);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--out',required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 if a.n < 1 or a.repetitions < 1: p.error('n and repetitions must be positive')
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 sim=b.load_module(root/'reference/code/HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix.py','profile_sim')
 cls=b.load_module(root/'reference/code/HurstNeuralEstimator_v2_15_empirical_crypto_multiasset.py','profile_cls')
 cfg=sim.Config(seq_len=768,h_block=32,h_phi=.97,h_latent_sd=2.,bss_kernel_lambda=.08,bss_kernel_size=512,bss_burn_in=512,bss_xi=.04,bss_eta=1.,rb_xi0=.04,rb_eta=1.,rb_rho=-.70,T=3.,resolutions=(8,16,32,64),context_lengths=(384,))
 cfg.benchmark_scales=(1,2,4,8,16,32,64);cfg.dfa_scales=(8,16,32,64)
 model=sim.HurstLSTM(input_size=27,cfg=cfg)
 state=torch.load(root/'reference/checkpoints/seed_42/model_state_W384.pt',map_location='cpu',weights_only=True)
 model.load_state_dict(state.get('state_dict',state));model.eval()
 with np.load(root/'reference/checkpoints/seed_42/standardizer_W384.npz') as s: mean=s['mean'].reshape(1,-1);std=s['std'].reshape(1,-1)
 rows=[]
 for gen in ('bss','rbergomi'):
  for rep in range(a.repetitions):
   for i in range(-2,a.n):
    seed=920000+i;rng=np.random.default_rng(seed)
    t=time.perf_counter();path=b.generate_path(sim,gen,cfg,rng);generation=time.perf_counter()-t
    # No observation cache: each method pays its own preprocessing cost.
    methods=['neural','variogram','dfa'];methods=methods[rep%3:]+methods[:rep%3]
    for method in methods:
     t=time.perf_counter()
     if method=='neural':
      feat=sim.make_representation_context(path,gen,cfg,.05,384)
      feat=(np.asarray(feat,np.float32)-mean)/np.maximum(std,1e-8)
      tensor=torch.from_numpy(feat[None].astype(np.float32));prep=time.perf_counter()-t
      with torch.inference_mode(): pred=float(model(tensor).numpy().reshape(-1)[0])
     else:
      x=np.log(sim.observation_rv(path,gen,cfg,max(cfg.resolutions),.05))[-384:];prep=time.perf_counter()-t
      pred=float(cls.variogram_h(x,cfg.benchmark_scales) if method=='variogram' else cls.dfa_h(x,cfg.dfa_scales))
     total=time.perf_counter()-t
     if not np.isfinite(pred):raise ValueError('Non-finite prediction')
     if i>=0:rows.append(dict(generator=gen,repetition=rep,path_id=i,seed=seed,method=method,H_last=float(path.H_path[-1]),prediction=pred,generation_seconds=generation,preprocess_seconds=prep,estimator_seconds=total-prep,total_seconds=total))
 pd.DataFrame(rows).to_csv(out/'stage_profile_raw.csv',index=False)
 (out/'scope.json').write_text(json.dumps({'device':'cpu','threads':1,'noise':.05,'H_mode':'stochastic','warmup_paths_per_rep':2,'simulation_backend':'original numpy convolution; no observation cache','scope':'Single-path latency; frozen neural versus RAW classical. Not calibrated accuracy comparison, batching, training-cost or channel-ablation experiment.'},indent=2))
if __name__=='__main__':main()
