#!/usr/bin/env python3
"""Paired, controlled serial-versus-parallel benchmark for one worker-count setting for CCPE feature preparation.

Within each generator/design, serial and parallel batches are measured in
balanced pseudorandom order on identical task definitions. A persistent process pool is
started and warmed before timed repetitions; its start-up cost is reported
separately. Strong scaling fixes total paths. Weak scaling fixes paths per
worker. PSS (proportional set size) is sampled where the operating system supports it;
Windows may return zero/unavailable PSS, so process-tree RSS is retained only
as an aggregate diagnostic. Only path generation and 27-channel feature
construction are timed; no neural training/inference is included.
"""
from __future__ import annotations
import os
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_n] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse, hashlib, importlib.util, json, multiprocessing as mp, platform, random, statistics, sys, threading, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import psutil

_SIM = None

def load_simulator(path: Path):
    spec = importlib.util.spec_from_file_location("ccpe_paired_simulator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import simulator: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def init_worker(path: str):
    global _SIM
    _SIM = load_simulator(Path(path))

def worker_task(task):
    gen, seed, cfg_kwargs, noise = task
    cfg = _SIM.Config(**cfg_kwargs)
    rng = np.random.default_rng(int(seed))
    path = _SIM.generate_bss_path(cfg, rng) if gen == "bss" else _SIM.generate_rbergomi_path(cfg, rng)
    feat = np.asarray(_SIM.make_representation(path, gen, cfg, float(noise)), dtype=np.float32)
    return int(seed), feat

def make_tasks(sim, gen, seeds, noise, seq_len):
    cfg = sim.Config(seq_len=seq_len)
    keep = {"seq_len", "h_min", "h_max", "h_block", "h_phi", "h_latent_sd", "bss_kernel_lambda", "bss_kernel_size", "bss_burn_in", "bss_xi", "bss_eta", "rb_xi0", "rb_eta", "rb_rho", "T", "return_scale", "resolutions", "temporal_lag", "roughness_lag", "smooth_window"}
    kw = {k: getattr(cfg, k) for k in sim.Config.__dataclass_fields__ if k in keep}
    return [(gen, int(seed), kw, noise) for seed in seeds]

def serial_batch(sim, tasks):
    global _SIM
    old = _SIM; _SIM = sim
    try: return [worker_task(t) for t in tasks]
    finally: _SIM = old

def max_diff(a, b):
    aa, bb = sorted(a, key=lambda x: x[0]), sorted(b, key=lambda x: x[0])
    if len(aa) != len(bb): raise RuntimeError("Output length mismatch")
    d = 0.0
    for (sa, xa), (sb, xb) in zip(aa, bb):
        if sa != sb or xa.shape != xb.shape: raise RuntimeError("Output key/shape mismatch")
        d = max(d, float(np.max(np.abs(xa - xb))))
    return d

class MemMonitor:
    def __init__(self, interval=0.05):
        self.p = psutil.Process(os.getpid()); self.interval=interval; self.stop=threading.Event(); self.th=None
        self.peak_rss=0; self.peak_pss=0; self.peak_child_pss=0; self.peak_child_rss=0; self.peak_one_child_pss=0
    def sample(self):
        while not self.stop.is_set():
            try:
                procs=[self.p]+self.p.children(recursive=True); rss=[]; pss=[]
                for pr in procs:
                    try:
                        if pr.is_running():
                            mi=pr.memory_info(); rss.append(mi.rss)
                            try:
                                value = pr.memory_full_info().pss
                                pss.append(value if value and value > 0 else None)
                            except Exception: pss.append(None)
                    except Exception: pass
                if rss:
                    self.peak_rss=max(self.peak_rss,sum(rss)); self.peak_child_rss=max(self.peak_child_rss,sum(rss[1:]))
                valid=[x for x in pss if x is not None]
                child=[x for x in pss[1:] if x is not None]
                if valid: self.peak_pss=max(self.peak_pss,sum(valid))
                if child:
                    self.peak_child_pss=max(self.peak_child_pss,sum(child)); self.peak_one_child_pss=max(self.peak_one_child_pss,max(child))
            except Exception: pass
            self.stop.wait(self.interval)
    def __enter__(self): self.th=threading.Thread(target=self.sample,daemon=True); self.th.start(); return self
    def __exit__(self,*a): self.stop.set(); self.th.join(timeout=2)

def read_cpu_stat():
    try:
        vals={}
        for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines():
            k,v=line.split(); vals[k]=int(v)
        return vals
    except Exception: return {}

def main():
    here=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--simulator',default=str(here/'code'/'HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix.py'))
    ap.add_argument('--out-dir',default=str(here/'results'/'paired_controlled'))
    ap.add_argument('--strong-paths',type=int,default=384)
    ap.add_argument('--weak-paths-per-worker',type=int,default=64)
    ap.add_argument('--repetitions',type=int,default=9)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--noise',type=float,default=0.05)
    ap.add_argument('--base-seed',type=int,default=925000)
    ap.add_argument('--seq-len',type=int,default=384)
    ap.add_argument('--affinity-count',type=int,default=0)
    ap.add_argument('--generators',nargs='+',choices=['bss','rbergomi'],default=['bss','rbergomi'])
    a=ap.parse_args()
    if a.repetitions<5 or a.strong_paths<1 or a.weak_paths_per_worker<1 or a.workers<2: ap.error('Use >=5 repetitions, positive path counts, and >=2 workers')
    affinity=None
    if a.affinity_count>0:
        try:
            avail=sorted(os.sched_getaffinity(0)); affinity=avail[:min(a.affinity_count,len(avail))]; os.sched_setaffinity(0,set(affinity))
        except Exception:
            try:
                p=psutil.Process(); avail=p.cpu_affinity(); affinity=avail[:min(a.affinity_count,len(avail))]; p.cpu_affinity(affinity)
            except Exception: affinity=None
    sim_path=Path(a.simulator).resolve(); sim=load_simulator(sim_path)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    seq_len=a.seq_len; outrows=[]; rawpairs=[]
    for gi,gen in enumerate(a.generators):
        for design in ('strong','weak'):
            n=a.strong_paths if design=='strong' else a.weak_paths_per_worker
            seeds=[a.base_seed+gi*1000000+i for i in range(n)]
            tasks=make_tasks(sim,gen,seeds,a.noise,seq_len)
            p_tasks=tasks if design=='strong' else make_tasks(sim,gen,[a.base_seed+gi*1000000+i for i in range(n*a.workers)],a.noise,seq_len)
            # Reference features are produced outside the timing protocol.
            ref=serial_batch(sim,p_tasks)
            serial_batch(sim,tasks)  # warm-up
            t0=time.perf_counter()
            executor=ProcessPoolExecutor(max_workers=a.workers,initializer=init_worker,initargs=(str(sim_path),))
            try:
                # Warm every child, then warm the complete parallel workload.
                list(executor.map(worker_task,p_tasks[:a.workers*2],chunksize=1))
                startup=time.perf_counter()-t0
                warm_t0=time.perf_counter()
                list(executor.map(worker_task,p_tasks,chunksize=2))
                full_warmup_seconds=time.perf_counter()-warm_t0
                serial_times=[]; parallel_times=[]; ratios=[]; diffs=[]
                order_rng=random.Random(20261104+gi*10+(0 if design=='strong' else 1))
                order=[0,1]*((a.repetitions+1)//2); order=order[:a.repetitions]; order_rng.shuffle(order)
                mem=MemMonitor()
                with mem:
                    for rep,first in enumerate(order):
                        results={}
                        for which in (first,1-first):
                            if which==0:
                                t=time.perf_counter(); result=serial_batch(sim,tasks); elapsed=time.perf_counter()-t
                                results['serial']=(elapsed,result)
                            else:
                                t=time.perf_counter(); result=list(executor.map(worker_task,p_tasks,chunksize=2)); elapsed=time.perf_counter()-t
                                results['parallel']=(elapsed,result)
                        st,sout=results['serial']; pt,pout=results['parallel']
                        serial_times.append(st); parallel_times.append(pt); ratios.append(st/pt); diffs.append(max_diff(ref,pout))
                        rawpairs.append({'generator':gen,'design':design,'repetition':rep+1,'order':'serial-first' if first==0 else 'parallel-first','serial_seconds':st,'parallel_seconds':pt,'paired_ratio':st/pt,'max_abs_feature_diff':diffs[-1]})
                med_s=float(statistics.median(serial_times)); med_p=float(statistics.median(parallel_times)); med_ratio=float(statistics.median(ratios))
                q1,q3=np.quantile(ratios,[.25,.75])
                outrows.append({
                    'generator':gen,'scaling_design':design,'workers':a.workers,
                    'serial_paths':len(tasks),'parallel_paths':len(p_tasks),
                    'paths_per_worker_target':a.strong_paths/a.workers if design=='strong' else a.weak_paths_per_worker,
                    'sequence_length':seq_len,'feature_channels':int(ref[0][1].shape[1]),'noise_level':a.noise,
                    'median_serial_seconds':med_s,'median_parallel_seconds':med_p,
                    'median_paired_speedup_or_weak_efficiency':med_ratio,
                    'q1_paired_ratio':float(q1),'q3_paired_ratio':float(q3),
                    'serial_paths_per_second':len(tasks)/med_s,'parallel_paths_per_second':len(p_tasks)/med_p,
                    'pool_startup_seconds':startup,'full_batch_warmup_seconds':full_warmup_seconds,
                    'peak_process_tree_pss_mib':mem.peak_pss/1024**2,
                    'peak_child_pss_mib':mem.peak_child_pss/1024**2,
                    'peak_single_worker_pss_mib':mem.peak_one_child_pss/1024**2,
                    'peak_process_tree_rss_mib':mem.peak_rss/1024**2,
                    'max_abs_feature_diff_vs_serial':max(diffs),
                    'repetitions':a.repetitions,'timing':'paired balanced-random order; persistent warmed pool; startup excluded and reported separately'
                })
                print(f'{gen:8s} {design:6s} serial_n={len(tasks):2d} parallel_n={len(p_tasks):2d} serial={med_s:.4f}s parallel={med_p:.4f}s ratio={med_ratio:.3f} [{q1:.3f},{q3:.3f}] PSS={mem.peak_pss/1024**2:.1f}MiB diff={max(diffs):.3g}',flush=True)
            finally:
                executor.shutdown(wait=True,cancel_futures=True)
    df=pd.DataFrame(outrows); df.to_csv(out/'paired_controlled_benchmark_summary.csv',index=False,float_format='%.8g')
    pd.DataFrame(rawpairs).to_csv(out/'paired_controlled_benchmark_repetitions.csv',index=False,float_format='%.8g')
    try:
        sim_hash=hashlib.sha256(sim_path.read_bytes()).hexdigest()
    except Exception: sim_hash=None
    cpu={}
    try:
        cpu['cpu_affinity_ids']=sorted(os.sched_getaffinity(0))
        cpu['cpu_affinity_count']=len(cpu['cpu_affinity_ids'])
    except Exception: pass
    rawcpu=read_cpu_stat()
    env={
      'platform':platform.platform(),'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,'psutil':psutil.__version__,
      'logical_cpus_reported':psutil.cpu_count(logical=True),'physical_cpus_reported':psutil.cpu_count(logical=False),**cpu,
      'cgroup_cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None,
      'cgroup_memory_max_bytes':Path('/sys/fs/cgroup/memory.max').read_text().strip() if Path('/sys/fs/cgroup/memory.max').exists() else None,
      'cgroup_cpu_stat_at_end':rawcpu,'process_start_method':mp.get_start_method(allow_none=True) or 'default',
      'simulator_filename':sim_path.name,'simulator_sha256':sim_hash,'sequence_length':seq_len,'feature_channels':27,
      'noise_level':a.noise,'strong_total_paths':a.strong_paths,'weak_paths_per_worker':a.weak_paths_per_worker,
      'workers':a.workers,'repetitions':a.repetitions,'pair_order':'balanced pseudorandom serial-first/parallel-first order with a fixed seed',
      'pool_persistent_during_repetitions':True,'pool_startup_excluded_from_steady_state_timings':True,'one_full_batch_warmup_excluded_per_configuration':True,
      'memory_metric':'PSS where supported; zero/unavailable PSS is not interpreted; process-tree RSS is an aggregate parent/child sum and may include shared pages more than once',
      'memory_sample_interval_seconds':0.05,'numerical_library_threads_per_process':1,
      'timed_scope':'synthetic path generation + 27-channel feature construction + result transfer',
      'neural_training_or_inference_included':False
    }
    (out/'paired_controlled_benchmark_environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    print('\nSUMMARY\n'+df.round(4).to_string(index=False))
    print('\nWrote',out)

if __name__=='__main__': main()
