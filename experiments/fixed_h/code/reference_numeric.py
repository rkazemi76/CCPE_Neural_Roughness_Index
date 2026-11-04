"""Load explicitly whitelisted non-neural definitions, unmodified, via AST.
No fake torch module; neural imports/operations are never emulated.
"""
import ast
import dataclasses
import math
import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'reference/code'
SIM = 'HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix.py'
BENCH = 'HurstNeuralEstimator_v2_22_synthetic_classical_benchmark.py'
CLASSICAL = 'HurstNeuralEstimator_v2_15_empirical_crypto_multiasset.py'


def selected(filename, names, module_name, extra_globals=None):
    tree = ast.parse((REF / filename).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Missing reference definitions')
    module = types.ModuleType(module_name)
    module.__dict__.update(np=np, pd=pd, math=math, dataclass=dataclasses.dataclass,
                           fftconvolve=fftconvolve, GEN_BSS='bss', GEN_RBERGOMI='rbergomi')
    module.__dict__.update(extra_globals or {})
    sys.modules[module_name] = module
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    exec(compile(code, str(REF / filename), 'exec'), module.__dict__)
    return module


def load_numeric():
    sim = selected(SIM, ['Config', 'BSSPath', 'RBergomiPath', 'generate_stochastic_H_path',
        'bss_kernel', 'bss_factor_from_H', 'generate_bss_path', 'make_bss_counterfactual',
        'rb_factor_from_H', 'generate_rbergomi_path', 'make_rb_counterfactual',
        'fine_returns', 'aggregate', 'observation_rv', 'causal_ma', 'diff',
        'robust_representation'], 'compute_numeric_sim')
    bench = selected(BENCH, ['_NumpyFastConvolveProxy', 'install_observation_cache',
                           'generate_path'], 'compute_numeric_bench')
    classical = selected(CLASSICAL, ['variogram_h', 'dfa_h'], 'compute_numeric_classical')
    sim.np = bench._NumpyFastConvolveProxy(np)
    _, clear = bench.install_observation_cache(sim)
    return sim, bench, classical, clear


def load_evaluation():
    return selected('matched_retraining.py', ['metric', 'evaluate'], 'compute_numeric_evaluation')
