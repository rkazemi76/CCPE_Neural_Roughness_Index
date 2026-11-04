# Paired CPU feature-generation benchmark

This experiment measures synthetic BSS/rough-Bergomi path generation, 27-channel feature construction, and return/transfer overhead on the author's local Windows Intel Core i5-11400 desktop (6 physical cores, 12 logical processors, 16 GB RAM, Intel UHD Graphics 730). The GPU was not used.

## Archived reported design

- 2, 4, and 6 worker processes;
- strong scaling: 384 total paths in both serial and parallel execution;
- weak scaling: 64 paths per worker;
- sequence length 384, 27 feature channels, observation noise 0.05;
- nine paired replications for each generator/design/worker configuration;
- 108 paired records total, with a maximum serial-versus-parallel feature-array absolute difference of zero at stored float32 precision.

The final evidence is in `evidence/workers_2/`, `workers_4/`, `workers_6/`, with combined CSVs under `evidence/combined/`. The protocol describes pairing order, warm-up handling, and timing scope.

## Reproduction

From this directory, install `code/requirements.txt`, then run (PowerShell or Bash):

```bash
python code/paired_controlled_benchmark.py --simulator code/HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix.py --out-dir results/paired_w2 --strong-paths 384 --weak-paths-per-worker 64 --workers 2 --repetitions 9 --noise 0.05 --seq-len 384 --affinity-count 0 --generators bss rbergomi
python code/paired_controlled_benchmark.py --simulator code/HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix.py --out-dir results/paired_w4 --strong-paths 384 --weak-paths-per-worker 64 --workers 4 --repetitions 9 --noise 0.05 --seq-len 384 --affinity-count 0 --generators bss rbergomi
python code/paired_controlled_benchmark.py --simulator code/HurstNeuralEstimator_v2_13_effective_memory_saturation_memoryfix.py --out-dir results/paired_w6 --strong-paths 384 --weak-paths-per-worker 64 --workers 6 --repetitions 9 --noise 0.05 --seq-len 384 --affinity-count 0 --generators bss rbergomi
```

Use fresh output directories; these commands rerun timings and do not reproduce the exact wall-clock numbers on different hardware.

## Interpretation limits

The measurements do not include neural model training or inference, CUDA/GPU use, minute-data ingestion, complete pipeline runtime, multi-node scaling, or production throughput. The Windows monitoring environment returned PSS as zero; PSS is therefore not a usable memory measurement in these files and must not be interpreted as zero memory use. Process-tree RSS sums may double-count shared memory and are retained only as auxiliary diagnostics. Weak-scaling efficiency declines at higher worker counts; strong scaling is sublinear.
