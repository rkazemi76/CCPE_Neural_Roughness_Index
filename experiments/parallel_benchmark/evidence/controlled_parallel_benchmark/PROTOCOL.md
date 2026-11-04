# Controlled paired CPU benchmark: protocol and interpretation (i5-11400)

## Purpose and scope

The benchmark isolates process-level CPU concurrency for independent synthetic path generation and construction of the 27-channel, four-resolution log-realized-variance representation. It does not include neural-network training or inference, GPU execution, raw-market ingestion, or the complete downstream pipeline.

## Hardware and execution environment

The runs were performed locally on a Windows desktop with an Intel Core i5-11400 (6 physical cores, 12 logical processors), 16 GB RAM, and Intel UHD Graphics 730. The recorded environment reports Windows build 10.0.22631, Python 3.11.7, NumPy 2.3.5, pandas 2.2.3, and psutil 7.2.2. The multiprocessing start method is `spawn`. No CUDA/GPU work is part of this benchmark. Numerical-library threads are limited to one per process. The three environment manifests record the worker count and simulator SHA-256.

## Experimental design

- Generators: BSS-style and finite-grid rough-Bergomi-style.
- Sequence length: 384 observations.
- Feature representation: 27 channels from four realized-variance resolutions.
- Observation-noise amplitude: 0.05.
- Worker-process counts: 2, 4, and 6.
- Strong scaling: 384 total paths for the serial and parallel batches at every worker count.
- Weak scaling: 64 paths per worker; the serial reference has 64 paths, while parallel batches have 128, 256, and 384 paths for 2, 4, and 6 workers, respectively.
- Repetitions: nine paired repetitions for every generator, design, and worker count (12 conditions per repetition set; 108 paired records total).
- Pair order: serial-first versus parallel-first order is balanced pseudorandomly with a fixed seed.
- Pool handling: a persistent process pool is initialized and warmed before timed repetitions and remains alive across the paired measurements. One full-batch warm-up is also excluded from steady-state timing.
- Startup: pool initialization/worker warm-up and full-batch warm-up are recorded separately; they are excluded from steady-state batch times.
- Correctness: outputs are compared elementwise with the serial reference, keyed by deterministic path seed.

## Timing definitions

For strong scaling, each pair compares the same 384 paths. The paired speedup is the serial batch time divided by the parallel batch time. The reported value is the median of the nine paired speedup ratios; the bracketed range is the interquartile range (IQR) of those ratios.

For weak scaling, the serial batch contains 64 paths and the parallel batch contains 64 paths per worker. The paired weak-scaling efficiency is the serial 64-path time divided by the parallel batch time. Its reported value is the median of the nine paired ratios, with IQR in brackets. Therefore, values below one indicate that the expanded parallel workload takes longer than the fixed 64-path serial reference.

The IQRs describe run-to-run timing variation. They are not statistical confidence intervals and do not quantify uncertainty over a population of machines or workloads.

## Startup and memory

Across the 12 worker/generator/design configurations, process-pool startup plus worker warm-up ranged from 2.47 to 2.82 seconds, and full-batch warm-up ranged from 0.203 to 0.660 seconds. Both were excluded from steady-state timing. This is a material one-off cost: the reported ratios describe repeated batches with a persistent, warmed pool, not a single task that must initialize a new pool.

PSS values were returned as zero by the Windows monitoring environment and are therefore not interpreted or reported. Aggregate process-tree RSS was recorded in the CSVs as an auxiliary diagnostic; because it is the sum of parent/child RSS, it is not treated as unique physical-memory use.

## Results and interpretation

Strong-scaling speedups rise with worker count, reaching medians of 4.87 for BSS-style paths and 4.67 for rBergomi-style paths at six workers. Scaling is sublinear. Weak-scaling efficiency declines at higher worker counts, reaching 0.74 and 0.77 at six workers. All 108 serial/parallel feature-array comparisons have zero maximum absolute difference at the stored float32 precision.

The results support a bounded claim about task-level CPU feature preparation on this one desktop and workload. They do not establish general scalability, production throughput, end-to-end pipeline acceleration, distributed neural training, or multi-node execution.
