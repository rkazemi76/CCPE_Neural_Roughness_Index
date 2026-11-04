# Out-of-domain smooth semimartingale control

This folder contains the predictions, summary statistics, protocol, and feature-parity check for the Ornstein-Uhlenbeck (OU) semimartingale control reported in Table 8 of the manuscript.

## Design

- Frozen W384 checkpoints: training seeds 42, 123, and 2024.
- OU log-variance factor with stationary standard deviation 1 and exact coarse-grid transitions.
- Mean-reversion speeds: 0.5 and 3.0.
- Observation-noise amplitudes: 0, 0.05, and 0.10.
- 200 independently simulated paths per mean-reversion speed; each path is reused across the three noise amplitudes.
- 64 fine Gaussian returns per coarse observation; 768 coarse observations over T=3.
- No retraining and no calibration against smooth-process labels.

The Brownian semimartingale regularity boundary H=0.5 is a conceptual reference only. Since the network is trained on rough-path targets and its output is constrained to [0.05, 0.50], these outputs are not a calibrated smooth-versus-rough test or a false-positive-rate estimate.

`feature_parity.json` reports a maximum absolute discrepancy of 0.0 (float32 output) between the vectorized feature builder used by the experiment and the reference feature routine for the parity-check path.

## Reproduction

From the `code` directory, install `requirements-smooth-control.txt` and run:

```bash
python run_smooth_semimartingale_control.py
```

The default command recreates 200 paths per kappa, for kappas 0.5 and 3.0 and noise amplitudes 0, 0.05, and 0.10. Results are written to `evidence/smooth_semimartingale_reproduced/` relative to the package root. Optional arguments `--n-paths`, `--kappas`, `--noise-levels`, `--seed`, and `--out` can be used for additional sensitivity runs; such runs are not the reported experiment unless they use the protocol above.
