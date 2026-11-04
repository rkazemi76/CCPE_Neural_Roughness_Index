# Out-of-domain smooth semimartingale control

This folder contains the frozen-W384-checkpoint Ornstein–Uhlenbeck (OU) semimartingale stress diagnostic, its output predictions and summaries, and the code needed to rerun it.

## Reproduction

From this directory:

```bash
python -m pip install -r code/requirements-smooth-control.txt
python code/run_smooth_semimartingale_control.py --out evidence/smooth_semimartingale_reproduced
```

The archived evidence is in `evidence/smooth_semimartingale/`. The reported design uses 200 paths per mean-reversion parameter, kappas 0.5 and 3.0, noise amplitudes 0, 0.05, and 0.10, and checkpoints with seeds 42, 123, and 2024.

## Interpretation

The OU process is a smooth semimartingale and is out-of-domain relative to rough-process training. The Brownian regularity boundary H=0.5 is a conceptual reference, not a supervised target observed for these paths. The frozen neural predictions are diagnostic outputs, not calibrated smooth-versus-rough classification, not a false-positive rate, and not a test that the neural output should equal 0.5.
