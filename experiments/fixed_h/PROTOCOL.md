# Fixed-H volatility sensitivity: prospective protocol v1

This is a NEW controlled synthetic diagnostic, not historical reproduction,
retraining, or validation of unknown market H. Manuscripts are not modified.
This protocol is specified before any new neural predictions are obtained.

## Design
- Reference discrete BSS and rBergomi-style generators; nominal constant
  H in {0.10, 0.25, 0.40}; sequence 768, final context 384, resolutions 8/16/32/64.
- Full: 200 independent innovation bundles per generator/H (1,200 bundles).
  Smoke: 3 per cell (18 bundles); smoke is an engineering test, not evidence.
- New root seed 48261917; SeedSequence([root, generator_index, H_index, path_index]).
  Smoke/full intentionally share their first three bundles. Never pool them.
  Distinct from documented prior root seeds, but not proof of training exclusion.
- Within a bundle, H and all volatility, return and observation-noise innovations
  are fixed. Six conditions: baseline; variance level x0.25; variance level x4;
  eta=0.5; eta=1.5; smooth log-variance shift log(4)*sin(2*pi*t/768).
  Level multipliers correspond to volatility multipliers 0.5 and 2.
- Eta conditions retain each reference generator's compensator. These alter
  vol-of-vol, not just realized variance level. The smooth multiplier alters
  the finite-scale pattern, without introducing a jump or changing nominal H.
- Retain the reference log-variance clipping [-15,8] and record saturation counts.
  Clipping, finite sampling, and BSS path normalization limit interpretation of H;
  this is NOT an assertion of exact continuous-time sample-path regularity.
- Noise 0 and 0.1 separately, using the original observation construction.
  Noise amplitude is absolute and NOT rescaled with volatility: noisy level
  contrasts combine level and signal-to-noise changes. The reference driver is
  shared across fine subintervals of each period, not independent microstructure noise.

## Estimands (no recalibration, tuning, or retraining)
Evaluate retained single_raw, single_derived, multi_original checkpoints,
seeds 42/123/2024, using their original frozen training scalers. Report each seed
and the arithmetic ensemble. Report uncalibrated variogram and DFA separately.
For every generator/H/noise/method: mean prediction, bias, MAE, RMSE by condition;
paired change from baseline and mean absolute paired change. Full-run intervals:
2,000 paired-path percentile bootstrap replicates, pointwise 95%, conditional on
fixed trained models, no multiplicity correction. Seeds are not independent paths.
No pooled headline test or causal/market generalization is prespecified.

## Controls and interpretation
Noiseless constant level scaling adds a common constant to log RV if clipping and
floors are inactive. Centered/differenced representations are invariant in exact
arithmetic; raw log-RV is not. This is primarily a structural negative control,
NOT an empirical proof that learned models distinguish roughness from volatility.
Record floating-point deviations, including amplified cross-channel normalized
features. Eta and smooth-pattern changes are the more informative stress tests.
A nonzero shift shows sensitivity in this design; it does not establish the
mechanism behind observed market regime associations. A near-zero shift is not
validation of market H or universal robustness. Report accuracy and sensitivity
separately; a constant predictor has zero sensitivity but need not be accurate.

## Execution and failure policy
Numeric stage runs without torch using explicit reference AST definitions, never
fake torch. Neural stage requires real torch, validates checkpoint/scaler identity,
strict state loading, finite bounded predictions, batch/serial and ensemble parity.
No neural results are created on dependency failure. Smoke results cannot be
promoted to full results. Fresh output directories are mandatory. Verify hashes
before loading inputs and retain data, code, environment, and output manifests.
