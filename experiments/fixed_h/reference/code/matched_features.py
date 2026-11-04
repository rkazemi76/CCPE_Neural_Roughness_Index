"""Matched information: all single-source features are functions of one W-vector only."""
import numpy as np


def single_features(logrv):
    x = np.asarray(logrv, np.float32)
    if x.ndim != 1 or len(x) < 8 or not np.isfinite(x).all():
        raise ValueError('Expected one finite log-RV context vector')
    z = x - x.mean()
    def diff(lag):
        out = np.zeros_like(z)
        out[lag:] = z[lag:] - z[:-lag]
        return out
    d1, d4 = diff(1), diff(4)
    ma = np.asarray([z[max(0, i-3):i+1].mean() for i in range(len(z))])
    # Temporal SD of this source, NOT cross-resolution SD (undefined for K=1).
    scale = max(float(z.std()), 1e-6)
    return {'single_raw': x[:, None].copy(),
            'single_derived': np.stack([z, d1, d4, ma, d1/scale, d4/scale], axis=1).astype(np.float32)}


def scaler_fit(x):
    mean = np.mean(x, axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(np.std(x, axis=(0, 1), dtype=np.float64), 1e-6).astype(np.float32)
    return mean, std


def split_seed(split, generator_index):
    # Disjoint path RNG streams; noise replicas remain within a split.
    return {'train': 710001, 'validation': 720001, 'calibration': 730001, 'test': 740001}[split] + 1009 * generator_index
