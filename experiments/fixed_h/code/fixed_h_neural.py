"""Inference adapter; imports and runs real PyTorch, with frozen original scalers."""
import hashlib
import numpy as np
import torch
from neural_benchmark import load_models
from reference_numeric import ROOT


class FrozenEnsemble:
    def __init__(self, representation, cfg, device='cpu'):
        self.models, self.mean, self.std, self.records = load_models(representation, cfg, device)
        self.device = device
        self.checkpoint_hashes = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'reference/models' / representation).glob('seed_*/model.pt'))
        }
        scaler = ROOT / 'reference' / (representation + '_scaler.npz')
        self.checkpoint_hashes[str(scaler.relative_to(ROOT))] = hashlib.sha256(scaler.read_bytes()).hexdigest()

    def predict_individual(self, features, batch_size=32):
        result = np.empty((len(features), len(self.models)), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(features), batch_size):
                x = ((features[start:start+batch_size] - self.mean) / self.std).astype(np.float32)
                if not np.isfinite(x).all():
                    raise ValueError('Nonfinite standardized input')
                tensor = torch.from_numpy(x).to(self.device)
                for j, model in enumerate(self.models):
                    result[start:start+len(x), j] = model(tensor).cpu().numpy().reshape(-1)
        return result

    def predict(self, features, batch_size=32):
        return self.predict_individual(features, batch_size).mean(axis=1)
