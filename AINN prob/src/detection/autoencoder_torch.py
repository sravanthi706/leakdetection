"""
PyTorch version of the behavioural-anomaly autoencoder (optional backend).

Same architecture, features and reconstruction-error interface as
autoencoder.py (13 -> 8 -> 13, trained on normal-traffic sessions only)
so it is a drop-in replacement for engine.py: same `.loss(x) -> float`
method, same `extract_session_features`. The NumPy version stays as the
dependency-free default and the gradient-checked teaching implementation;
this module is used automatically when PyTorch is installed (see
`ae_backend="auto"` in engine.py).

Trained with full-batch Adam instead of the NumPy version's per-sample
SGD -- equivalent objective (mean reconstruction MSE on normal traffic),
faster and numerically identical in spirit.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.detection.autoencoder import RAW_DIM, BOTTLENECK_DIM, extract_session_features  # noqa: F401  (re-exported)


class TorchAutoencoder(nn.Module):
    """Single hidden layer each side: 13 -> 8 (ReLU) -> 13 (linear), MSE loss -- mirrors Autoencoder."""

    def __init__(self, input_dim: int = RAW_DIM, bottleneck_dim: int = BOTTLENECK_DIM, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Linear(input_dim, bottleneck_dim)
        self.dec = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(F.relu(self.enc(x)))

    @torch.no_grad()
    def loss(self, x: np.ndarray) -> float:
        """Same signature as Autoencoder.loss: one feature vector in, reconstruction MSE out."""
        self.eval()
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
        return float(F.mse_loss(self.forward(xt), xt).item())


def train_autoencoder_torch(model: TorchAutoencoder, feature_vectors: list, epochs: int, lr: float,
                             verbose: bool = True):
    """Full-batch Adam on the stacked feature matrix -- same objective as train_autoencoder (NumPy)."""
    X = torch.from_numpy(np.stack(feature_vectors).astype(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(model(X), X)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  avg_reconstruction_mse={loss.item():.5f}")
    return history


if __name__ == "__main__":
    from src.detection.synthetic_data import generate_dataset

    print("=== Training TorchAutoencoder on NORMAL sessions only ===")
    dataset = generate_dataset(n_normal=80, n_leak=80, seed=5)
    normal_sessions = [s for s in dataset["sessions"] if s["label"] == "normal"]
    leak_sessions = [s for s in dataset["sessions"] if s["label"] == "leak"]

    split = int(0.75 * len(normal_sessions))
    train_normal, test_normal = normal_sessions[:split], normal_sessions[split:]

    train_features = [extract_session_features(s) for s in train_normal]
    model = TorchAutoencoder(seed=9)
    train_autoencoder_torch(model, train_features, epochs=60, lr=0.02)

    test_normal_errors = [model.loss(extract_session_features(s)) for s in test_normal]
    leak_errors = [model.loss(extract_session_features(s)) for s in leak_sessions]

    print(f"\nHeld-out NORMAL reconstruction error: mean={np.mean(test_normal_errors):.5f}, "
          f"max={np.max(test_normal_errors):.5f}")
    print(f"LEAK reconstruction error:            mean={np.mean(leak_errors):.5f}, "
          f"min={np.min(leak_errors):.5f}")

    assert np.mean(leak_errors) > np.mean(test_normal_errors) * 1.5, \
        "leak sessions should reconstruct noticeably worse than held-out normal sessions"
    print("\nConfirmed: leak-pattern sessions are genuinely out-of-distribution for the Torch autoencoder too.")
