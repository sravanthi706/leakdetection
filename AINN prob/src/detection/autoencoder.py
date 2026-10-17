"""
Classical feature extraction + dense autoencoder anomaly detector
(spec 5.1/5.2) -- trained ONLY on normal-traffic sessions, so
reconstruction error spikes on out-of-distribution (leak-pattern)
input. Hand-rolled in NumPy for the same reason as the GNN module
(gnn.py): PyTorch could not be reliably installed in this environment.
Gradient-checked before training is trusted -- same discipline as the
GNN, even though this network is architecturally much simpler.

FEATURE INDEPENDENCE, enforced deliberately: every feature here is
purely TIMING/VOLUME/LEXICAL-KEYWORD derived from the session as a
whole. None of it touches content-overlap-with-the-true-paper (that's
content_match.py's exclusive signal) or image-sharing (that's the
pHash branch's exclusive signal) -- the same mistake caught and fixed
in gnn.py is deliberately avoided here from the start.

13 raw features -> 8-d bottleneck (matches the spec's "~8-dimensional
bottleneck" as real dimensionality REDUCTION, not just a label):
  [0:6]  hour-of-day histogram, 6 buckets of 4 hours each (fractions)
  [6]    keyword_density   -- fraction of text events containing a
                               GENERIC broker/transaction keyword
                               (different signal from content_match's
                               paper-specific similarity)
  [7]    burst_ratio        -- max events in any 1-hour window / total
  [8]    sender_novelty     -- fraction of senders seen exactly once
  [9]    distinct_senders_norm
  [10]   distinct_groups_norm
  [11]   avg_inter_event_gap_norm (inverted -- higher = burstier)
  [12]   event_volume_norm  -- log-scaled total event count
"""

import numpy as np

BROKER_KEYWORDS = ["confirmed", "advance", "pdf", "full set", "genuine", "lakh", "₹", "deal", "payment"]

RAW_DIM = 13
BOTTLENECK_DIM = 8


def extract_session_features(session: dict) -> np.ndarray:
    events = session["events"]
    n = len(events)
    if n == 0:
        return np.zeros(RAW_DIM)

    timestamps = sorted(e["timestamp"] for e in events)

    # [0:6] hour-of-day histogram, 6 buckets of 4 hours
    hour_hist = np.zeros(6)
    for e in events:
        hour = int((e["timestamp"] / 3600) % 24)
        hour_hist[hour // 4] += 1
    hour_hist /= n

    # [6] keyword density -- generic broker language, NOT paper-content similarity
    text_events = [e for e in events if e["event_type"] == "text"]
    if text_events:
        kw_hits = sum(1 for e in text_events if any(k in e["text"].lower() for k in BROKER_KEYWORDS))
        keyword_density = kw_hits / len(text_events)
    else:
        keyword_density = 0.0

    # [7] burst ratio -- max events in any 1-hour sliding window / total
    max_in_hour = 0
    for t0 in timestamps:
        count = sum(1 for t in timestamps if t0 <= t < t0 + 3600)
        max_in_hour = max(max_in_hour, count)
    burst_ratio = max_in_hour / n

    # [8] sender novelty -- fraction of senders who appear exactly once
    senders = [e["sender_id"] for e in events]
    unique, counts = np.unique(senders, return_counts=True)
    sender_novelty = float(np.sum(counts == 1)) / len(unique) if len(unique) > 0 else 0.0

    # [9], [10] distinct senders / groups, normalized by event volume
    distinct_senders_norm = len(set(senders)) / n
    groups = [e["group_id"] for e in events]
    distinct_groups_norm = len(set(groups)) / n

    # [11] average inter-event gap, inverted so higher value = burstier
    if len(timestamps) > 1:
        gaps = np.diff(timestamps)
        avg_gap = float(np.mean(gaps))
        avg_inter_event_gap_norm = 1.0 / (1.0 + avg_gap / 3600.0)  # ~1 if events seconds apart, ~0 if hours apart
    else:
        avg_inter_event_gap_norm = 0.0

    # [12] event volume, log-scaled
    event_volume_norm = min(np.log1p(n) / 6.0, 1.0)

    features = np.concatenate([
        hour_hist,
        [keyword_density, burst_ratio, sender_novelty, distinct_senders_norm,
         distinct_groups_norm, avg_inter_event_gap_norm, event_volume_norm],
    ])
    assert features.shape[0] == RAW_DIM
    return features


class Autoencoder:
    """Single hidden layer each side: 13 -> 8 (ReLU) -> 13 (linear). Trained with MSE loss."""

    def __init__(self, input_dim: int = RAW_DIM, bottleneck_dim: int = BOTTLENECK_DIM, seed: int = 0):
        rng = np.random.default_rng(seed)
        limit1 = np.sqrt(6.0 / (input_dim + bottleneck_dim))
        self.W1 = rng.uniform(-limit1, limit1, size=(bottleneck_dim, input_dim))
        self.b1 = np.zeros(bottleneck_dim)
        limit2 = np.sqrt(6.0 / (bottleneck_dim + input_dim))
        self.W2 = rng.uniform(-limit2, limit2, size=(input_dim, bottleneck_dim))
        self.b2 = np.zeros(input_dim)

    def forward(self, x: np.ndarray):
        z1 = self.W1 @ x + self.b1
        h = np.maximum(0, z1)  # bottleneck code
        x_hat = self.W2 @ h + self.b2
        return x_hat, {"x": x, "z1": z1, "h": h, "x_hat": x_hat}

    def loss(self, x: np.ndarray) -> float:
        x_hat, _ = self.forward(x)
        return float(np.mean((x_hat - x) ** 2))

    def backward(self, cache: dict):
        x, z1, h, x_hat = cache["x"], cache["z1"], cache["h"], cache["x_hat"]
        d = x.shape[0]
        d_xhat = 2.0 * (x_hat - x) / d
        dW2 = np.outer(d_xhat, h)
        db2 = d_xhat
        dh = self.W2.T @ d_xhat
        dz1 = dh * (z1 > 0)
        dW1 = np.outer(dz1, x)
        db1 = dz1
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def apply_gradients(self, grads: dict, lr: float):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]


def train_autoencoder(model: Autoencoder, feature_vectors: list, epochs: int, lr: float, verbose: bool = True):
    history = []
    for epoch in range(epochs):
        order = np.random.default_rng(epoch).permutation(len(feature_vectors))
        total = 0.0
        for idx in order:
            x = feature_vectors[idx]
            _, cache = model.forward(x)
            total += model.loss(x)
            grads = model.backward(cache)
            model.apply_gradients(grads, lr)
        avg = total / len(feature_vectors)
        history.append(avg)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  avg_reconstruction_mse={avg:.5f}")
    return history


if __name__ == "__main__":
    from src.detection.synthetic_data import generate_dataset

    # --- Gradient check first ---
    print("=== Gradient check ===")
    rng = np.random.default_rng(1)
    model_test = Autoencoder(seed=2)
    x_test = rng.uniform(0, 1, size=RAW_DIM)
    _, cache = model_test.forward(x_test)
    analytical = model_test.backward(cache)

    eps = 1e-5
    max_rel_error = 0.0
    for name, param in [("W1", model_test.W1), ("b1", model_test.b1), ("W2", model_test.W2), ("b2", model_test.b2)]:
        grad = analytical[name]
        it = np.nditer(param, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = param[idx]
            param[idx] = orig + eps
            loss_plus = model_test.loss(x_test)
            param[idx] = orig - eps
            loss_minus = model_test.loss(x_test)
            param[idx] = orig
            numerical = (loss_plus - loss_minus) / (2 * eps)
            rel_error = abs(numerical - grad[idx]) / max(1e-8, abs(numerical) + abs(grad[idx]))
            max_rel_error = max(max_rel_error, rel_error)
    print(f"Max relative error across all parameters: {max_rel_error:.2e} (must be < 1e-4)")
    assert max_rel_error < 1e-4, "GRADIENT CHECK FAILED"
    print("Gradient check PASSED.\n")

    # --- Train on NORMAL sessions only, per spec ---
    print("=== Training on NORMAL sessions only ===")
    dataset = generate_dataset(n_normal=80, n_leak=80, seed=5)
    normal_sessions = [s for s in dataset["sessions"] if s["label"] == "normal"]
    leak_sessions = [s for s in dataset["sessions"] if s["label"] == "leak"]

    split = int(0.75 * len(normal_sessions))
    train_normal, test_normal = normal_sessions[:split], normal_sessions[split:]

    train_features = [extract_session_features(s) for s in train_normal]
    model = Autoencoder(seed=9)
    train_autoencoder(model, train_features, epochs=60, lr=0.05)

    test_normal_errors = [model.loss(extract_session_features(s)) for s in test_normal]
    leak_errors = [model.loss(extract_session_features(s)) for s in leak_sessions]

    print(f"\nHeld-out NORMAL reconstruction error: mean={np.mean(test_normal_errors):.5f}, "
          f"max={np.max(test_normal_errors):.5f}")
    print(f"LEAK reconstruction error:            mean={np.mean(leak_errors):.5f}, "
          f"min={np.min(leak_errors):.5f}")

    assert np.mean(leak_errors) > np.mean(test_normal_errors) * 1.5, \
        "leak sessions should reconstruct noticeably worse than held-out normal sessions"
    print("\nConfirmed: leak-pattern sessions are genuinely out-of-distribution for a model "
          "trained only on normal traffic.")
