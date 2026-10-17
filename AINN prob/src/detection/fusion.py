"""
Fusion + alerting -- spec 5.4. Combines the four independent branch
scores (cascade/GNN, content-match, reconstruction-error/autoencoder,
image-match/pHash) into one anomaly score via weighted ensemble, with
per-signal feature attribution so an alert explains WHY it fired --
feeding a human decision, not an automated takedown, per spec.

Session-level aggregation choice, stated explicitly: content-match and
image-match are computed PER EVENT (per message/image); this module
takes the MAX over a session's events for each, on the reasoning that
one damning message is enough to warrant attention -- averaging would
dilute a single confirmed leak photo across dozens of harmless chat
messages in the same session.
"""

import numpy as np

from src.detection.gnn import CascadeGNN, build_propagation_graph
from src.detection.autoencoder import Autoencoder, extract_session_features
from src.detection.content_match import ContentMatcher
from src.detection.phash_match import image_match_score

DEFAULT_WEIGHTS = {"gnn": 0.25, "content": 0.25, "autoencoder": 0.25, "image": 0.25}
DEFAULT_THRESHOLD = 0.5


class FusionEngine:
    def __init__(self, gnn_model: CascadeGNN, autoencoder_model: Autoencoder,
                 content_matcher: ContentMatcher, normal_mse_baseline: float,
                 weights: dict = None, threshold: float = DEFAULT_THRESHOLD):
        self.gnn_model = gnn_model
        self.autoencoder_model = autoencoder_model
        self.content_matcher = content_matcher
        self.normal_mse_baseline = normal_mse_baseline  # mean reconstruction MSE on held-out NORMAL sessions
        self.weights = weights or DEFAULT_WEIGHTS
        self.threshold = threshold

    def _autoencoder_score(self, session: dict) -> float:
        features = extract_session_features(session)
        mse = self.autoencoder_model.loss(features)
        # calibrated as "how many multiples of the typical normal reconstruction error is this",
        # capped at 1.0 -- interpretable, and anchored to a measured baseline, not a guess.
        return float(min(1.0, mse / (10.0 * self.normal_mse_baseline)))

    def _gnn_score(self, session: dict) -> float:
        _, X0, adjacency = build_propagation_graph(session)
        prob, _ = self.gnn_model.forward(X0, adjacency)
        return float(prob)

    def _content_score(self, session: dict) -> float:
        text_events = [e for e in session["events"] if e["event_type"] == "text"]
        if not text_events:
            return 0.0
        return max(self.content_matcher.content_match_score(e["text"]) for e in text_events)

    def _image_score(self, session: dict) -> float:
        image_events = [e for e in session["events"] if e["event_type"] == "image"]
        if not image_events:
            return 0.0
        return max(image_match_score(e["image_phash_hamming_to_true"]) for e in image_events)

    def score_session(self, session: dict) -> dict:
        scores = {
            "gnn": self._gnn_score(session),
            "content": self._content_score(session),
            "autoencoder": self._autoencoder_score(session),
            "image": self._image_score(session),
        }
        contributions = {k: scores[k] * self.weights[k] for k in scores}
        combined = sum(contributions.values())
        alert = combined >= self.threshold

        attribution = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)

        return {
            "session_id": session["session_id"],
            "true_label": session["label"],
            "scores": scores,
            "contributions": contributions,
            "combined_score": combined,
            "alert": alert,
            "attribution_ranked": attribution,
        }


def explain(result: dict) -> str:
    lines = [f"Session {result['session_id']} (true label: {result['true_label']}) "
             f"-> combined_score={result['combined_score']:.3f} -> ALERT={result['alert']}"]
    for name, contribution in result["attribution_ranked"]:
        raw = result["scores"][name]
        lines.append(f"    {name:12s} raw={raw:.3f}  weighted_contribution={contribution:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    from src.detection.synthetic_data import generate_dataset, TRUE_PAPER_QUESTIONS
    from src.detection.gnn import train as train_gnn, evaluate as evaluate_gnn
    from src.detection.autoencoder import train_autoencoder

    print("=== Building and training all four branches ===")
    dataset = generate_dataset(n_normal=80, n_leak=80, seed=21)
    sessions = dataset["sessions"]
    split = int(0.75 * len(sessions))
    train_sessions, test_sessions = sessions[:split], sessions[split:]
    train_normal = [s for s in train_sessions if s["label"] == "normal"]

    gnn_model = CascadeGNN(in_dim=4, hidden_dim=8, num_layers=2, seed=7)
    train_gnn(gnn_model, train_sessions, epochs=30, lr=0.3, verbose=False)
    print("GNN trained.")

    ae_model = Autoencoder(seed=9)
    ae_features = [extract_session_features(s) for s in train_normal]
    train_autoencoder(ae_model, ae_features, epochs=60, lr=0.05, verbose=False)
    normal_mse_baseline = float(np.mean([ae_model.loss(f) for f in ae_features]))
    print(f"Autoencoder trained. Normal MSE baseline = {normal_mse_baseline:.5f}")

    content_matcher = ContentMatcher(TRUE_PAPER_QUESTIONS)
    print("Content matcher ready (frozen TF-IDF).")
    print("pHash scoring ready (no training needed -- classical).\n")

    engine = FusionEngine(gnn_model, ae_model, content_matcher, normal_mse_baseline)

    print("=== Example: one leak session, full attribution ===")
    example_leak = next(s for s in test_sessions if s["label"] == "leak")
    print(explain(engine.score_session(example_leak)))

    print("\n=== Example: one normal session, full attribution ===")
    example_normal = next(s for s in test_sessions if s["label"] == "normal")
    print(explain(engine.score_session(example_normal)))

    print("\n=== Evaluating fused decision on the FULL held-out test set ===")
    tp = fp = tn = fn = 0
    for s in test_sessions:
        result = engine.score_session(s)
        is_leak = s["label"] == "leak"
        if is_leak and result["alert"]:
            tp += 1
        elif is_leak and not result["alert"]:
            fn += 1
        elif not is_leak and result["alert"]:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    print(f"n={total}  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Accuracy={accuracy:.2%}  Precision={precision:.2%}  Recall={recall:.2%}")
    print("\n(Reporting the actual measured numbers above, not assuming they'll be perfect --")
    print(" the four branches were independently strong, so a strong fused result is expected,")
    print(" but it's checked here, not asserted blind.)")
