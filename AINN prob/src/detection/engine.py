"""
LeakDetector -- the upgraded, end-to-end detection layer.

Replaces fusion.FusionEngine for all evaluation (fusion.py is kept
unchanged so the original orchestrator/dashboard still run). What is
different, and why:

1. Seven signals instead of four fixed scores. Each branch now reports
   the separate pieces of evidence it computes, so the fusion can learn
   which ones are trustworthy:
      gnn                 propagation-graph GNN (PyTorch Geometric by default when
                           installed, auto-falls back to the NumPy implementation)
      autoencoder         behavioural anomaly, reconstruction error vs normal traffic
                           (PyTorch by default when installed, same auto-fallback)
      content_lexical     char-TF-IDF match to a committed question  (paper_matcher.py)
      content_semantic    WordLlama embedding match                  (paper_matcher.py)
      content_answer_key  posted answer list vs the committed key    (paper_matcher.py)
      image_phash         global perceptual hash vs committed pages  (image_match.py)
      image_orb           verified ORB+RANSAC page match             (image_match.py)

2. LEARNED fusion instead of 0.25 x 4. A logistic-regression meta-model
   (stacking) is fitted on a validation split the base models never saw,
   so its weights measure how much each signal actually helps. The
   weights are printed -- they are the model's own answer to "which
   evidence matters".

3. Threshold chosen on validation data, never on the test set.

4. Explanations that point at evidence, not just at a number: the
   message that matched which committed question, the photo that matched
   which page (with the number of geometric inliers), each signal's
   contribution to the final score, and -- with the PyTorch GNN -- which
   accounts drove the graph score.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.detection.gnn import CascadeGNN, build_propagation_graph, train as train_numpy_gnn
from src.detection.autoencoder import Autoencoder, extract_session_features, train_autoencoder
from src.detection.paper_matcher import matcher_for


def _has_torch_geometric() -> bool:
    try:
        import torch_geometric  # noqa: F401
        return True
    except ImportError:
        return False


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_backend(pref: str, available: bool) -> str:
    """"auto" -> torch when the dependency is importable, else numpy (graceful fallback).
    An explicit "numpy" or "torch" is always honoured as given (used by the benchmark to
    compare both deliberately)."""
    if pref == "auto":
        return "torch" if available else "numpy"
    return pref

# pHash is still computed and reported (image_benchmark.py), but it is NOT a fusion input:
# on phone photos it is at chance level (ROC-AUC ~0.51), and in a first run the meta-model gave
# it a NEGATIVE weight -- it had only learned "sessions with more photos are more often coaching
# channels". Verified ORB covers clean digital copies as well (ROC-AUC 1.00), so nothing is lost.
BRANCHES = ["gnn", "autoencoder", "content_lexical", "content_semantic", "content_answer_key", "image_orb"]


# ---------------------------------------------------------------------------
# Per-event evidence (model-independent: matchers are frozen, so this is
# computed once per event and cached on the event itself)
# ---------------------------------------------------------------------------

def annotate_events(sessions: list, pool_by_id: dict, semantic_backend: str = "wordllama"):
    """Adds `_ev` (content/image evidence) to every event. Idempotent."""
    by_paper = {}
    for s in sessions:
        for e in s["events"]:
            if "_ev" not in e and e["event_type"] == "text" and e.get("text"):
                by_paper.setdefault(s["exam_paper"], []).append(e)
    for paper, evs in by_paper.items():
        m = matcher_for(paper, semantic_backend)
        for i in range(0, len(evs), 512):
            chunk = evs[i:i + 512]
            for e, r in zip(chunk, m.score_messages([e["text"] for e in chunk])):
                e["_ev"] = {"kind": "text", "lexical": r["lexical"], "semantic": r["semantic"],
                            "answer_key": r["answer_key"], "matched_qid": r["matched_qid"], "evidence": r["evidence"]}
    for s in sessions:
        for e in s["events"]:
            if "_ev" in e:
                continue
            if e["event_type"] == "image" and e.get("image_id") in pool_by_id:
                sc = pool_by_id[e["image_id"]]["scores"][s["exam_paper"]]
                e["_ev"] = {"kind": "image", "phash": sc["phash_score"], "orb": sc["orb_score"],
                            "page": sc["page"], "inliers": sc["inliers"], "qids": sc["qids"]}
                # a captioned photo can also carry text evidence, but captions here are generic
            else:
                e["_ev"] = {"kind": "none"}


def prefix_views(sessions: list, rng, per_session: int = 2) -> list:
    """
    Truncated copies of sessions ("what the monitor had seen by hour h").
    The detector runs continuously, so it must be trained on partial
    sessions too -- otherwise a 3-message prefix looks nothing like any
    training example and the anomaly signals fire on it. Label of a prefix
    = whether it already contains leak activity (ground truth per event).
    """
    out = []
    for s in sessions:
        if not s["events"]:
            continue
        for _ in range(per_session):
            h = rng.uniform(0, 240)
            cut = s["window_start"] - h * 3600
            evs = [e for e in s["events"] if e["timestamp"] <= cut]
            if len(evs) >= 2:
                out.append(dict(s, events=evs, label="leak" if any(e["is_leak_event"] for e in evs) else "normal",
                                session_id=f"{s['session_id']}@{h:.0f}h"))
    return out


def static_signals(events: list) -> dict:
    """Max over events of each content/image signal (one damning message or photo is enough)."""
    out = {"content_lexical": 0.0, "content_semantic": 0.0, "content_answer_key": 0.0, "image_phash": 0.0, "image_orb": 0.0}
    for e in events:
        ev = e.get("_ev", {})
        if ev.get("kind") == "text":
            out["content_lexical"] = max(out["content_lexical"], ev["lexical"])
            out["content_semantic"] = max(out["content_semantic"], ev["semantic"])
            out["content_answer_key"] = max(out["content_answer_key"], ev["answer_key"])
        elif ev.get("kind") == "image":
            out["image_phash"] = max(out["image_phash"], ev["phash"])
            out["image_orb"] = max(out["image_orb"], ev["orb"])
    return out


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------

class LeakDetector:
    def __init__(self, gnn_backend: str = "auto", ae_backend: str = "auto", branches: list = None,
                 gnn_epochs: int = 12, ae_epochs: int = 40, seed: int = 0, streaming: bool = True):
        """
        gnn_backend / ae_backend: "auto" (default) uses PyTorch when it's installed
        (torch_geometric for the GNN, torch for the autoencoder) and falls back to the
        dependency-free NumPy implementation otherwise -- so the same code is "top spec"
        on a machine with PyTorch and still runs with zero extra installs on one without.
        Pass "numpy" or "torch" explicitly to force a backend (used by the benchmark to
        compare them side by side).
        """
        self.streaming = streaming
        self.gnn_backend = _resolve_backend(gnn_backend, _has_torch_geometric())
        self.ae_backend = _resolve_backend(ae_backend, _has_torch())
        self.branches = branches or list(BRANCHES)
        self.gnn_epochs, self.ae_epochs, self.seed = gnn_epochs, ae_epochs, seed
        self.gnn = None
        self.ae = None
        self.ae_baseline = 1.0
        self.meta = None
        self.scaler = None
        self.threshold = 0.5

    # --- base models -------------------------------------------------------
    def _fit_gnn(self, sessions, labels):
        if self.gnn_backend == "torch":
            from src.detection.gnn_torch import TorchCascadeGNN
            self.gnn = TorchCascadeGNN(seed=self.seed)
            self.gnn.fit(sessions, labels)
        else:
            self.gnn = CascadeGNN(in_dim=4, hidden_dim=8, num_layers=2, seed=self.seed)
            relabelled = [dict(s, label="leak" if y else "normal") for s, y in zip(sessions, labels)]
            train_numpy_gnn(self.gnn, relabelled, epochs=self.gnn_epochs, lr=0.1, verbose=False)

    def gnn_scores(self, sessions) -> np.ndarray:
        if self.gnn_backend == "torch":
            return self.gnn.predict_proba(sessions)
        out = []
        for s in sessions:
            if not s["events"]:
                out.append(0.0)
                continue
            _, X0, adj = build_propagation_graph(s)
            out.append(float(self.gnn.forward(X0, adj)[0]))
        return np.array(out)

    def _fit_autoencoder(self, normal_sessions):
        feats = [extract_session_features(s) for s in normal_sessions]
        if self.ae_backend == "torch":
            from src.detection.autoencoder_torch import TorchAutoencoder, train_autoencoder_torch
            self.ae = TorchAutoencoder(seed=self.seed)
            train_autoencoder_torch(self.ae, feats, epochs=self.ae_epochs, lr=0.02, verbose=False)
        else:
            self.ae = Autoencoder(seed=self.seed)
            train_autoencoder(self.ae, feats, epochs=self.ae_epochs, lr=0.05, verbose=False)
        self.ae_baseline = float(np.mean([self.ae.loss(f) for f in feats])) + 1e-9

    def ae_scores(self, sessions) -> np.ndarray:
        # log-ratio of reconstruction error to the normal-traffic baseline: 0 = typical, >0 = unusual
        return np.array([np.log((self.ae.loss(extract_session_features(s)) + 1e-9) / self.ae_baseline) for s in sessions])

    # --- features ------------------------------------------------------------
    def branch_matrix(self, sessions) -> np.ndarray:
        cols = {"gnn": None, "autoencoder": None}
        if "gnn" in self.branches:
            cols["gnn"] = self.gnn_scores(sessions)
        if "autoencoder" in self.branches:
            cols["autoencoder"] = self.ae_scores(sessions)
        stat = [static_signals(s["events"]) for s in sessions]
        X = []
        for i in range(len(sessions)):
            row = []
            for b in self.branches:
                row.append(cols[b][i] if b in ("gnn", "autoencoder") else stat[i][b])
            X.append(row)
        return np.array(X, dtype=float)

    # --- training ----------------------------------------------------------------
    def fit(self, train_sessions, val_sessions, train_labels=None, val_labels=None):
        """
        Base models on `train_sessions`; meta-model + threshold on `val_sessions`
        (out-of-sample for the base models). `train_labels` / `val_labels` let an
        experiment inject label noise without touching the sessions.
        With streaming=True both sets are augmented with truncated prefixes.
        """
        import random as _random
        rng = _random.Random(self.seed)
        y_tr = list(train_labels if train_labels is not None else [s["label"] == "leak" for s in train_sessions])
        y_va = list(val_labels if val_labels is not None else [s["label"] == "leak" for s in val_sessions])
        train_sessions, val_sessions = list(train_sessions), list(val_sessions)
        if self.streaming:
            # prefixes inherit the (possibly noisy) session label only if the prefix already holds leak activity
            for sessions, labels in ((train_sessions, y_tr), (val_sessions, y_va)):
                label_of = {s["session_id"]: y for s, y in zip(sessions, labels)}
                pv = prefix_views(list(sessions), rng)
                sessions += pv
                labels += [bool(label_of[p["session_id"].split("@")[0]]) and p["label"] == "leak" for p in pv]
        y_tr = np.array(y_tr, dtype=int)
        if "gnn" in self.branches:
            self._fit_gnn(train_sessions, y_tr)
        if "autoencoder" in self.branches:
            self._fit_autoencoder([s for s, y in zip(train_sessions, y_tr) if y == 0])
        Xv = self.branch_matrix(val_sessions)
        yv = np.array(y_va, dtype=int)
        self.scaler = StandardScaler().fit(Xv)
        Z = self.scaler.transform(Xv)
        self.meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Z, yv)
        # threshold from cross-validated meta predictions on the validation set (no test peeking)
        cv = StratifiedKFold(5, shuffle=True, random_state=self.seed)
        p_cv = cross_val_predict(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000), Z, yv,
                                 cv=cv, method="predict_proba")[:, 1]
        self.threshold = best_f1_threshold(yv, p_cv)
        return self

    def predict_proba(self, sessions) -> np.ndarray:
        return self.meta.predict_proba(self.scaler.transform(self.branch_matrix(sessions)))[:, 1]

    def weights(self) -> dict:
        return {b: round(float(w), 3) for b, w in zip(self.branches, self.meta.coef_[0])}

    # --- explanation ---------------------------------------------------------------
    def explain(self, session: dict, top_k: int = 3) -> dict:
        x = self.branch_matrix([session])[0]
        z = self.scaler.transform([x])[0]
        contrib = dict(zip(self.branches, (self.meta.coef_[0] * z).round(3)))
        prob = float(self.meta.predict_proba([z])[0, 1])
        # forwarded copies of the same item are shown once (first sender), strongest evidence first
        texts, seen = [], set()
        for e in sorted((e for e in session["events"] if e.get("_ev", {}).get("kind") == "text"),
                        key=lambda e: (-max(e["_ev"]["lexical"], e["_ev"]["semantic"], e["_ev"]["answer_key"]), e["timestamp"])):
            if e["text"] not in seen:
                seen.add(e["text"])
                texts.append(e)
        imgs, seen = [], set()
        for e in sorted((e for e in session["events"] if e.get("_ev", {}).get("kind") == "image"),
                        key=lambda e: (-e["_ev"]["orb"], e["timestamp"])):
            if e["image_id"] not in seen:
                seen.add(e["image_id"])
                imgs.append(e)
        out = {
            "session_id": session["session_id"], "probability": prob, "alert": prob >= self.threshold,
            "threshold": self.threshold, "signals": {b: round(float(v), 3) for b, v in zip(self.branches, x)},
            "contributions": {b: float(v) for b, v in contrib.items()},
            "top_messages": [{"text": e["text"][:120], "sender": e["sender_id"], "group": e["group_id"],
                              "matched_qid": e["_ev"]["matched_qid"], "via": e["_ev"]["evidence"],
                              "score": round(max(e["_ev"]["lexical"], e["_ev"]["semantic"], e["_ev"]["answer_key"]), 3)}
                             for e in texts[:top_k] if max(e["_ev"]["lexical"], e["_ev"]["semantic"], e["_ev"]["answer_key"]) > 0],
            "top_images": [{"image_id": e["image_id"], "sender": e["sender_id"], "page": e["_ev"]["page"],
                            "inliers": e["_ev"]["inliers"], "orb": round(e["_ev"]["orb"], 3), "qids": e["_ev"]["qids"]}
                           for e in imgs[:top_k] if e["_ev"]["orb"] > 0.2],
        }
        if self.gnn_backend == "torch" and hasattr(self.gnn, "explain_nodes"):
            out["top_accounts"] = self.gnn.explain_nodes(session, top_k=5)
        return out


def best_f1_threshold(y, p) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.round(p, 4)):
        pred = p >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def format_explanation(ex: dict) -> str:
    lines = [f"Session {ex['session_id']}: P(leak) = {ex['probability']:.3f}  (threshold {ex['threshold']:.3f})  "
             f"-> {'ALERT' if ex['alert'] else 'no alert'}"]
    lines.append("  contribution of each signal to the decision (log-odds, + pushes towards leak):")
    for k, v in sorted(ex["contributions"].items(), key=lambda kv: -kv[1]):
        lines.append(f"    {k:20s} {v:+.2f}   (raw signal {ex['signals'][k]:.3f})")
    for m in ex["top_messages"]:
        lines.append(f"  message matched {m['matched_qid']} via {m['via']} (score {m['score']}): \"{m['text']}\"")
    for im in ex["top_images"]:
        lines.append(f"  photo {im['image_id']} matched committed page {im['page']} with {im['inliers']} geometric "
                     f"inliers (score {im['orb']}) -- questions on that page: {', '.join(q.split('#')[1] for q in im['qids'])}")
    for acc in ex.get("top_accounts", []):
        lines.append(f"  graph (GNNExplainer): node {acc['node']} importance {acc['importance']:.3f}")
    return "\n".join(lines)
