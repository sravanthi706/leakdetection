"""
Baselines the AINN detector has to beat. Without these, "F1 = 0.9" is
meaningless -- an examiner's first question is "compared to what?".

  keyword_rule       the obvious non-ML approach: alert when the share of
                     messages containing broker/transaction words
                     ("confirmed", "advance", "PDF", "payment", ...) is high.
  features_lr        logistic regression on hand-made session statistics
                     (volume, timing, burstiness, #groups, #regions, #images,
                     keyword density) -- classical ML, no graph, no content match.
  features_rf        random forest on the same statistics.
  text_tfidf_lr      supervised text classifier: TF-IDF of ALL the session's
                     messages + logistic regression. A strong generic
                     baseline -- it learns leak vocabulary from labels, but
                     knows nothing about the committed paper.

All baselines are trained on exactly the same training split as AINN and
their threshold is chosen on the same validation split.
"""

import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.detection.autoencoder import extract_session_features, BROKER_KEYWORDS
from src.detection.engine import best_f1_threshold


def keyword_density(session) -> float:
    texts = [(e.get("text") or "").lower() for e in session["events"] if e.get("text")]
    if not texts:
        return 0.0
    return sum(any(k in t for k in BROKER_KEYWORDS) for t in texts) / len(texts)


def handcrafted(session) -> np.ndarray:
    ev = session["events"]
    base = extract_session_features(session)
    extra = [math.log1p(len(ev)), sum(e["event_type"] == "image" for e in ev),
             len({e["region"] for e in ev}), len({e["group_id"] for e in ev}), math.log1p(len({e["sender_id"] for e in ev}))]
    return np.concatenate([base, extra])


def session_text(session) -> str:
    return " \n ".join((e.get("text") or "") for e in session["events"])


class KeywordRule:
    name = "keyword_rule"

    def fit(self, train, val):
        yv = np.array([s["label"] == "leak" for s in val], dtype=int)
        self.threshold = best_f1_threshold(yv, self.predict_proba(val))
        return self

    def predict_proba(self, sessions):
        return np.array([keyword_density(s) for s in sessions])


class _Supervised:
    def fit(self, train, val):
        y = np.array([s["label"] == "leak" for s in train], dtype=int)
        self.model.fit(self._X(train), y)
        yv = np.array([s["label"] == "leak" for s in val], dtype=int)
        self.threshold = best_f1_threshold(yv, self.predict_proba(val))
        return self

    def predict_proba(self, sessions):
        return self.model.predict_proba(self._X(sessions))[:, 1]


class FeaturesLR(_Supervised):
    name = "features_lr"

    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))

    def _X(self, ss):
        return np.array([handcrafted(s) for s in ss])


class FeaturesRF(_Supervised):
    name = "features_rf"

    def __init__(self, seed=0):
        self.model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                            random_state=seed, n_jobs=2)

    def _X(self, ss):
        return np.array([handcrafted(s) for s in ss])


class TextTfidfLR(_Supervised):
    name = "text_tfidf_lr"

    def __init__(self):
        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000),
                                   LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000))

    def _X(self, ss):
        return [session_text(s) for s in ss]


ALL_BASELINES = [KeywordRule, FeaturesLR, FeaturesRF, TextTfidfLR]
