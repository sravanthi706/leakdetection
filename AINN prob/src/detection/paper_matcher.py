"""
Content branch, upgraded: does this chat message contain (part of) the
COMMITTED paper -- in any form it realistically travels in?

The original content_match.py compared each message with ONE pooled
vector for the whole paper, using character TF-IDF only. That works for
copy-pasted text and fails on anything reworded (its own self-test
showed a paraphrase scoring 0.03). This module fixes three things:

1. Per-question matching. A message is compared with EACH committed
   question separately and the best match is kept -- so the alert can
   say "this message matches Q11 of the committed paper", which is the
   explanation a human reviewer needs.

2. Three independent signals, because leaks take three different forms:
     lexical     character 3-5-gram TF-IDF. Survives OCR noise, typos,
                 truncation and LaTeX-stripping. Blind to rewording.
     semantic    WordLlama sentence embeddings (a real pretrained model,
                 trained on sentence-similarity data, weights ship inside
                 the pip package so it works offline). Survives rewording
                 in English. Optionally a MiniLM sentence-transformer if
                 you have it installed locally (`backend="minilm"`).
     answer_key  parses "Q3-B, Q4-AC, Q11-281" style lists and checks
                 them against the COMMITTED answer key. An answer-key
                 leak contains no question text at all, so neither
                 similarity model can see it -- but the exam authority
                 knows the true key, and random guessing matches it
                 rarely (1 in 4 for single-choice, far less for
                 multi-select and integer answers).

3. Calibration against legitimate traffic, not a guessed threshold.
   Every raw similarity is rescaled so that the 95th percentile of
   scores on KNOWN-LEGITIMATE text (past papers on the same syllabus,
   ordinary study chat) maps to 0. "0.7 similar" means nothing on its
   own; "higher than 95% of genuine past-paper discussion" does.
"""

import re
import random
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.detection.exam_corpus import load_papers, past_papers
from src.detection import leak_text

_QPREFIX = re.compile(r"^\s*(q\.?\s*\d+[\.\):\-]?|\d+[\.\)])\s*", re.I)


def normalise(text: str) -> str:
    t = _QPREFIX.sub("", text or "")
    return re.sub(r"\s+", " ", t.lower()).strip()


# ---------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def _wordllama():
    import wordllama
    from pathlib import Path
    from wordllama import WordLlama
    # The package ships its weights and tokenizer, but its own lookup path for the
    # tokenizer is wrong ("tokenizer" vs "tokenizers"); pointing cache_dir at the
    # package folder makes it find both bundled files -- no network needed.
    return WordLlama.load(cache_dir=Path(wordllama.__file__).parent, disable_download=True)


@lru_cache(maxsize=1)
def _minilm():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(texts: list, backend: str = "wordllama") -> np.ndarray:
    if backend == "minilm":
        v = _minilm().encode(texts, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)
    v = np.asarray(_wordllama().embed(texts), dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def semantic_backend_available(backend: str) -> bool:
    try:
        embed(["test"], backend)
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _global_char_vectorizer():
    """Fit ONCE on the public question corpus + generic chat, then frozen (never fit on leak/normal labels)."""
    corpus = [normalise(q["plain"]) for qs in load_papers().values() for q in qs]
    corpus += [normalise(t) for t in leak_text.STUDY_CHATTER + leak_text.COACHING_BROADCAST + leak_text.SCAM_POSTS]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    vec.fit(corpus)
    return vec


# ---------------------------------------------------------------------
# Answer-key check
# ---------------------------------------------------------------------

# "Q3-B", "Q.3) AC", "3. 281", "3-12.5" -- the optional Q prefix must carry its own word
# boundary; a \b between "Q" and "3" would never match (both are word characters).
_PAIR = re.compile(r"(?:\bq\.?\s*|\b)(\d{1,2})\s*[\.\)\-:]\s*([a-d]{1,4}|\d+(?:\.\d+)?)(?![\w.])", re.I)


def _norm_answer(a: str) -> str:
    a = a.strip().upper()
    if re.fullmatch(r"[A-D]+", a):
        return "".join(sorted(set(a)))
    try:
        return f"{float(a):g}"
    except ValueError:
        return a


def answer_key_evidence(text: str, gold_by_number: dict) -> dict:
    pairs = [(int(n), _norm_answer(a)) for n, a in _PAIR.findall(text or "")]
    pairs = [(n, a) for n, a in pairs if n in gold_by_number]
    if len(pairs) < 3:
        return {"pairs": len(pairs), "matched": 0, "score": 0.0, "qids": []}
    matched = [n for n, a in pairs if a == gold_by_number[n]["answer"]]
    frac = len(matched) / len(pairs)
    score = min(1.0, len(pairs) / 4.0) * float(np.clip((frac - 0.25) / 0.75, 0.0, 1.0))
    return {"pairs": len(pairs), "matched": len(matched), "score": score,
            "qids": [gold_by_number[n]["qid"] for n in matched]}


# ---------------------------------------------------------------------
# The matcher
# ---------------------------------------------------------------------

class PaperContentMatcher:
    def __init__(self, paper_name: str, backend: str = "wordllama", calibrate: bool = True, seed: int = 0):
        self.paper_name = paper_name
        self.backend = backend
        self.questions = load_papers()[paper_name]
        self.gold = {q["number"]: {"answer": _norm_answer(q["gold"]), "qid": q["qid"]} for q in self.questions}
        self.vec = _global_char_vectorizer()
        # each question is represented by its full text AND its stem alone (a leak often drops the options)
        self._qtexts = [normalise(q["plain"]) for q in self.questions] + [normalise(q["stem_plain"]) for q in self.questions]
        self._qids = [q["qid"] for q in self.questions] * 2
        self.Q_char = self.vec.transform(self._qtexts)
        self.Q_sem = None
        if backend:
            try:
                self.Q_sem = embed(self._qtexts, backend)
            except Exception:
                self.Q_sem = None            # no semantic model available -> lexical + answer-key only
        self.cal = {"lexical": (0.0, 1.0), "semantic": (0.0, 1.0)}
        if calibrate:
            self._calibrate(seed)

    # raw scores ---------------------------------------------------------
    def _raw(self, texts: list):
        norm = [normalise(t) for t in texts]
        C = self.vec.transform(norm)
        lex = (C @ self.Q_char.T).toarray()                      # rows: messages, cols: question variants
        sem = None
        if self.Q_sem is not None:
            sem = embed(norm, self.backend) @ self.Q_sem.T
        return lex, sem

    def _calibrate(self, seed: int):
        """Legitimate reference traffic: past-paper questions in every rendering + ordinary chat."""
        rng = random.Random(seed)
        ref = []
        for pp in past_papers():
            for q in rng.sample(load_papers()[pp], min(12, len(load_papers()[pp]))):
                for style in ("verbatim", "excerpt", "typed", "paraphrase", "ocr"):
                    ref.append(leak_text.RENDERERS[style](q, rng))
        ref += [leak_text.fill(t, rng) for t in leak_text.STUDY_CHATTER + leak_text.COACHING_BROADCAST + leak_text.SCAM_POSTS]
        lex, sem = self._raw(ref)
        lo = float(np.quantile(lex.max(axis=1), 0.95))
        self.cal["lexical"] = (lo, 1.0)
        if sem is not None:
            self.cal["semantic"] = (float(np.quantile(sem.max(axis=1), 0.95)), 1.0)

    def _calibrated(self, raw: float, kind: str) -> float:
        lo, hi = self.cal[kind]
        return float(np.clip((raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

    # public -------------------------------------------------------------
    def score_messages(self, texts: list) -> list:
        """Per message: raw + calibrated scores for each signal, combined score, and the matched question."""
        if not texts:
            return []
        lex, sem = self._raw(texts)
        out = []
        for i, t in enumerate(texts):
            li = int(lex[i].argmax())
            r = {"lexical_raw": float(lex[i, li]), "lexical": self._calibrated(float(lex[i, li]), "lexical"),
                 "lexical_qid": self._qids[li]}
            if sem is not None:
                si = int(sem[i].argmax())
                r.update(semantic_raw=float(sem[i, si]), semantic=self._calibrated(float(sem[i, si]), "semantic"),
                         semantic_qid=self._qids[si])
            else:
                r.update(semantic_raw=0.0, semantic=0.0, semantic_qid=None)
            ak = answer_key_evidence(t, self.gold)
            r.update(answer_key=ak["score"], answer_key_pairs=ak["pairs"], answer_key_matched=ak["matched"])
            parts = {"lexical": r["lexical"], "semantic": r["semantic"], "answer_key": r["answer_key"]}
            best = max(parts, key=parts.get)
            r["score"] = parts[best]
            r["evidence"] = best
            r["matched_qid"] = {"lexical": r["lexical_qid"], "semantic": r["semantic_qid"],
                                "answer_key": (ak["qids"][0] if ak["qids"] else None)}[best]
            out.append(r)
        return out

    def score(self, text: str) -> float:
        return self.score_messages([text])[0]["score"]


@lru_cache(maxsize=16)
def matcher_for(paper_name: str, backend: str = "wordllama") -> PaperContentMatcher:
    return PaperContentMatcher(paper_name, backend=backend)


if __name__ == "__main__":
    rng = random.Random(3)
    paper = "JEE Adv 2023 Paper 1"
    m = matcher_for(paper)
    print(f"Calibration (95th pct of legitimate traffic): lexical={m.cal['lexical'][0]:.3f} "
          f"semantic={m.cal['semantic'][0]:.3f}")
    q = load_papers()[paper][10]
    past_q = load_papers()["JEE Adv 2019 Paper 1"][4]
    msgs = {
        "verbatim leak": leak_text.verbatim(q, rng),
        "OCR-noisy leak": leak_text.ocr(q, rng),
        "paraphrased leak": leak_text.paraphrase(q, rng),
        "hinglish leak": leak_text.hinglish(q, rng),
        "answer-key leak": leak_text.answer_key(load_papers()[paper][7:13], rng),
        "PAST paper, verbatim": leak_text.verbatim(past_q, rng),
        "study chat": "anyone have notes for chapter 5?",
        "scam post": leak_text.fill(leak_text.SCAM_POSTS[0], rng),
    }
    for name, r in zip(msgs, m.score_messages(list(msgs.values()))):
        print(f"{name:22s} lex={r['lexical']:.2f} sem={r['semantic']:.2f} key={r['answer_key']:.2f} "
              f"-> {r['score']:.2f} via {r['evidence']:10s} -> {r['matched_qid']}")
