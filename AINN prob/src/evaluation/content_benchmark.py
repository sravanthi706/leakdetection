"""
Content-branch benchmark -- two parts, both on REAL text.

Part 1: standard paraphrase benchmark (MSRP, Dolan & Brockett 2005).
   1,725 human-labelled sentence pairs (test split). For each text
   representation we score every pair by cosine similarity and report
   ROC-AUC (how well "is a paraphrase" is ranked above "is not") plus the
   best-threshold accuracy. This is an external, standard yardstick for
   the claim "TF-IDF can't handle rewording, embeddings can".

Part 2: leak-message detection against a REAL committed paper.
   Committed papers: JEE Advanced 2023 Paper 1 and Paper 2 (real questions).
   Positives: messages carrying a committed question, in every realistic
   form -- verbatim, excerpt, OCR-noisy, typed shorthand, English
   paraphrase and Hinglish retype (hand-written for 40 questions), and
   answer-key lists.
   Negatives: the SAME renderings applied to PAST JEE papers (same
   syllabus, same style), past-paper answer keys, ordinary study chat,
   coaching ads and scam posts.
   Reported per form: ROC-AUC, recall at 1% false-positive rate, and --
   for explainability -- how often the matcher names the right question.

Run:  python -m src.evaluation.content_benchmark
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve

from src.detection.exam_corpus import load_papers, past_papers
from src.detection import leak_text
from src.detection.paper_matcher import PaperContentMatcher, embed, semantic_backend_available

ROOT = Path(__file__).resolve().parents[2]
MSRP = ROOT / "data" / "external" / "msrp"
OUT = ROOT / "reports" / "results"
COMMITTED = ["JEE Adv 2023 Paper 1", "JEE Adv 2023 Paper 2"]


def _read_msrp(split):
    rows = []
    with open(MSRP / f"msr_paraphrase_{split}.txt", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 5:
                rows.append((int(p[0]), p[3], p[4]))
    return rows


def _best_acc(y, s):
    best = 0.0
    for t in np.unique(np.round(s, 4)):
        best = max(best, float(np.mean((s >= t) == y)))
    return best


def msrp_benchmark(backends):
    train, test = _read_msrp("train"), _read_msrp("test")
    y = np.array([r[0] for r in test])
    a, b = [r[1] for r in test], [r[2] for r in test]
    fit_corpus = [r[1] for r in train] + [r[2] for r in train]
    out = {}
    for name, vec in {"tfidf_char_3_5": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
                      "tfidf_word_1_2": TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)}.items():
        vec.fit(fit_corpus)
        A, B = vec.transform(a), vec.transform(b)
        s = np.asarray(A.multiply(B).sum(axis=1)).ravel()
        out[name] = {"roc_auc": float(roc_auc_score(y, s)), "best_threshold_acc": _best_acc(y, s)}
    for be in backends:
        EA, EB = embed(a, be), embed(b, be)
        s = np.sum(EA * EB, axis=1)
        out[be] = {"roc_auc": float(roc_auc_score(y, s)), "best_threshold_acc": _best_acc(y, s)}
    out["majority_class_acc"] = float(max(y.mean(), 1 - y.mean()))
    return out


def build_messages(paper: str, rng: random.Random):
    """(text, label, form, true_qid) for one committed paper."""
    papers = load_papers()
    msgs = []
    for q in papers[paper]:
        for form in ("verbatim", "excerpt", "ocr", "typed"):
            msgs.append((leak_text.RENDERERS[form](q, rng), 1, form, q["qid"]))
        auth = leak_text.authored_paraphrases().get(q["qid"])
        if auth:
            msgs.append((auth["paraphrase_en"], 1, "paraphrase (hand-written)", q["qid"]))
            msgs.append((auth["student_hinglish"], 1, "hinglish (hand-written)", q["qid"]))
    for _ in range(30):
        qs = sorted(rng.sample(papers[paper], rng.randint(4, 12)), key=lambda q: q["number"])
        msgs.append((leak_text.answer_key(qs, rng), 1, "answer_key", qs[0]["qid"]))
    # negatives: same forms, PAST papers
    for pp in past_papers():
        for q in rng.sample(papers[pp], min(10, len(papers[pp]))):
            for form in ("verbatim", "excerpt", "ocr", "typed", "paraphrase"):
                msgs.append((leak_text.RENDERERS[form](q, rng), 0, f"past paper: {form}", None))
        qs = sorted(rng.sample(papers[pp], 8), key=lambda q: q["number"])
        msgs.append((leak_text.answer_key(qs, rng), 0, "past paper: answer_key", None))
    for t in leak_text.STUDY_CHATTER + leak_text.COACHING_BROADCAST + leak_text.SCAM_POSTS + leak_text.BROKER_LINES:
        for _ in range(2):
            msgs.append((leak_text.fill(t, rng), 0, "chat / ads / scams", None))
    return msgs


def tpr_at_fpr(y, s, fpr_target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr[fpr <= fpr_target])) if np.any(fpr <= fpr_target) else 0.0


def leak_message_benchmark(backend, seed=0):
    rng = random.Random(seed)
    rows = []
    for paper in COMMITTED:
        m = PaperContentMatcher(paper, backend=backend, seed=seed + 1)   # calibration sample differs from test negatives
        msgs = build_messages(paper, rng)
        scores = m.score_messages([t for t, *_ in msgs])
        for (t, y, form, qid), r in zip(msgs, scores):
            rows.append({"y": y, "form": form, "qid": qid, **{k: r[k] for k in
                         ("lexical", "semantic", "answer_key", "score", "matched_qid", "evidence")}})
    signals = {"lexical only": "lexical", "semantic only": "semantic", "answer-key only": "answer_key", "combined": "score"}
    neg = [r for r in rows if r["y"] == 0]
    forms = sorted({r["form"] for r in rows if r["y"] == 1})
    table = defaultdict(dict)
    for form in forms + ["ALL leak forms"]:
        pos = [r for r in rows if r["y"] == 1 and (form == "ALL leak forms" or r["form"] == form)]
        for name, key in signals.items():
            y = np.array([1] * len(pos) + [0] * len(neg))
            s = np.array([r[key] for r in pos] + [r[key] for r in neg])
            table[form][name] = {"roc_auc": float(roc_auc_score(y, s)), "recall_at_1pct_fpr": tpr_at_fpr(y, s, 0.01),
                                 "n_pos": len(pos)}
        det = [r for r in pos if r["score"] > 0]
        table[form]["right_question_named"] = (float(np.mean([r["matched_qid"] == r["qid"] for r in det]))
                                              if det and form != "answer_key" else None)
    fp_by_neg = defaultdict(list)
    for r in neg:
        fp_by_neg[r["form"]].append(r["score"] > 0.5)
    table["false_positive_rate_at_0.5_by_negative_type"] = {k: float(np.mean(v)) for k, v in fp_by_neg.items()}
    return dict(table), len(rows)


def to_markdown(msrp, leak):
    lines = ["### MSRP paraphrase benchmark (test split, 1,725 pairs)", "",
             "| Representation | ROC-AUC | Best-threshold accuracy |", "|---|---|---|"]
    for k, v in msrp.items():
        if isinstance(v, dict):
            lines.append(f"| {k} | {v['roc_auc']:.3f} | {v['best_threshold_acc']:.3f} |")
    lines.append(f"| majority class | – | {msrp['majority_class_acc']:.3f} |")
    lines += ["", "### Leak-message detection vs the committed JEE 2023 papers", "",
              "| Leak form | n | Lexical AUC | Semantic AUC | Answer-key AUC | **Combined AUC** | Combined recall @1% FPR | Right question named |",
              "|---|---|---|---|---|---|---|---|"]
    for form, d in leak.items():
        if form.startswith("false_positive"):
            continue
        rq = d["right_question_named"]
        lines.append(f"| {form} | {d['combined']['n_pos']} | {d['lexical only']['roc_auc']:.3f} | {d['semantic only']['roc_auc']:.3f} | "
                     f"{d['answer-key only']['roc_auc']:.3f} | **{d['combined']['roc_auc']:.3f}** | "
                     f"{d['combined']['recall_at_1pct_fpr']:.3f} | {'–' if rq is None else f'{rq:.2f}'} |")
    lines += ["", "False-positive rate (combined score > 0.5) by type of legitimate message:", ""]
    for k, v in leak["false_positive_rate_at_0.5_by_negative_type"].items():
        lines.append(f"- {k}: {v:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    backends = [b for b in ("wordllama", "minilm") if semantic_backend_available(b)]
    print("Semantic backends available:", backends)
    msrp = msrp_benchmark(backends)
    leak, n = leak_message_benchmark(backends[0] if backends else None)
    json.dump({"msrp": msrp, "leak_messages": leak, "n_messages": n}, open(OUT / "content_benchmark.json", "w"), indent=1)
    md = to_markdown(msrp, leak)
    open(OUT / "content_benchmark.md", "w").write(md + "\n")
    print(md)
