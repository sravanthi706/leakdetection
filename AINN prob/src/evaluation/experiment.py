"""
Main session-level benchmark for the AINN detection layer.

PROTOCOL -- designed to avoid the ways ML results usually fool people:

1. Hard data (realistic_data.py): 25% leaks, deliberate look-alikes on
   both sides (scams, coaching broadcasts, past-paper discussion vs quiet,
   buried, paraphrased, answer-key-only and coordination-only leaks).

2. LEAVE-ONE-PAPER-OUT cross-validation. Every session belongs to one
   upcoming exam (one of 4 real JEE papers). Each fold tests on the
   sessions of a paper that NO model saw during training. This matters:
   in a first pilot with ordinary random folds, a plain supervised text
   classifier scored ROC-AUC 1.00 simply by memorising the words of the
   four papers -- impossible in reality, where this year's paper has
   never been seen before it leaks. Holding the paper out removes that
   cheat; AINN's content branch doesn't need it anyway, because it
   compares messages with the COMMITTED paper at run time.

3. Base models are trained on the training papers; the fusion weights
   and the alert threshold are fitted on a validation split of the
   training papers. The test paper is only ever used for scoring.

4. Repeated over 2 independently generated datasets -> 8 folds; every
   metric is reported as mean +- std, with a bootstrap 95% CI for the
   headline model and a paired comparison against the best baseline.

5. Ablations (each signal alone, each branch removed), label-noise
   robustness (10% / 20% of training labels flipped), per-scenario
   recall and false-alarm rates, calibration, and time-to-detection
   (how many hours BEFORE the exam each leak is flagged, re-scoring the
   session as it grows).

Run:  python -m src.evaluation.experiment            (full, ~30-40 min on a laptop CPU)
      python -m src.evaluation.experiment --quick    (1 dataset, smaller, no PyTorch GNN)
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

from src.detection.engine import LeakDetector, annotate_events, best_f1_threshold, BRANCHES
from src.detection.realistic_data import generate_hard_dataset, LEAK_MIX, NORMAL_MIX
from src.detection.exam_corpus import TARGET_PAPERS
from src.evaluation.baselines import ALL_BASELINES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "results"

GROUPS = {"gnn": ["gnn"], "autoencoder": ["autoencoder"],
          "content": ["content_lexical", "content_semantic", "content_answer_key"],
          "image": ["image_orb"]}
CHECKPOINTS_H = [240, 216, 192, 168, 144, 120, 96, 72, 60, 48, 36, 24, 18, 12, 6, 3, 0]


def metrics(y, p, thr):
    pred = p >= thr
    return {"precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)), "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)), "accuracy": float(accuracy_score(y, pred))}


def meta_on_columns(Xv, yv, Xt, cols, seed):
    """Stacking meta-model restricted to a subset of signals (for ablations); threshold via CV on validation."""
    sc = StandardScaler().fit(Xv[:, cols])
    Zv, Zt = sc.transform(Xv[:, cols]), sc.transform(Xt[:, cols])
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Zv, yv)
    p_cv = cross_val_predict(LogisticRegression(class_weight="balanced", max_iter=2000), Zv, yv,
                             cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return lr.predict_proba(Zt)[:, 1], best_f1_threshold(yv, p_cv)


def equal_weight_score(X, branches):
    """The ORIGINAL fusion rule (0.25 per branch), applied to the new signals for a like-for-like comparison."""
    ix = {b: i for i, b in enumerate(branches)}
    gnn = X[:, ix["gnn"]]
    ae = 1 - np.exp(-np.maximum(0, X[:, ix["autoencoder"]]))
    content = X[:, [ix[b] for b in GROUPS["content"]]].max(axis=1)
    image = X[:, [ix[b] for b in GROUPS["image"]]].max(axis=1)
    return 0.25 * (gnn + ae + content + image)


def prefix(session, hours_before):
    cut = session["window_start"] - hours_before * 3600
    return dict(session, events=[e for e in session["events"] if e["timestamp"] <= cut])


def time_to_detection(det, test_sessions):
    """
    Streaming evaluation: re-score every test session at each checkpoint
    (240 h ... 0 h before the exam) using only the messages seen so far.
      * normal session alerting at ANY checkpoint      -> streaming false alarm
      * leak session alerting BEFORE any leak activity -> premature (false) alarm
      * otherwise: hours before the exam of the first alert, and delay after
        the first leaked item appeared.
    """
    batch, meta = [], []
    for s in test_sessions:
        for h in CHECKPOINTS_H:
            p = prefix(s, h)
            batch.append(p)
            meta.append(any(e["is_leak_event"] for e in p["events"]))
    P = det.predict_proba(batch).reshape(len(test_sessions), len(CHECKPOINTS_H))
    has = np.array(meta).reshape(len(test_sessions), len(CHECKPOINTS_H))
    leaks, normals = [], []
    for s, row, hrow in zip(test_sessions, P, has):
        alerts = row >= det.threshold
        if s["label"] == "normal":
            normals.append(bool(alerts.any()))
            continue
        first_leak_h = (s["window_start"] - min(e["timestamp"] for e in s["events"] if e["is_leak_event"])) / 3600
        premature = bool(np.any(alerts & ~hrow))
        valid = [h for h, a, hl in zip(CHECKPOINTS_H, alerts, hrow) if a and hl]
        leaks.append({"scenario": s["scenario"], "first_leak_h_before_exam": float(first_leak_h),
                      "first_alert_h_before_exam": valid[0] if valid else None,
                      "delay_h": (float(first_leak_h - valid[0]) if valid else None), "premature_alarm": premature})
    return leaks, normals


def run_fold(ds_seed, paper, sessions, use_torch, log):
    test = [s for s in sessions if s["exam_paper"] == paper]
    train_all = [s for s in sessions if s["exam_paper"] != paper]
    strat = [s["scenario"] for s in train_all]
    train, val = train_test_split(train_all, test_size=0.2, random_state=ds_seed, stratify=strat)
    yv = np.array([s["label"] == "leak" for s in val], dtype=int)
    yt = np.array([s["label"] == "leak" for s in test], dtype=int)
    res, probs = {}, {}

    t0 = time.time()
    det = LeakDetector(gnn_backend="numpy", ae_backend="numpy", seed=ds_seed).fit(train, val)
    Xv, Xt = det.branch_matrix(val), det.branch_matrix(test)
    p = det.predict_proba(test)
    res["AINN (learned fusion)"] = metrics(yt, p, det.threshold)
    probs["AINN (learned fusion)"] = (p, det.threshold)
    weights = det.weights()
    log(f"    AINN numpy fitted in {time.time() - t0:.0f}s  F1={res['AINN (learned fusion)']['f1']:.3f}")

    if use_torch:
        t0 = time.time()
        det_t = LeakDetector(gnn_backend="torch", ae_backend="torch", seed=ds_seed).fit(train, val)
        p_t = det_t.predict_proba(test)
        res["AINN (learned fusion, full PyTorch)"] = metrics(yt, p_t, det_t.threshold)
        probs["AINN (learned fusion, full PyTorch)"] = (p_t, det_t.threshold)
        res["single: gnn (PyTorch)"] = metrics(yt, det_t.branch_matrix(test)[:, 0], best_f1_threshold(yv, det_t.branch_matrix(val)[:, 0]))
        log(f"    AINN torch fitted in {time.time() - t0:.0f}s  F1={res['AINN (learned fusion, full PyTorch)']['f1']:.3f}")

    # the original fixed-weight fusion rule, on the same signals
    sv, st = equal_weight_score(Xv, det.branches), equal_weight_score(Xt, det.branches)
    res["equal-weight fusion (original rule)"] = metrics(yt, st, best_f1_threshold(yv, sv))

    # ablations
    ix = {b: i for i, b in enumerate(det.branches)}
    for g, members in GROUPS.items():
        cols = [ix[m] for m in members]
        sv_g, st_g = Xv[:, cols].max(axis=1), Xt[:, cols].max(axis=1)
        res[f"single: {g}"] = metrics(yt, st_g, best_f1_threshold(yv, sv_g))
        keep = [ix[b] for b in det.branches if b not in members]
        pt, thr = meta_on_columns(Xv, yv, Xt, keep, ds_seed)
        res[f"without {g}"] = metrics(yt, pt, thr)

    # baselines
    for B in ALL_BASELINES:
        m = B().fit(train, val)
        pb = m.predict_proba(test)
        res[f"baseline: {m.name}"] = metrics(yt, pb, m.threshold)
        probs[f"baseline: {m.name}"] = (pb, m.threshold)

    # label-noise robustness
    for rate in (0.1, 0.2):               # noise in BOTH training and validation labels, as it would be in reality
        rng = np.random.default_rng(ds_seed * 7 + int(rate * 100))
        y_tr = np.array([s["label"] == "leak" for s in train], dtype=int)
        f_tr, f_va = rng.random(len(y_tr)) < rate, rng.random(len(yv)) < rate
        d_n = LeakDetector(gnn_backend="numpy", ae_backend="numpy", seed=ds_seed).fit(train, val, train_labels=np.where(f_tr, 1 - y_tr, y_tr),
                                                                  val_labels=np.where(f_va, 1 - yv, yv))
        res[f"AINN with {int(rate * 100)}% noisy training labels"] = metrics(yt, d_n.predict_proba(test), d_n.threshold)

    # per-scenario behaviour of the headline model and the strongest baselines
    per_scen = defaultdict(dict)
    for name, (pp, thr) in probs.items():
        for scen in list(LEAK_MIX) + list(NORMAL_MIX):
            idx = [i for i, s in enumerate(test) if s["scenario"] == scen]
            if idx:
                per_scen[name][scen] = {"alert_rate": float(np.mean(pp[idx] >= thr)), "n": len(idx)}

    lead, ever = time_to_detection(det, test)
    oof = [{"y": int(y), "p": float(pp), "thr": probs["AINN (learned fusion)"][1], "scenario": s["scenario"],
            **{f"p_{k}": float(v[0][i]) for k, v in probs.items()}, **{f"thr_{k}": v[1] for k, v in probs.items()}}
           for i, (s, y, pp) in enumerate(zip(test, yt, probs["AINN (learned fusion)"][0]))]
    return {"dataset_seed": ds_seed, "test_paper": paper, "n_test": len(test), "n_test_leak": int(yt.sum()),
            "metrics": res, "per_scenario": per_scen, "fusion_weights": {k: float(v) for k, v in weights.items()},
            "lead_time": lead, "normal_ever_alerted": ever, "oof": oof}


def bootstrap(oof, key_p, key_thr, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([r["y"] for r in oof])
    p = np.array([r[key_p] for r in oof])
    pred = np.array([r[key_p] >= r[key_thr] for r in oof])
    f1s, prs = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        f1s.append(f1_score(y[i], pred[i]))
        prs.append(average_precision_score(y[i], p[i]))
    return {"f1": float(f1_score(y, pred)), "f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
            "pr_auc": float(average_precision_score(y, p)), "pr_auc_ci95": [float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5))]}


def paired_f1_diff(oof, a, b, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    y = np.array([r["y"] for r in oof])
    pa = np.array([r[f"p_{a}"] >= r[f"thr_{a}"] for r in oof])
    pb = np.array([r[f"p_{b}"] >= r[f"thr_{b}"] for r in oof])
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(f1_score(y[i], pa[i]) - f1_score(y[i], pb[i]))
    return {"mean_diff": float(np.mean(diffs)), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_value_one_sided": float(np.mean(np.array(diffs) <= 0))}


def calibration(oof, bins=10):
    y = np.array([r["y"] for r in oof])
    p = np.array([r["p"] for r in oof])
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            rows.append({"bin": [float(lo), float(hi)], "mean_pred": float(p[m].mean()), "frac_leak": float(y[m].mean()), "n": int(m.sum())})
            ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return {"bins": rows, "ece": float(ece)}


def summarise(folds):
    names = sorted({k for f in folds for k in f["metrics"]})
    table = {}
    for n in names:
        vals = [f["metrics"][n] for f in folds if n in f["metrics"]]
        table[n] = {m: {"mean": float(np.mean([v[m] for v in vals])), "std": float(np.std([v[m] for v in vals]))}
                    for m in vals[0]}
    oof = [r for f in folds for r in f["oof"]]
    base_names = [n for n in names if n.startswith("baseline:")]
    best_base = max(base_names, key=lambda n: table[n]["f1"]["mean"])
    lead = [x for f in folds for x in f["lead_time"]]
    first = [x["first_alert_h_before_exam"] for x in lead]
    detected = [h for h in first if h is not None]
    delays = [x["delay_h"] for x in lead if x["delay_h"] is not None]
    per_scen = defaultdict(lambda: defaultdict(list))
    for f in folds:
        for model, d in f["per_scenario"].items():
            for scen, v in d.items():
                per_scen[model][scen].append(v["alert_rate"])
    weights = defaultdict(list)
    for f in folds:
        for k, v in f["fusion_weights"].items():
            weights[k].append(v)
    lead_by_scen = defaultdict(list)
    for x in lead:
        lead_by_scen[x["scenario"]].append(x)
    return {
        "table": table,
        "headline_bootstrap": bootstrap(oof, "p", "thr"),
        "best_baseline": best_base,
        "best_baseline_bootstrap": bootstrap(oof, f"p_{best_base}", f"thr_{best_base}"),
        "ainn_minus_best_baseline_f1": paired_f1_diff(oof, "AINN (learned fusion)", best_base),
        "calibration": calibration(oof),
        "fusion_weights_mean": {k: float(np.mean(v)) for k, v in weights.items()},
        "fusion_weights_std": {k: float(np.std(v)) for k, v in weights.items()},
        "per_scenario_alert_rate": {m: {s: float(np.mean(v)) for s, v in d.items()} for m, d in per_scen.items()},
        "time_to_detection": {
            "n_leaks": len(lead),
            "detected_before_exam": len(detected) / max(len(lead), 1),
            "detected_24h_or_more_before_exam": float(np.mean([h is not None and h >= 24 for h in first])) if first else None,
            "median_hours_before_exam": float(np.median(detected)) if detected else None,
            "median_delay_after_first_leaked_item_h": float(np.median(delays)) if delays else None,
            "leak_sessions_with_premature_alarm": float(np.mean([x["premature_alarm"] for x in lead])) if lead else None,
            "normal_sessions_alerting_at_any_checkpoint": float(np.mean([x for f in folds for x in f["normal_ever_alerted"]])),
            "detected_by_scenario": {k: float(np.mean([x["first_alert_h_before_exam"] is not None for x in v])) for k, v in lead_by_scen.items()},
            "median_delay_h_by_scenario": {k: (float(np.median([x["delay_h"] for x in v if x["delay_h"] is not None]))
                                               if any(x["delay_h"] is not None for x in v) else None) for k, v in lead_by_scen.items()},
        },
        "n_folds": len(folds), "n_test_sessions_total": sum(f["n_test"] for f in folds),
    }


ORDER = ["AINN (learned fusion)", "AINN (learned fusion, full PyTorch)", "equal-weight fusion (original rule)",
         "baseline: text_tfidf_lr", "baseline: features_rf", "baseline: features_lr", "baseline: keyword_rule",
         "single: content", "single: image", "single: gnn", "single: gnn (PyTorch)", "single: autoencoder",
         "without content", "without image", "without gnn", "without autoencoder",
         "AINN with 10% noisy training labels", "AINN with 20% noisy training labels"]


def to_markdown(S):
    t = S["table"]
    lines = ["| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|"]
    for n in ORDER + [k for k in t if k not in ORDER]:
        if n in t:
            r = t[n]
            fmt = lambda m: f"{r[m]['mean']:.3f} ± {r[m]['std']:.3f}"
            name = f"**{n}**" if n.startswith("AINN (learned fusion)") else n
            lines.append(f"| {name} | {fmt('f1')} | {fmt('precision')} | {fmt('recall')} | {fmt('roc_auc')} | {fmt('pr_auc')} |")
    hb, bb, d = S["headline_bootstrap"], S["best_baseline_bootstrap"], S["ainn_minus_best_baseline_f1"]
    lines += ["", f"Pooled out-of-fold (n={S['n_test_sessions_total']}, {S['n_folds']} leave-one-paper-out folds): "
                  f"AINN F1 = {hb['f1']:.3f} (95% CI {hb['f1_ci95'][0]:.3f}-{hb['f1_ci95'][1]:.3f}), "
                  f"PR-AUC = {hb['pr_auc']:.3f} (95% CI {hb['pr_auc_ci95'][0]:.3f}-{hb['pr_auc_ci95'][1]:.3f}).",
              f"Best baseline ({S['best_baseline']}): F1 = {bb['f1']:.3f} (95% CI {bb['f1_ci95'][0]:.3f}-{bb['f1_ci95'][1]:.3f}).",
              f"Paired bootstrap F1 difference AINN - best baseline: {d['mean_diff']:+.3f} "
              f"(95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}, one-sided p = {d['p_value_one_sided']:.4f}).",
              f"Calibration: expected calibration error = {S['calibration']['ece']:.3f}.", "",
              "Learned fusion weights (standardised log-odds per signal, mean ± std over folds):", ""]
    for k in BRANCHES:
        if k in S["fusion_weights_mean"]:
            lines.append(f"- {k}: {S['fusion_weights_mean'][k]:+.2f} ± {S['fusion_weights_std'][k]:.2f}")
    lines += ["", "Alert rate per scenario (leak rows = recall, normal rows = false-alarm rate):", "",
              "| Scenario | " + " | ".join(S["per_scenario_alert_rate"]) + " |",
              "|---|" + "---|" * len(S["per_scenario_alert_rate"])]
    models = list(S["per_scenario_alert_rate"])
    for scen in list(LEAK_MIX) + list(NORMAL_MIX):
        tag = "leak" if scen in LEAK_MIX else "normal"
        lines.append(f"| {scen} ({tag}) | " + " | ".join(f"{S['per_scenario_alert_rate'][m].get(scen, float('nan')):.2f}" for m in models) + " |")
    tt = S["time_to_detection"]
    lines += ["", "Streaming time-to-detection (each session re-scored at 17 checkpoints, 240 h -> 0 h before the exam, "
                  "using only messages seen so far):", "",
              f"- leaks flagged before the exam starts: {tt['detected_before_exam']:.1%}",
              f"- flagged at least 24 h before the exam: {tt['detected_24h_or_more_before_exam']:.1%}",
              f"- median lead time of flagged leaks: {tt['median_hours_before_exam']} h before the exam",
              f"- median delay between the first leaked item appearing and the alert: {tt['median_delay_after_first_leaked_item_h']} h",
              f"- normal sessions raising an alert at ANY checkpoint (streaming false-alarm rate): "
              f"{tt['normal_sessions_alerting_at_any_checkpoint']:.1%}",
              f"- leak sessions alerting before any leak activity existed (premature alarm): {tt['leak_sessions_with_premature_alarm']:.1%}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-torch", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [0] if args.quick else [0, 1]
    n = 600 if args.quick else 1200
    use_torch = not (args.no_torch or args.quick)
    if use_torch:
        try:
            import torch_geometric  # noqa: F401
        except ImportError:
            print("PyTorch Geometric not installed -- running without the PyTorch GNN.")
            use_torch = False
    log = lambda m: print(m, flush=True)
    folds = []
    t_all = time.time()
    for seed in seeds:
        ds = generate_hard_dataset(n_sessions=n, leak_fraction=0.25, seed=seed)
        pool = {r["image_id"]: r for r in ds["pool"]}
        t0 = time.time()
        annotate_events(ds["sessions"], pool)
        log(f"dataset seed {seed}: {n} sessions, evidence annotated in {time.time() - t0:.0f}s")
        for paper in TARGET_PAPERS:
            log(f"  fold: test paper = {paper}")
            folds.append(run_fold(seed, paper, ds["sessions"], use_torch, log))
    S = summarise(folds)
    S["runtime_minutes"] = (time.time() - t_all) / 60
    tag = "_quick" if args.quick else ""
    json.dump({"summary": S, "folds": [{k: v for k, v in f.items() if k != "oof"} for f in folds]},
              open(OUT / f"session_benchmark{tag}.json", "w"), indent=1)
    json.dump([r for f in folds for r in f["oof"]], open(OUT / f"session_benchmark_oof{tag}.json", "w"))
    md = to_markdown(S)
    open(OUT / f"session_benchmark{tag}.md", "w").write(md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
