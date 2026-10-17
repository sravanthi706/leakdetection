"""
Figures for the report / viva slides, drawn from the JSON results in
reports/results/. Run after the benchmarks:

    python -m src.evaluation.plots

Style: one light surface, thin marks, a single highlighted series (AINN)
against neutral grey comparisons, text in ink colours rather than series
colours, one y-axis per chart.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "reports" / "results"
FIG = ROOT / "reports" / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                            "#e87ba4", "#008300", "#4a3aa7", "#e34948")
NEUTRAL = "#c3c2b7"


def _ax(figsize=(9, 4.5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    return fig, ax


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print("  wrote", (FIG / name).relative_to(ROOT))


def _hbar(ax, names, means, stds, highlight):
    y = np.arange(len(names))[::-1]
    colors = [BLUE if h else NEUTRAL for h in highlight]
    ax.barh(y, means, xerr=stds, color=colors, height=0.62, error_kw={"ecolor": MUTED, "lw": 1, "capsize": 2})
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK2, fontsize=9)
    for yi, m, sd in zip(y, means, stds):
        ax.text(m + sd + 0.015, yi, f"{m:.2f}", va="center", fontsize=8.5, color=INK2)   # label clears the error bar
    ax.set_xlim(0, 1.08)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def session_figures(tag=""):
    f = RES / f"session_benchmark{tag}.json"
    if not f.exists():
        return
    S = json.load(open(f))["summary"]
    t = S["table"]
    main = ["AINN (learned fusion)", "AINN (learned fusion, full PyTorch)", "equal-weight fusion (original rule)",
            "baseline: text_tfidf_lr", "baseline: features_rf", "baseline: features_lr", "baseline: keyword_rule"]
    main = [m for m in main if m in t]
    fig, ax = _ax((9, 4))
    _hbar(ax, main, [t[m]["f1"]["mean"] for m in main], [t[m]["f1"]["std"] for m in main],
          [m.startswith("AINN") for m in main])
    ax.set_xlabel("F1 on an exam paper never seen in training (mean ± std over folds)", color=INK2)
    ax.set_title("AINN vs baselines -- leave-one-paper-out", loc="left", color=INK)
    _save(fig, "session_models_f1.png")

    abl = ["AINN (learned fusion)", "without content", "without image", "without gnn", "without autoencoder",
           "single: content", "single: image", "single: gnn", "single: autoencoder",
           "AINN with 10% noisy training labels", "AINN with 20% noisy training labels"]
    abl = [a for a in abl if a in t]
    fig, ax = _ax((9, 4.6))
    _hbar(ax, abl, [t[a]["f1"]["mean"] for a in abl], [t[a]["f1"]["std"] for a in abl], [a == abl[0] for a in abl])
    ax.set_xlabel("F1 (mean ± std over folds)", color=INK2)
    ax.set_title("Ablations: which signals carry the detection?", loc="left", color=INK)
    _save(fig, "session_ablations_f1.png")

    ps = S["per_scenario_alert_rate"]
    models = [m for m in ["AINN (learned fusion)", f"baseline: {S['best_baseline'].split(': ')[-1]}"] if m in ps]
    scen = list(ps[models[0]].keys())
    fig, ax = _ax((10, 4.6))
    x = np.arange(len(scen))
    w = 0.38
    for i, (m, c) in enumerate(zip(models, [BLUE, ORANGE])):
        ax.bar(x + (i - 0.5) * w, [ps[m][s] for s in scen], width=w - 0.03, color=c, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in scen], rotation=35, ha="right", fontsize=8.5, color=INK2)
    ax.set_ylabel("alert rate", color=INK2)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Per scenario: leak rows should be high (recall), normal rows low (false alarms)", loc="left", color=INK, fontsize=11)
    _save(fig, "session_per_scenario.png")

    oof_f = RES / f"session_benchmark_oof{tag}.json"
    if oof_f.exists():
        oof = json.load(open(oof_f))
        y = np.array([r["y"] for r in oof])
        keys = [k[2:] for k in oof[0] if k.startswith("p_")]
        order = [k for k in main if k in keys]
        palette = {order[0]: BLUE}
        others = [ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, GREEN]
        for k in order[1:]:
            palette[k] = others.pop(0)
        for kind in ("roc", "pr"):
            fig, ax = _ax((6.2, 5))
            for k in order:
                p = np.array([r[f"p_{k}"] for r in oof])
                if kind == "roc":
                    fx, fy, _ = roc_curve(y, p)
                else:
                    fy, fx, _ = precision_recall_curve(y, p)
                ax.plot(fx, fy, color=palette[k], lw=2.2 if k.startswith("AINN") else 1.4, label=k)
            ax.set_xlabel("false-positive rate" if kind == "roc" else "recall", color=INK2)
            ax.set_ylabel("true-positive rate" if kind == "roc" else "precision", color=INK2)
            ax.grid(color=GRID, lw=0.8)
            ax.legend(frameon=False, fontsize=8, loc="lower right" if kind == "roc" else "lower left")
            ax.set_title(("ROC" if kind == "roc" else "Precision-recall") + " curves, pooled out-of-fold", loc="left", color=INK)
            _save(fig, f"session_{kind}_curves.png")

    cal = S["calibration"]["bins"]
    fig, ax = _ax((5.4, 5))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--")
    ax.plot([b["mean_pred"] for b in cal], [b["frac_leak"] for b in cal], color=BLUE, lw=2, marker="o", ms=6)
    ax.set_xlabel("predicted P(leak)", color=INK2)
    ax.set_ylabel("observed fraction of leaks", color=INK2)
    ax.grid(color=GRID, lw=0.8)
    ax.set_title(f"Calibration (ECE = {S['calibration']['ece']:.3f})", loc="left", color=INK)
    _save(fig, "session_calibration.png")

    tt = S["time_to_detection"]
    det = tt["detected_by_scenario"]
    fig, ax = _ax((8.5, 4))
    names = list(det)
    _hbar(ax, [n.replace("_", " ") for n in names], [det[n] for n in names], [0] * len(names), [True] * len(names))
    ax.set_xlabel("share of leak sessions flagged before the exam (streaming)", color=INK2)
    ax.set_title(f"Streaming detection -- median delay after first leaked item: "
                 f"{tt['median_delay_after_first_leaked_item_h']:.1f} h", loc="left", color=INK, fontsize=11)
    _save(fig, "session_streaming_detection.png")


def image_figure():
    f = RES / "image_benchmark.json"
    if not f.exists():
        return
    R = json.load(open(f))
    methods = ["phash", "orb_raw", "orb_verified"]
    labels = ["pHash (global)", "ORB raw inliers", "ORB verified"]
    pools = [("standard_pool", "standard photos"), ("held_out_stress_pool", "held-out harsh photos")]
    fig, ax = _ax((8, 4))
    x = np.arange(len(methods))
    w = 0.38
    for i, ((key, name), c) in enumerate(zip(pools, [BLUE, ORANGE])):
        vals = [R[key]["target_photo"][m]["recall_at_1pct_fpr"] for m in methods]
        ax.bar(x + (i - 0.5) * w, vals, width=w - 0.03, color=c, label=name)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK2)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("recall of photographed committed pages\nat 1% false-positive rate", color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.set_title("Image branch: phone photos defeat pHash; verified ORB recovers them", loc="left", color=INK, fontsize=11)
    _save(fig, "image_branch.png")


def content_figure():
    f = RES / "content_benchmark.json"
    if not f.exists():
        return
    L = json.load(open(f))["leak_messages"]
    forms = [k for k in L if not k.startswith("false_positive") and k != "ALL leak forms"]
    forms.sort(key=lambda k: -L[k]["combined"]["recall_at_1pct_fpr"])
    fig, ax = _ax((8.5, 4))
    _hbar(ax, forms, [L[k]["combined"]["recall_at_1pct_fpr"] for k in forms], [0] * len(forms), [True] * len(forms))
    ax.set_xlabel("recall at 1% false-positive rate (negatives = past papers, chat, ads, scams)", color=INK2)
    ax.set_title("Content branch: how each leak form is caught", loc="left", color=INK)
    _save(fig, "content_branch.png")


def twitter_figure():
    f = RES / "twitter_benchmark.json"
    if not f.exists():
        return
    R = json.load(open(f))
    fig, axes = plt.subplots(1, len(R), figsize=(5.2 * len(R), 4), facecolor=SURFACE, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, R):
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ms = list(r["models"])
        _hbar(ax, ms, [r["models"][m]["acc_mean"] for m in ms], [r["models"][m]["acc_std"] for m in ms],
              [m == "bigcn" for m in ms])
        ax.axvline(r["chance_acc"], color=MUTED, ls="--", lw=1)
        ax.text(r["chance_acc"] + 0.01, -0.6, "chance", fontsize=8, color=MUTED)
        ax.set_title(f"{r['dataset']} (real cascades, n={r['n_samples']})", loc="left", color=INK, fontsize=11)
        ax.set_xlabel("4-class accuracy, 5-fold CV", color=INK2)
    _save(fig, "twitter_benchmark.png")


if __name__ == "__main__":
    # full-run results if present, otherwise the --quick run's
    session_figures("" if (RES / "session_benchmark.json").exists() else "_quick")
    image_figure()
    content_figure()
    twitter_figure()
