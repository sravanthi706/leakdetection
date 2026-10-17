"""
Image-branch benchmark: which matcher recognises a photographed page of
the committed paper?

Uses the scored image pool (src/detection/image_pool.py): pages of the
four protected JEE papers photographed with realistic phone distortions,
plus clean digital copies, versus photos of PAST papers in the same
layout, coaching mock sheets and formula notes. Every image is scored
against every protected paper, so each (image, paper) pair is one test
case: positive if the image shows a page of that paper.

Compared:
  phash            global perceptual hash (the original method)
  orb_raw          ORB+RANSAC, raw inlier count only
  orb_verified     ORB+RANSAC + page-specificity + pixel agreement (used)
  combined         max(phash, orb_verified)

Reported separately for phone photos and for clean digital copies,
because the two methods fail in opposite places.

Run:  python -m src.evaluation.image_benchmark
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from src.detection.exam_corpus import TARGET_PAPERS
from src.detection.image_pool import build_image_pool
from src.evaluation.content_benchmark import tpr_at_fpr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "results"

METHODS = {"phash": "phash_score", "orb_raw": "orb_raw_score", "orb_verified": "orb_score", "combined": "score"}


def pairs(pool, positive_kind):
    ys, rows = [], []
    for r in pool:
        for t in TARGET_PAPERS:
            is_pos = r.get("paper") == t and r["kind"] == positive_kind
            is_other_form_of_pos = r.get("paper") == t and r["kind"].startswith("target") and not is_pos
            if is_other_form_of_pos:
                continue                          # evaluate photos and digital copies separately
            ys.append(int(is_pos))
            rows.append((r, r["scores"][t]))
    return np.array(ys), rows


def evaluate(pool):
    res = {}
    for kind in ("target_photo", "target_digital"):
        y, rows = pairs(pool, kind)
        res[kind] = {}
        for name, key in METHODS.items():
            s = np.array([sc[key] for _, sc in rows])
            res[kind][name] = {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
                               "recall_at_1pct_fpr": tpr_at_fpr(y, s, 0.01), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
        pos = [sc for (r, sc), yy in zip(rows, y) if yy]
        correct_page = [bool(sc["qids"]) and r["page_idx"] + 1 == sc["page"] for (r, sc), yy in zip(rows, y) if yy]
        res[kind]["correct_page_identified"] = float(np.mean(correct_page))
        res[kind]["median_inliers_positive"] = float(np.median([sc["inliers"] for sc in pos]))
    # the failure that motivated verification: raw inliers on clean pages of a DIFFERENT paper
    neg_digital = [r["scores"][t]["inliers"] for r in pool for t in TARGET_PAPERS
                   if r["kind"] == "target_digital" and r["paper"] != t]
    res["raw_inliers_on_other_papers_clean_pages"] = {"median": float(np.median(neg_digital)), "max": int(max(neg_digital))}
    return res


def to_markdown(res):
    lines = ["| Positive images | Method | ROC-AUC | PR-AUC | Recall @1% FPR |", "|---|---|---|---|---|"]
    for kind in ("target_photo", "target_digital"):
        label = "phone photos of committed pages" if kind == "target_photo" else "clean digital copies"
        for m in METHODS:
            d = res[kind][m]
            lines.append(f"| {label} (n={d['n_pos']}) | {m} | {d['roc_auc']:.3f} | {d['pr_auc']:.3f} | {d['recall_at_1pct_fpr']:.3f} |")
    lines += ["", f"Correct page identified (photos): {res['target_photo']['correct_page_identified']:.2f}; "
                  f"(digital copies): {res['target_digital']['correct_page_identified']:.2f}",
              f"Raw ORB inliers against the WRONG paper for clean pages: median "
              f"{res['raw_inliers_on_other_papers_clean_pages']['median']:.0f}, max "
              f"{res['raw_inliers_on_other_papers_clean_pages']['max']} -- why raw inlier counts are not used on their own."]
    return "\n".join(lines)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"standard_pool": evaluate(build_image_pool()),
           # HELD-OUT: new seed, harsher distortion (1.0-1.8), low-res 900 px forwards -- built after the
           # verification rule was designed, so it measures generalisation, not tuning
           "held_out_stress_pool": evaluate(build_image_pool(seed=99, stress=True))}
    json.dump(res, open(OUT / "image_benchmark.json", "w"), indent=1)
    md = ("#### Standard pool (distortion 0.6-1.35, up to 1600 px)\n\n" + to_markdown(res["standard_pool"]) +
          "\n\n#### Held-out stress pool (new seed, distortion 1.0-1.8, 900 px low-res forwards)\n\n" +
          to_markdown(res["held_out_stress_pool"]))
    open(OUT / "image_benchmark.md", "w").write(md + "\n")
    print(md)
