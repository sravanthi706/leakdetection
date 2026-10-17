"""
Report / dashboard -- item 18 of the build-order list, spec's final
"demonstrate it" piece. This does NOT reimplement any custody, crypto,
or detection logic -- it runs the real modules already built (items
1-24) for one demo pass, and renders what happened: detection scores
per session, custody-chain outcome, and (optionally) the attack-
resistance scenario results, as one PNG plus a plain-text CLI summary.

Colors follow a fixed status convention (never themed per-run):
  good (#0ca30c)     -- a correct decision / a passed check
  warning (#fab219)  -- a false positive (unnecessary alert -- costly,
                         but not dangerous)
  critical (#d03b3b) -- a false negative / a failed check (a real leak
                         missed, or an attack scenario that did NOT
                         behave as the spec requires -- the dangerous
                         failure mode, colored to stand out accordingly)
"""

import os
import time

import matplotlib
matplotlib.use("Agg")  # headless -- this runs from a CLI, not a notebook
import matplotlib.pyplot as plt

from src.orchestrator import run_custody_flow, build_trained_fusion_engine
from src.attacks.attack_harness import run_all_scenarios

GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def _classify(session_result: dict) -> str:
    """Confusion-matrix category for one scored session, used to color its bar."""
    is_leak = session_result["true_label"] == "leak"
    alert = session_result["alert"]
    if is_leak and alert:
        return "TP"
    if (not is_leak) and (not alert):
        return "TN"
    if (not is_leak) and alert:
        return "FP"
    return "FN"  # is_leak and not alert -- a real leak that was missed


_CATEGORY_COLOR = {"TP": GOOD, "TN": GOOD, "FP": WARNING, "FN": CRITICAL}


def _run_detection_pass(seed: int):
    engine, test_sessions = build_trained_fusion_engine(seed=seed, verbose=False)
    scored = [engine.score_session(s) for s in test_sessions]
    return engine, scored


def _run_custody_pass(vdf_bits: int):
    paper = b"Q1. ... [the real finalized exam paper content -- dashboard demo run] ..."
    result = run_custody_flow(
        paper, paper_id="DASHBOARD-DEMO-A1", window_start_label="2026-09-20T09:00:00Z",
        center_id="CTR-DEMO", vdf_bits=vdf_bits, verbose=False,
    )
    return result


def _print_text_dashboard(custody_result: dict, scored: list, attack_results: list):
    print("=" * 78)
    print(" AINN -- SYSTEM RUN REPORT")
    print("=" * 78)

    print("\n[Custody chain]")
    print(f"  9-step flow completed without error : True")
    print(f"  Plaintext matches original          : {custody_result['matches_original']}")
    print(f"  Shares released                     : {custody_result['released_count']}/3 required")
    print(f"  Commitment ledger valid             : {custody_result['ledger_valid']}")

    print("\n[Detection layer -- held-out test sessions]")
    tp = sum(1 for r in scored if _classify(r) == "TP")
    tn = sum(1 for r in scored if _classify(r) == "TN")
    fp = sum(1 for r in scored if _classify(r) == "FP")
    fn = sum(1 for r in scored if _classify(r) == "FN")
    n = len(scored)
    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    print(f"  n={n}  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Accuracy={accuracy:.1%}  Precision={precision:.1%}  Recall={recall:.1%}")
    if fn > 0:
        print(f"  NOTE: {fn} real leak session(s) were NOT alerted on -- see FN rows in the plot.")

    print("\n[Attack-resistance scenarios -- spec section C.7]")
    n_pass = sum(1 for r in attack_results if r.passed)
    for r in attack_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.scenario}")
    print(f"  {n_pass}/{len(attack_results)} scenarios passed")

    print("\n" + "=" * 78)


def generate_dashboard(output_dir: str = "reports", seed: int = 42,
                        vdf_bits: int = 12, run_attack_scenarios: bool = True) -> str:
    """
    Runs one full demo pass (custody chain + detection layer + attack
    scenarios) against the real modules, prints a CLI summary, renders
    one PNG, and returns the PNG's path.
    """
    t_start = time.time()

    custody_result = _run_custody_pass(vdf_bits)
    engine, scored = _run_detection_pass(seed)
    attack_results = run_all_scenarios(seed=seed + 1, verbose=False) if run_attack_scenarios else []

    _print_text_dashboard(custody_result, scored, attack_results)

    # --- Figure: 3 stacked panels sharing one figure, none sharing an axis ---
    scored_sorted = sorted(scored, key=lambda r: r["combined_score"])
    n = len(scored_sorted)
    categories = [_classify(r) for r in scored_sorted]
    colors = [_CATEGORY_COLOR[c] for c in categories]
    scores = [r["combined_score"] for r in scored_sorted]
    threshold = engine.threshold

    fig = plt.figure(figsize=(11, 8.5), facecolor=SURFACE)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[3.2, 0.9, 0.9], hspace=0.55)

    # Panel 1: detection scores, one bar per held-out session
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(SURFACE)
    ax1.bar(range(n), scores, color=colors, width=0.85, linewidth=0)
    ax1.axhline(threshold, color=INK_SECONDARY, linestyle="--", linewidth=1.2)
    ax1.text(-0.5, threshold + 0.02, f"alert threshold ({threshold:.2f})", color=INK_SECONDARY,
              fontsize=9, va="bottom", ha="left")
    ax1.set_xlim(-1, n)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks([])
    ax1.set_ylabel("combined anomaly score", color=INK_SECONDARY, fontsize=10)
    ax1.set_title("Detection layer -- combined score per held-out session", color=INK_PRIMARY,
                   fontsize=12, loc="left", pad=10)
    for spine in ("top", "right", "bottom"):
        ax1.spines[spine].set_visible(False)
    ax1.spines["left"].set_color(GRIDLINE)
    ax1.tick_params(colors=INK_MUTED, labelsize=9)
    ax1.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax1.set_axisbelow(True)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=GOOD, label="correct (TP / TN)"),
        plt.Rectangle((0, 0), 1, 1, color=WARNING, label="false positive"),
        plt.Rectangle((0, 0), 1, 1, color=CRITICAL, label="false negative (missed leak)"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    # Panel 2: custody chain outcome, as status chips
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(SURFACE)
    ax2.set_title("Custody chain (this run)", color=INK_PRIMARY, fontsize=11, loc="left")
    custody_chips = [
        ("9 steps completed", True),
        ("plaintext matches", custody_result["matches_original"]),
        (f"shares {custody_result['released_count']}/3", custody_result["released_count"] >= 3),
        ("ledger valid", custody_result["ledger_valid"]),
    ]
    _draw_chip_row(ax2, custody_chips)

    # Panel 3: attack-resistance scenario outcomes, as status chips
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(SURFACE)
    if attack_results:
        ax3.set_title("Attack-resistance scenarios (spec C.7)", color=INK_PRIMARY, fontsize=11, loc="left")
        attack_chips = [(r.scenario, r.passed) for r in attack_results]
        _draw_chip_row(ax3, attack_chips, small=True)
    else:
        ax3.axis("off")

    fig.suptitle("AINN -- tamper-evident exam distribution + AI/NN leak detection: system run report",
                  color=INK_PRIMARY, fontsize=13, x=0.01, ha="left", y=0.985)
    fig.text(0.01, 0.005, f"generated {time.strftime('%Y-%m-%d %H:%M:%S')} UTC  |  "
                           f"total run time {time.time() - t_start:.1f}s", color=INK_MUTED, fontsize=8)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"dashboard_{int(time.time())}.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDashboard image saved: {out_path}")
    return out_path


def _draw_chip_row(ax, chips, small: bool = False):
    """Draws a horizontal row of pass/fail status chips (colored dot + label)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(chips)
    if n == 0:
        return
    font_size = 8 if small else 9.5
    cols = min(n, 4) if small else n
    rows = -(-n // cols)  # ceil
    x_positions = [(i % cols) / cols + 0.02 for i in range(n)]
    y_positions = [1 - 0.5 / rows - (i // cols) * (1 / rows) for i in range(n)]
    for (label, ok), x, y in zip(chips, x_positions, y_positions):
        color = GOOD if ok else CRITICAL
        ax.scatter([x], [y], s=70 if not small else 45, color=color, zorder=3)
        ax.text(x + 0.018, y, label, va="center", ha="left", color=INK_SECONDARY, fontsize=font_size)


if __name__ == "__main__":
    generate_dashboard()
