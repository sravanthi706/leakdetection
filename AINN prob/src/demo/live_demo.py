"""
LIVE DEMO -- a real question paper, a leak unfolding message by message,
and an explainable alert.

What happens, end to end:

  1. A REAL question paper (JEE Advanced 2023 Paper 1, from JEEBench) is
     committed to the co-signed commitment ledger before the exam:
     hash(paper) and hash(answer key), signed by all four stakeholders.
     This is the custody layer's ledger (src/custody/commitment_ledger.py)
     -- the same commitment the detection layer now matches against.

  2. The detector is trained WITHOUT this paper (only sessions about the
     other exams), exactly as it would be in reality: nobody gets to train
     on this year's paper before it leaks.

  3. A monitored channel is replayed in time order, 10 days before the
     exam. After every message the whole channel so far is re-scored.
     Messages or photos that match the committed paper are flagged as
     they arrive, and P(leak) is shown as it rises.

  4. When P(leak) crosses the alert threshold, the alert prints its
     evidence: which message matched which committed question, which photo
     matched which committed page (with the geometric-match count), how
     much each signal contributed, and -- with PyTorch installed -- which
     accounts the graph model relied on. It also shows the forensic check:
     the matched questions really are in the paper whose hash was
     committed before the exam.

  5. The same is done for a scam channel ("100% genuine paper, pay
     Rs 5000") that looks like a leak but carries no committed content,
     to show it does NOT alert.

Usage (from the project root):
    python -m src.demo.live_demo                         # full_leak vs scam, paced replay
    python -m src.demo.live_demo --scenario photo_leak   # any leak scenario from realistic_data.py
    python -m src.demo.live_demo --fast                  # no pauses
    python -m src.demo.live_demo --chat my_chat.txt      # score YOUR OWN messages (see below)

--chat file format, one message per line:
    HH:MM | group | sender | message text
    HH:MM | group | sender | IMG: path/to/photo.jpg
(times are read as "on the day before the exam")
"""

import argparse
import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

from src.custody.commitment_ledger import CommitmentLedger, generate_stakeholder_keypair, hash_paper
from src.detection.engine import LeakDetector, annotate_events, format_explanation
from src.detection.exam_corpus import load_papers, TARGET_PAPERS
from src.detection.image_pool import build_image_pool
from src.detection.realistic_data import generate_hard_dataset, generate_session, LEAK_MIX

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
FIG = ROOT / "reports" / "figures"
STAKEHOLDERS = ["NTA_HQ", "STATE_AUTHORITY", "INDEPENDENT_AUDIT", "CUSTODIAN"]


def commit_paper(paper_name: str):
    """Pre-exam commitment of the real paper + its answer key, co-signed by all 4 stakeholders."""
    qs = load_papers()[paper_name]
    paper_bytes = "\n".join(q["raw"] for q in qs).encode()
    key_bytes = json.dumps({q["number"]: q["gold"] for q in qs}, sort_keys=True).encode()
    keys = {sid: generate_stakeholder_keypair() for sid in STAKEHOLDERS}
    ledger = CommitmentLedger({sid: pub for sid, (priv, pub) in keys.items()})
    entry = ledger.append(
        data={"paper_id": paper_name, "paper_hash": hash_paper(paper_bytes),
              "answer_key_hash": hashlib.sha256(key_bytes).hexdigest(), "n_questions": len(qs)},
        cosigners={sid: priv for sid, (priv, pub) in keys.items()},
    )
    return ledger, entry, paper_bytes


def trained_detector(exclude_paper: str, backend: str, verbose=True) -> LeakDetector:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"demo_detector_{backend}_{exclude_paper.replace(' ', '_')}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    if verbose:
        print(f"Training the detector on sessions about the OTHER exams (never on {exclude_paper}) -- one-off, cached ...")
    ds = generate_hard_dataset(n_sessions=800, seed=11)
    pool = {r["image_id"]: r for r in ds["pool"]}
    sessions = [s for s in ds["sessions"] if s["exam_paper"] != exclude_paper]
    annotate_events(sessions, pool)
    rng = random.Random(0)
    rng.shuffle(sessions)
    cut = int(0.8 * len(sessions))
    det = LeakDetector(gnn_backend=backend, seed=0).fit(sessions[:cut], sessions[cut:])
    pickle.dump(det, open(f, "wb"))
    return det


def _fmt_event(e, ws):
    h = (ws - e["timestamp"]) / 3600
    who = f"{e['group_id'].split('_')[0]:10s} {e['sender_id'][:10]}"
    body = f"[photo {e['image_id']}] {e.get('text') or ''}" if e["event_type"] == "image" else (e.get("text") or "")
    body = body.replace("\n", " ")
    flag = ""
    ev = e.get("_ev", {})
    if ev.get("kind") == "text":
        best = max(("lexical", ev["lexical"]), ("semantic", ev["semantic"]), ("answer-key", ev["answer_key"]), key=lambda kv: kv[1])
        if best[1] > 0.3:
            flag = f"  <== matches {ev['matched_qid'].split('#')[1] if ev['matched_qid'] else '?'} ({best[0]} {best[1]:.2f})"
    elif ev.get("kind") == "image" and ev["orb"] > 0.3:
        flag = f"  <== photo of committed page {ev['page']} ({ev['inliers']} inliers)"
    return f"T-{h:6.1f}h  {who}  {body[:70]:70s}{flag}"


def replay(det, session, title, pace, ledger, entry, paper_name):
    print("\n" + "=" * 110)
    print(f" {title}")
    print("=" * 110)
    ws = session["window_start"]
    evs = session["events"]
    prefixes = [dict(session, events=evs[: i + 1]) for i in range(len(evs))]
    probs = det.predict_proba(prefixes)
    alerted_at = None
    for i, (e, p) in enumerate(zip(evs, probs)):
        line = _fmt_event(e, ws)
        if "<==" in line or i % max(1, len(evs) // 25) == 0 or (alerted_at is None and p >= det.threshold):
            print(f"{line}   P(leak)={p:.2f}")
            if pace:
                time.sleep(pace)
        if alerted_at is None and p >= det.threshold:
            alerted_at = i
            print("\n" + "!" * 110)
            print(f"  ALERT at T-{(ws - e['timestamp']) / 3600:.1f} h before the exam, after {i + 1} of {len(evs)} messages "
                  f"-- routed to a HUMAN reviewer, not an automatic takedown")
            print("!" * 110)
            ex = det.explain(prefixes[i])
            print(format_explanation(ex))
            matched = sorted({m["matched_qid"] for m in ex["top_messages"] if m["matched_qid"]} |
                             {q for im in ex["top_images"] for q in im["qids"]})
            committed_ids = {q["qid"] for q in load_papers()[paper_name]}
            print(f"\n  Forensic check against the ledger: entry #{entry['index']} committed {paper_name} "
                  f"(paper hash {entry['data']['paper_hash'][:16]}..., co-signed by {len(entry['signatures'])} stakeholders).")
            print(f"  Matched questions {', '.join(q.split('#')[1] for q in matched) or '-'} are "
                  f"{'ALL' if matched and set(matched) <= committed_ids else 'NOT'} part of that committed paper. "
                  f"Ledger chain valid: {ledger.verify_chain()[0]}")
            print("!" * 110 + "\n")
            if pace:
                time.sleep(pace * 4)
    if alerted_at is None:
        print(f"\n  No alert: final P(leak) = {probs[-1]:.2f} < threshold {det.threshold:.2f}.")
    return probs, alerted_at


def plot_timeline(results, det, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#eb6834"]                       # categorical slots 1-2 (validated palette)
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for (title, session, probs, alerted_at), c in zip(results, colors):
        hrs = [-(session["window_start"] - e["timestamp"]) / 3600 for e in session["events"]]
        ax.plot(hrs, probs, color=c, lw=2, label=title)
        if alerted_at is not None:
            ax.scatter([hrs[alerted_at]], [probs[alerted_at]], s=70, color=c, edgecolor="#fcfcfb", linewidth=2, zorder=3)
            ax.annotate("alert", (hrs[alerted_at], probs[alerted_at]), textcoords="offset points", xytext=(6, 8),
                        fontsize=9, color="#0b0b0b")
    ax.axhline(det.threshold, color="#898781", lw=1, ls="--")
    ax.text(ax.get_xlim()[0], det.threshold + 0.02, f"alert threshold {det.threshold:.2f}", fontsize=9, color="#52514e")
    ax.set_xlabel("hours relative to exam start", color="#52514e")
    ax.set_ylabel("P(leak) so far", color="#52514e")
    ax.set_ylim(-0.02, 1.02)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Live detection: probability of a leak as each channel unfolds", loc="left", color="#0b0b0b")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def load_user_chat(path, paper_name):
    """Parse the --chat file into a session; photos are matched live against the committed pages."""
    from PIL import Image
    from src.detection.image_match import CommittedPaperImages
    from src.detection.photo_sim import paper_pages
    ws = 1_800_000_000.0
    events, pages_index = [], None
    for ln in open(path, encoding="utf-8"):
        if not ln.strip() or ln.startswith("#"):
            continue
        hhmm, group, sender, msg = [x.strip() for x in ln.split("|", 3)]
        h, m = map(int, hhmm.split(":"))
        t = ws - 86400 + h * 3600 + m * 60
        e = {"timestamp": t, "sender_id": sender, "group_id": f"{group}_group", "region": group, "role": "unknown",
             "is_leak_event": False, "event_type": "text", "text": msg, "image_id": None}
        if msg.upper().startswith("IMG:"):
            if pages_index is None:
                pages_index = CommittedPaperImages(paper_name, paper_pages(paper_name, load_papers()[paper_name]))
            img_path = msg[4:].strip()
            r = pages_index.match(Image.open(img_path))
            e.update(event_type="image", text="", image_id=Path(img_path).name,
                     _ev={"kind": "image", "phash": r["phash_score"], "orb": r["orb_score"], "page": r["page"],
                          "inliers": r["inliers"], "qids": r["qids"]})
        events.append(e)
    events.sort(key=lambda e: e["timestamp"])
    return {"session_id": Path(path).stem, "label": "unknown", "scenario": "user_chat", "exam_paper": paper_name,
            "window_start": ws, "events": events}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", default="JEE Adv 2023 Paper 1", choices=TARGET_PAPERS)
    ap.add_argument("--scenario", default="full_leak", choices=list(LEAK_MIX))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--fast", action="store_true", help="no pauses between messages")
    ap.add_argument("--torch", action="store_true",
                     help="force the PyTorch backend (default: auto -- used automatically when torch/"
                          "torch_geometric are installed; --torch also adds GNNExplainer account attribution)")
    ap.add_argument("--numpy", action="store_true", help="force the dependency-free NumPy backend even if PyTorch is installed")
    ap.add_argument("--chat", help="score your own chat export instead of a simulated channel")
    args = ap.parse_args()
    pace = 0 if args.fast else 0.15
    backend = "torch" if args.torch else ("numpy" if args.numpy else "auto")

    print(f"Step 1 -- committing the REAL paper '{args.paper}' to the co-signed ledger before the exam ...")
    ledger, entry, _ = commit_paper(args.paper)
    print(f"  ledger entry #{entry['index']}: paper hash {entry['data']['paper_hash'][:24]}..., "
          f"answer-key hash {entry['data']['answer_key_hash'][:24]}..., signatures: {', '.join(entry['signatures'])}")

    print("Step 2 -- loading the detector (trained without this paper) ...")
    det = trained_detector(args.paper, backend)
    print(f"  learned fusion weights: {det.weights()}   alert threshold: {det.threshold:.3f}")

    pool = build_image_pool(verbose=False)
    pool_by_id = {r["image_id"]: r for r in pool}
    results = []
    if args.chat:
        s = load_user_chat(args.chat, args.paper)
        annotate_events([s], pool_by_id)
        probs, at = replay(det, s, f"YOUR CHAT: {args.chat}", pace, ledger, entry, args.paper)
        results.append(("your chat", s, probs, at))
    else:
        rng = random.Random(args.seed)
        leak = generate_session("demo_leak", args.scenario, rng, pool, exam_paper=args.paper)
        scam = generate_session("demo_scam", "fake_leak_scam", random.Random(args.seed + 1), pool, exam_paper=args.paper)
        annotate_events([leak, scam], pool_by_id)
        for title, s in ((f"Channel A -- a real leak ({args.scenario.replace('_', ' ')})", leak),
                         ("Channel B -- a 'genuine paper, pay Rs 5000' scam (NOT a leak)", scam)):
            probs, at = replay(det, s, title, pace, ledger, entry, args.paper)
            results.append((title.split(" -- ")[0] + (" (leak)" if s is leak else " (scam)"), s, probs, at))
    out = FIG / "live_demo_timeline.png"
    plot_timeline(results, det, out)
    print(f"\nTimeline chart saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
