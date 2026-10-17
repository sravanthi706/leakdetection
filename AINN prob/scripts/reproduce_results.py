"""
Re-create every number and figure in docs/ML_REPORT.md from scratch.

    python scripts/reproduce_results.py            # full run (~1.5-2 h on a laptop CPU)
    python scripts/reproduce_results.py --quick    # smaller settings, ~15 min, same pipeline

Steps: download + verify datasets -> image pool -> content benchmark -> image
benchmark -> session benchmark (leave-one-paper-out) -> Twitter15/16 benchmark
(needs PyTorch Geometric; skipped otherwise) -> figures. Outputs go to
reports/results/ and reports/figures/.
"""

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    print(f"\n>>> python {' '.join(args)}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable] + args, cwd=ROOT, check=True)
    print(f"    done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    quick = "--quick" in sys.argv
    has_pyg = importlib.util.find_spec("torch_geometric") is not None
    run(["scripts/download_data.py"])
    run(["-m", "src.detection.image_pool"])
    run(["-m", "src.detection.image_pool", "--stress"])
    run(["-m", "src.evaluation.content_benchmark"])
    run(["-m", "src.evaluation.image_benchmark"])
    run(["-m", "src.evaluation.experiment"] + (["--quick"] if quick else []) + ([] if has_pyg else ["--no-torch"]))
    if has_pyg:
        run(["-m", "src.evaluation.twitter_benchmark"] + (["--quick"] if quick else []))
    else:
        print("\nPyTorch Geometric not installed -- skipping the Twitter15/16 benchmark (pip install -r requirements-ml.txt).")
    run(["-m", "src.evaluation.plots"])
    print("\nAll results are in reports/results/ and reports/figures/.")


if __name__ == "__main__":
    main()
