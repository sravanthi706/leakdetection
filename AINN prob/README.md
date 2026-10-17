# AINN — Tamper-Evident, Time-Locked Exam Paper Distribution + Explainable Leak Detection

AINN has two layers.

1. **Custody layer.** No single person, device or server can read the question paper before the exam
   window opens. It uses AES-256-GCM, ML-KEM, 3-of-4 Shamir custody, a simulated TPM/TEE, a
   Verifiable Delay Function time-lock, an AND-gated release, and an active tamper response.
   `hash(paper)` and `hash(answer key)` are committed to a co-signed ledger before distribution.
2. **Detection layer (ML).** It watches channels for leaks and explains every alert. Messages and
   photos are matched against the **committed** paper. The learned fusion also uses propagation
   graph and behaviour signals.

| Document | Contents |
|---|---|
| `docs/ML_REPORT.md` | The ML evaluation: data, protocol, results, limitations. **Start here for the ML part.** |
| `docs/AINN_MASTER.md` | Full specification and patent strategy. |
| `docs/diagrams/` | Architecture diagrams. |

## Setup

```
pip install -r requirements.txt            # core: everything runs with this
python scripts/download_data.py            # real datasets, pinned + checksum-verified (~25 MB)
pip install -r requirements-ml.txt         # optional: PyTorch GNN + real Twitter benchmark
pip install -r requirements-dev.txt        # optional: tests
```

Run everything from the project root, as modules (`python -m src....`).

## The live demo (real paper, a leak unfolding, an explainable alert)

```
python -m src.demo.live_demo                          # a real leak channel vs a scam channel -- uses PyTorch automatically when installed
python -m src.demo.live_demo --scenario photo_leak    # other leak types: partial_leak, paraphrased_leak, answer_key_leak, quiet_leak, buried_leak ...
python -m src.demo.live_demo --torch                  # force PyTorch (also adds GNNExplainer account attribution) / --numpy forces the dependency-free backend
python -m src.demo.live_demo --chat my_chat.txt       # score your own messages / photos
```

What the demo does:

1. It commits JEE Advanced 2023 Paper 1 (real questions) to the co-signed ledger.
2. It loads a detector that was trained **without** that paper.
3. It replays a channel message by message, flagging matches as they arrive and showing P(leak).
4. When P(leak) crosses the threshold, it prints the evidence: which message matched which
   question, which photo matched which page, each signal's contribution, and a forensic check
   against the ledger.
5. It then replays a "genuine paper, pay Rs 5000" scam channel, which does **not** alert.

## Headline results (details in `docs/ML_REPORT.md`)

These are leave-one-paper-out results: the test exam paper is never seen in training. There are 8
folds and 2,400 test sessions, 25% of them leaks.

| Model | F1 | PR-AUC |
|---|---|---|
| **AINN (learned fusion)** | **0.920 ± 0.016** | **0.944 ± 0.011** |
| Best baseline (random forest on session statistics) | 0.735 ± 0.053 | 0.841 ± 0.044 |
| Supervised TF-IDF text classifier | 0.563 ± 0.046 | 0.806 ± 0.056 |
| Keyword rule | 0.395 ± 0.039 | 0.225 ± 0.035 |

Other results:

- **Streaming:** 97.8% of leaks that post any leaked content are flagged before the exam. The
  median delay is ≤ 11.8 h after the first leaked item appears. 1.5% of normal channels ever alert.
- **Photos:** verified ORB finds photographed committed pages with recall 1.00 on standard photos
  and 0.93 on held-out harsh photos, at 1% FPR. The original pHash method is at chance on phone
  photos.
- **Real cascades (Twitter15/16):** our BiGCN re-implementation reaches 0.820 / 0.844 accuracy
  (published 0.844 / 0.880, different protocol). A text-blind structure GNN is near chance (0.32 / 0.33).
- **Known blind spot:** coordination-only leaks, where the paper never appears in a monitored
  channel, are not detected (0%). The report explains why.

Reproduce all numbers and figures with `python scripts/reproduce_results.py` (add `--quick` for a
~15 min run).

## Folder layout

```
src/
  crypto/         AES-256-GCM, ML-KEM key wrap, Shamir secret sharing
  custody/        commitment ledger, simulated TPM + TEE, VDF, AND-gate release, tamper response
  detection/
    exam_corpus.py      real JEE papers (JEEBench) + LaTeX -> plain text
    leak_text.py        how questions travel in chats: verbatim, OCR, typed, paraphrase, Hinglish, answer keys
    paper_matcher.py    content branch: per-question lexical + semantic + answer-key matching
    photo_sim.py        exam-page renderer + realistic phone-camera simulation
    image_match.py      image branch: pHash + verified ORB/RANSAC page matching
    image_pool.py       pre-scored image pool (cached in data/cache/)
    realistic_data.py   HARD session generator (11 scenarios incl. scams and hidden leaks)
    engine.py           LeakDetector: learned fusion, streaming-aware training, explanations
    gnn.py / gnn_torch.py   propagation GNN (NumPy / PyTorch Geometric + GNNExplainer -- torch used automatically when installed)
    autoencoder.py / autoencoder_torch.py   behavioural anomaly branch (NumPy / PyTorch -- torch used automatically when installed)
    (synthetic_data.py, content_match.py, phash_match.py, fusion.py: original v1 modules, still used by the orchestrator)
  evaluation/
    experiment.py         main benchmark: leave-one-paper-out CV, baselines, ablations, noise, streaming
    baselines.py          keyword rule, LR, random forest, TF-IDF text classifier
    content_benchmark.py  MSRP + leak-message detection on real JEE text
    image_benchmark.py    pHash vs ORB on phone photos (standard + held-out harsh pool)
    twitter_benchmark.py  BiGCN / GCN / structure-only GNN on real Twitter15/16 cascades
    plots.py              figures for the report
  demo/live_demo.py       the live, explainable demo
  attacks/                attack-resistance harness (custody layer)
  report/                 dashboard for the custody + v1 detection run
  orchestrator.py         the 9-step custody chain end to end
scripts/
  download_data.py        fetch + verify datasets
  reproduce_results.py    regenerate every number and figure
  eval_user_photos.py     test the image branch on YOUR phone photos
data/                     datasets (downloaded), annotations, caches, your photos -- see data/README.md
reports/results/          benchmark outputs (JSON + Markdown tables)
reports/figures/          charts for slides / viva
tests/test_ainn.py        automated checks: python -m pytest -q
```

## Custody layer self-tests

```
python -m src.orchestrator            # full 9-step custody chain + v1 detection
python -m src.attacks.attack_harness  # 8 attack-resistance scenarios
python -m src.custody.vdf             # any module self-tests when run directly
```

## What is real and what is simulated

| Real | Simulated |
|---|---|
| Question text and answer keys (JEEBench, 515 questions) | Chat sessions (who/what/when) |
| Twitter15/16 propagation cascades | Phone-camera distortions |
| MSRP paraphrase pairs | TPM / TEE (software models of the real interfaces) |
| Every cryptographic primitive | VDF RSA modulus without a multi-party setup ceremony |

Each simulation is stated in its module's docstring and in the reports.
