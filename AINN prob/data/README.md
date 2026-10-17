# Data used by AINN

Nothing here was scraped from a live platform, and no real leak-channel data is used. Every
external dataset is a published research dataset, pinned to an exact git commit and verified by
SHA-256 when downloaded:

```
python scripts/download_data.py
```

| Folder | Dataset | What it is | Used for | Licence / terms |
|---|---|---|---|---|
| `external/jeebench/` | **JEEBench** (Arora, Singh & Mausam, EMNLP 2023) | 515 real JEE-Advanced questions, 2016-2023, tagged by paper, with official answers | The committed question papers (4 "protected" papers) and 12 past papers used as legitimate look-alike content | MIT |
| `external/twitter/` | **Twitter15 / Twitter16** (Ma, Gao & Wong, ACL 2017), pre-processed by **BiGCN** (Bian et al., AAAI 2020) | 1,490 + 818 real rumour cascades: source tweet + full retweet/reply tree, 5000-word bag-of-words per tweet, 4 veracity classes | Real-data benchmark for the propagation-graph GNN | Research dataset, distributed with the BiGCN code; cite both papers, academic use |
| `external/msrp/` | **Microsoft Research Paraphrase Corpus** (Dolan & Brockett, 2005) | 5,801 human-labelled sentence pairs | Standard paraphrase benchmark for the content matcher | Microsoft Research Shared Source licence (non-commercial research) |

## Files created by this project

| Path | What it is |
|---|---|
| `annotations/leak_paraphrases_jee2023.json` | 80 hand-written rewordings of 40 real JEE 2023 questions: one English paraphrase and one Hinglish student-style retype each. Written by an AI assistant acting as annotator, used **only for evaluation** and to build the hardest "paraphrased leak" sessions. No model is trained on it. |
| `cache/image_pool_v2_*.json` | Scores of 210 simulated phone photos / digital copies against every committed page (built once, ~10 min; delete to rebuild). The held-out `_stress_` pool uses a different seed and harsher distortions. |
| `cache/demo_detector_*.pkl` | The trained detector used by the live demo (rebuilt automatically if deleted). |
| `photos/` | Put **your own** phone photos of printed pages here and run `python scripts/eval_user_photos.py` (see `photos/README.md`). |

## What is synthetic, and what is not

| Real | Simulated (and stated as such) |
|---|---|
| Question text and answer keys (JEEBench) | Chat sessions: who posts what, when, in which group |
| Twitter propagation cascades (Twitter15/16) | Phone-photo distortions (perspective, blur, light, compression) |
| Paraphrase pairs (MSRP) | Scenario mix, volumes, timing (modelling assumptions) |
| Hand-written paraphrases of real questions | |

The session generator's *shape* is calibrated to publicly reported facts about the NEET-UG 2026
leak (multi-hop broker chain, multi-state spread, text + photos, ~10-day window; sources cited in
`src/detection/synthetic_data.py`). Everything else about sessions is a stated modelling choice.

## Citations

- D. Arora, H. G. Singh, Mausam. *Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark for Large Language Models.* EMNLP 2023.
- J. Ma, W. Gao, K.-F. Wong. *Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning.* ACL 2017.
- T. Bian, X. Xiao, T. Xu, P. Zhao, W. Huang, Y. Rong, J. Huang. *Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks.* AAAI 2020.
- W. B. Dolan, C. Brockett. *Automatically Constructing a Corpus of Sentential Paraphrases.* IWP 2005.
