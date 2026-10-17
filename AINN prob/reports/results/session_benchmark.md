| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **AINN (learned fusion)** | 0.920 ± 0.016 | 0.966 ± 0.028 | 0.879 ± 0.022 | 0.956 ± 0.012 | 0.944 ± 0.011 |
| AINN (learned fusion, PyTorch GNN) | 0.920 ± 0.015 | 0.964 ± 0.025 | 0.881 ± 0.029 | 0.975 ± 0.010 | 0.960 ± 0.009 |
| equal-weight fusion (original rule) | 0.904 ± 0.029 | 0.975 ± 0.029 | 0.846 ± 0.056 | 0.943 ± 0.012 | 0.932 ± 0.012 |
| baseline: text_tfidf_lr | 0.563 ± 0.046 | 0.833 ± 0.077 | 0.431 ± 0.058 | 0.935 ± 0.024 | 0.806 ± 0.056 |
| baseline: features_rf | 0.735 ± 0.053 | 0.753 ± 0.102 | 0.739 ± 0.100 | 0.928 ± 0.016 | 0.841 ± 0.044 |
| baseline: features_lr | 0.576 ± 0.054 | 0.585 ± 0.079 | 0.588 ± 0.102 | 0.829 ± 0.020 | 0.656 ± 0.040 |
| baseline: keyword_rule | 0.395 ± 0.039 | 0.250 ± 0.024 | 0.949 ± 0.134 | 0.450 ± 0.033 | 0.225 ± 0.035 |
| single: content | 0.817 ± 0.028 | 0.978 ± 0.025 | 0.703 ± 0.041 | 0.847 ± 0.019 | 0.814 ± 0.025 |
| single: image | 0.569 ± 0.053 | 0.951 ± 0.089 | 0.409 ± 0.049 | 0.659 ± 0.018 | 0.570 ± 0.026 |
| single: gnn | 0.463 ± 0.035 | 0.401 ± 0.106 | 0.657 ± 0.185 | 0.719 ± 0.029 | 0.488 ± 0.062 |
| single: gnn (PyTorch) | 0.550 ± 0.059 | 0.434 ± 0.054 | 0.762 ± 0.099 | 0.797 ± 0.053 | 0.604 ± 0.052 |
| single: autoencoder | 0.438 ± 0.044 | 0.345 ± 0.057 | 0.655 ± 0.148 | 0.671 ± 0.053 | 0.449 ± 0.067 |
| without content | 0.647 ± 0.042 | 0.762 ± 0.188 | 0.597 ± 0.083 | 0.830 ± 0.033 | 0.753 ± 0.031 |
| without image | 0.843 ± 0.022 | 0.972 ± 0.024 | 0.744 ± 0.032 | 0.917 ± 0.016 | 0.887 ± 0.023 |
| without gnn | 0.917 ± 0.019 | 0.962 ± 0.031 | 0.877 ± 0.021 | 0.951 ± 0.012 | 0.938 ± 0.013 |
| without autoencoder | 0.911 ± 0.014 | 0.973 ± 0.029 | 0.858 ± 0.030 | 0.946 ± 0.013 | 0.938 ± 0.009 |
| AINN with 10% noisy training labels | 0.909 ± 0.020 | 0.939 ± 0.039 | 0.883 ± 0.027 | 0.948 ± 0.022 | 0.938 ± 0.016 |
| AINN with 20% noisy training labels | 0.889 ± 0.040 | 0.890 ± 0.081 | 0.893 ± 0.015 | 0.948 ± 0.018 | 0.935 ± 0.011 |

Pooled out-of-fold (n=2400, 8 leave-one-paper-out folds): AINN F1 = 0.919 (95% CI 0.902-0.934), PR-AUC = 0.944 (95% CI 0.931-0.958).
Best baseline (baseline: features_rf): F1 = 0.735 (95% CI 0.707-0.762).
Paired bootstrap F1 difference AINN - best baseline: +0.184 (95% CI +0.156 to +0.214, one-sided p = 0.0000).
Calibration: expected calibration error = 0.078.

Learned fusion weights (standardised log-odds per signal, mean ± std over folds):

- gnn: +0.48 ± 0.40
- autoencoder: +0.44 ± 0.26
- content_lexical: +2.74 ± 0.23
- content_semantic: -0.39 ± 0.19
- content_answer_key: +1.08 ± 0.33
- image_orb: +1.66 ± 0.11

Alert rate per scenario (leak rows = recall, normal rows = false-alarm rate):

| Scenario | AINN (learned fusion) | AINN (learned fusion, PyTorch GNN) | baseline: keyword_rule | baseline: features_lr | baseline: features_rf | baseline: text_tfidf_lr |
|---|---|---|---|---|---|---|
| full_leak (leak) | 1.00 | 1.00 | 1.00 | 0.94 | 0.93 | 0.11 |
| partial_leak (leak) | 0.97 | 0.98 | 0.98 | 0.48 | 0.53 | 0.32 |
| paraphrased_leak (leak) | 0.99 | 0.99 | 1.00 | 0.51 | 0.71 | 0.17 |
| answer_key_leak (leak) | 1.00 | 1.00 | 1.00 | 0.33 | 0.73 | 0.81 |
| photo_leak (leak) | 1.00 | 1.00 | 0.89 | 0.66 | 0.77 | 0.96 |
| quiet_leak (leak) | 0.99 | 0.99 | 0.88 | 0.78 | 1.00 | 0.07 |
| buried_leak (leak) | 0.93 | 0.94 | 0.88 | 0.43 | 0.77 | 0.44 |
| coordination_only_leak (leak) | 0.00 | 0.00 | 1.00 | 0.30 | 0.34 | 0.84 |
| study_chatter (normal) | 0.00 | 0.00 | 0.95 | 0.09 | 0.09 | 0.02 |
| past_paper_discussion (normal) | 0.03 | 0.03 | 0.88 | 0.17 | 0.07 | 0.00 |
| coaching_broadcast (normal) | 0.01 | 0.01 | 1.00 | 0.00 | 0.01 | 0.00 |
| fake_leak_scam (normal) | 0.01 | 0.01 | 1.00 | 0.35 | 0.21 | 0.12 |

Streaming time-to-detection (each session re-scored at 17 checkpoints, 240 h -> 0 h before the exam, using only messages seen so far):

- leaks flagged before the exam starts: 88.0%
- flagged at least 24 h before the exam: 81.5%
- median lead time of flagged leaks: 168.0 h before the exam
- median delay between the first leaked item appearing and the alert: 11.842682599027953 h
- normal sessions raising an alert at ANY checkpoint (streaming false-alarm rate): 1.5%
- leak sessions alerting before any leak activity existed (premature alarm): 0.7%
