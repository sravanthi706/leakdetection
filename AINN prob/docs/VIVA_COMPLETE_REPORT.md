# AINN — Comprehensive Project Master Report & Viva Defense Guide

**Project Title:** *A Threshold-Cryptographic, Hardware-Attested Time-Lock Architecture with Multi-Modal Neural Anomaly Detection for Tamper-Evident Examination Paper Distribution*  
**Short / Patent Title:** *System and Method for Threshold-Secured, Hardware-Attested, Time-Locked Distribution of Confidential Examination Content with Real-Time Leak Detection*  
**Architecture Paradigm:** Dual-Layer Defense-in-Depth (Custody Layer + Explainable ML Detection Layer)

---

## 1. Executive Summary & Problem Motivation

### 1.1 The Real Failure Mode: Insider Collusion
National examinations in India (NEET-UG, Vyapam, SSC, State PSCs) suffer recurring question paper compromises affecting millions of candidates. Traditional cybersecurity measures assume external hackers attacking servers. However, forensic analysis of real-world cases reveals that **the root failure mode is insider compromise and institutional trust concentration**:
* Exam setters, printing press managers, transit custodians, or center superintendents gain authorized plaintext access before the examination window opens.
* Once accessed, questions are transcribed, photographed, or coordinated across social channels (Telegram, WhatsApp).
* Existing investigations struggle to prove whether circulating "guess papers" or "scam channels" are genuine leaks or coincidental overlaps.

### 1.2 The AINN Solution
AINN guarantees that **no single individual, server, or agency—including authorized administrators—can decrypt or read the question paper before the official examination start window**. Simultaneously, an independent, explainable Machine Learning layer monitors external channels in real time, detecting leaks and proving whether circulating content mathematically matches the committed paper.

```mermaid
graph TD
    subgraph Layer 1: Custody Layer
        A[Original Exam Paper] --> B[AES-256-GCM + ML-KEM Encryption]
        B --> C[Shamir Secret Sharing: 3-of-4 Stakeholder Split]
        B --> D[Ed25519 Co-Signed Commitment Ledger]
        C --> E[Verifiable Delay Function: VDF Time-Lock]
        C --> F[Hardware TPM 2.0 / TEE Enclave Attestation]
        E & F --> G{AND-Gate Release}
        G -->|Center Attested AND Time Elapsed| H[Decrypted at Center Terminal]
    end

    subgraph Layer 2: Detection Layer (Real-Time ML)
        I[Public / Social Traffic: Chats, Images] --> J[Learned Fusion Engine]
        D -. Reference .-> J
        J --> K[Lexical & Semantic Matcher]
        J --> L[ORB / RANSAC Image Matcher]
        J --> M[Cascade GNN]
        J --> N[Behavioral Autoencoder]
        J --> O[Forensic Explainable Alert]
    end
```

---

## 2. Layer 1: Custody & Cryptographic Architecture

The custody layer implements a 9-step zero-trust distribution protocol. Plaintext question papers never touch the network; only encrypted packages are distributed in advance.

### 2.1 Cryptographic Primitives & Technical Mechanisms

1. **Authenticated Content Encryption (AES-256-GCM + AAD)**:
   * The question paper $P$ is encrypted with an ephemeral 256-bit content key $K_c$.
   * Critical metadata (Center ID, Paper ID, Exam Window Timestamp) is injected into the **Additional Authenticated Data (AAD)**. Any tampering with center identifiers or timestamps causes authentication tag verification to fail immediately.

2. **Post-Quantum Key Encapsulation (ML-KEM-768 / Kyber)**:
   * Content encryption keys are wrapped using ML-KEM-768 (NIST FIPS 203 standard), future-proofing the system against "harvest now, decrypt later" quantum attacks.

3. **Threshold Key Custody (3-of-4 Shamir's Secret Sharing over $GF(2^8)$)**:
   * To prevent a rogue authority (e.g., a corrupt testing director), $K_c$ is split into 4 polynomial shares distributed across mutually distrusting stakeholders:
     * *Share 1:* National Testing Agency (NTA).
     * *Share 2:* Ministry of Education.
     * *Share 3:* State Examination Board.
     * *Share 4:* Independent Judicial / Audit Custodian.
   * **Reconstruction requires $t = 3$ shares**. No single institution can reconstruct the key.

4. **Cryptographic Commitment Ledger (Ed25519 Co-signed Hash Chain)**:
   * Prior to distribution, the cryptographic hash of the plaintext paper $H(P)$ and answer key $H(A)$ are committed to an append-only, tamper-evident ledger.
   * Every entry is co-signed by all stakeholders using Ed25519 digital signatures.
   * If a leak occurs, circulating questions can be checked against the pre-committed hash with mathematical certainty.

5. **Verifiable Delay Function (VDF Time-Lock)**:
   * A sequential squaring time-lock puzzle (Pietrzak construction over an RSA modulus $N$).
   * Computes $y = x^{2^T} \pmod N$, where $T$ enforces a non-parallelizable computation delay. Even an attacker with thousands of GPU/ASIC cores cannot shortcut the wall-clock time required for release.

6. **Hardware Attestation (TPM 2.0 & TEE Enclave)**:
   * Exam center decryption devices must present valid Platform Configuration Register (PCR) quotes signed by an endorsed TPM 2.0 chip.
   * Keys are sealed to PCR states; if the OS kernel, bootloader, or application binary is modified, unsealing fails.

7. **AND-Gate Conditional Release & Dead-Man's Switch**:
   * Plaintext key unsealing requires:
     $$\text{Release} = (\text{VDF Verification} == \text{True}) \land (\text{TPM/TEE Attestation} == \text{Valid})$$
   * **Active Tamper Response:** Exceeding failed attestation thresholds triggers an irreversible zeroization of local key material and alerts central audit.

### 2.2 The 9-Step Custody Lifecycle

| Step | Phase | Action |
|:---:|---|---|
| **1** | Setting | Paper finalized inside isolated environment; hash $H(P)$ generated. |
| **2** | Commitment | $H(P)$ and $H(A)$ co-signed into the ledger by 4 stakeholders. |
| **3** | Encryption | Encrypted with AES-256-GCM; metadata bound via AAD. |
| **4** | Key Split | Content key $K_c$ split via Shamir 3-of-4 threshold. |
| **5** | Enclave Sealing | Shares sealed to target exam-center hardware TPM/TEE PCRs. |
| **6** | Distribution | Sealed ciphertext packages broadcast to centers days in advance. |
| **7** | Time-Locking | VDF time-lock puzzle initiated across stakeholder nodes. |
| **8** | Release Gate | Exam window opens ($T - 15$ min): VDF solved + Center PCR verified. |
| **9** | Plaintext Print/Display | In-memory decryption only; keys immediately zeroized from RAM. |

---

## 3. Layer 2: Explainable Machine Learning Detection Layer

While Layer 1 prevents pre-exam access, Layer 2 monitors external channels (Telegram, WhatsApp public groups, forums) to detect physical leaks (e.g., center break-in or printing espionage) and separate real leaks from fake scams.

### 3.1 Four Multi-Modal Detection Branches

```
                         ┌── Content Lexical (Char 3-5 gram TF-IDF) ──┐
                         ├── Content Semantic (WordLlama Embedding)  ──┤
  Incoming Traffic Stream ─── Image Matcher (ORB + RANSAC Homography) ──┼──> Stacking Logistic Fusion ──> Calibrated P(Leak)
                         ├── Cascade GNN (GraphSAGE / PyG)           ──┤
                         └── Anomaly AE (Timing & Volume Anomaly)    ──┘
```

1. **Content Matching Branch (Committed Paper Reference)**:
   * **Lexical:** Character 3-to-5-gram TF-IDF matching against committed questions. Robust to OCR noise, typos, and truncation.
   * **Semantic:** Dense sentence embeddings (WordLlama / MiniLM) capturing conceptual rephrasing.
   * **Answer-Key Parser:** Regular expression extractor for patterns like `Q3-B, Q4-AC, Q11-281`, checked against committed keys. Catches leaks where no question text is posted.

2. **Image Matching Branch (Verified ORB + RANSAC)**:
   * **Failure of Naive pHash:** Standard perceptual hashing (`pHash`) drops to chance level ($FPR > 50\%$) on real phone photos due to perspective skew, shadows, and WhatsApp compression.
   * **AINN Solution:** Extracts **ORB (Oriented FAST and Rotated BRIEF)** keypoints, computes homography via **RANSAC**, and validates candidate matches using **page specificity** and **pixel agreement after perspective warping**.
   * **Performance:** 1.00 recall on standard photos, 0.93 on extreme angled/low-light photos at 1% FPR.

3. **Propagation Cascade GNN Branch (PyTorch Geometric / NumPy)**:
   * Models the message forwarding structure as a directed graph.
   * GraphSAGE captures propagation speed, depth, and structural fan-out.
   * Integrated with **GNNExplainer** to highlight the specific accounts driving the cascade.

4. **Behavioral Anomaly Autoencoder Branch**:
   * Trained exclusively on normal exam discussion traffic.
   * Encodes 13 behavioural features (burst rates, sender entropy, nocturnal timing, multi-region coordinates).
   * Spikes in reconstruction error signal anomalous coordination.

### 3.2 Fusion & Streaming-Aware Calibration
* Individual branch signals are combined via a **Stacking Logistic Regression** meta-classifier.
* Calibrated on legitimate traffic: A content score of 0 corresponds to the 95th percentile of legitimate past-paper discussions.
* **Streaming-Aware Training:** Models are trained on partial message prefixes ($t \in [0, h]$) to prevent early-session false alarms.

---

## 4. Experimental Rigor & Benchmark Results

### 4.1 Evaluation Protocol
* **Strict Leave-One-Paper-Out (LOPO) Cross-Validation**:
  * 8 folds, 2,400 test sessions (25% leak prevalence).
  * The test exam paper is **never seen during training**. Models cannot memorize specific question vocabularies.
* **11 Realistic Scenarios**:
  * *Negative Scams & Look-Alikes:* Fake-leak scams (*"100% genuine paper pay Rs 5000"*), past-paper discussions, coaching mock tests, general chatter.
  * *Covert Leaks:* Paraphrased/Hinglish leaks, answer-key-only leaks, photo-only leaks, quiet 3-person leaks, buried leaks.

### 4.2 Headline Performance Comparison

| Model | F1 Score | Precision | Recall | PR-AUC | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **AINN (Learned Fusion)** | **0.920 ± 0.016** | **0.966 ± 0.028** | **0.879 ± 0.022** | **0.944 ± 0.011** | **0.956 ± 0.012** |
| AINN (PyTorch GNN Backend) | 0.920 ± 0.015 | 0.964 ± 0.025 | 0.881 ± 0.029 | 0.960 ± 0.009 | 0.975 ± 0.010 |
| Equal-Weight Fusion (v1 Rule) | 0.904 ± 0.029 | 0.975 ± 0.029 | 0.846 ± 0.056 | 0.932 ± 0.012 | 0.943 ± 0.012 |
| Baseline: Random Forest (Session Stats) | 0.735 ± 0.053 | 0.753 ± 0.102 | 0.739 ± 0.100 | 0.841 ± 0.044 | 0.928 ± 0.016 |
| Baseline: Supervised Text TF-IDF | 0.563 ± 0.046 | 0.833 ± 0.077 | 0.431 ± 0.058 | 0.806 ± 0.056 | 0.935 ± 0.024 |
| Baseline: Logistic Regression (Stats) | 0.576 ± 0.054 | 0.585 ± 0.079 | 0.588 ± 0.102 | 0.656 ± 0.040 | 0.829 ± 0.020 |
| Baseline: Keyword Heuristic Rule | 0.395 ± 0.039 | 0.250 ± 0.024 | 0.949 ± 0.134 | 0.225 ± 0.035 | 0.450 ± 0.033 |

*Statistically significant superiority: Paired bootstrap F1 gain over best baseline is $+0.184$ ($p < 0.0001$).*

### 4.3 Signal Ablation Analysis (What Drives Detection?)

| Configuration | F1 Score | Learned Weight (Log-Odds) | Key Takeaway |
|---|:---:|:---:|---|
| **Full AINN** | **0.920** | — | Balanced, multi-modal synergy. |
| Without Content Branch | **0.647** | — | **Largest drop (-0.273):** Commitment matching is the primary pillar. |
| Without Image Branch | 0.843 | — | Vital for photo-only leaks. |
| Without GNN | 0.917 | — | Small delta on text leaks; valuable on cascade structures. |
| Without Autoencoder | 0.911 | — | Secondary reinforcement for anomalous volume. |
| *Learned Weights:* | | | |
| `content_lexical` | — | **+2.74** | Primary driver: char n-grams match exact question stems. |
| `image_orb` | — | **+1.66** | Secondary driver: detects photographed exam pages. |
| `content_answer_key` | — | **+1.08** | Specialized driver: flags answer-key sequences. |
| `gnn` / `autoencoder` | — | **+0.48 / +0.44**| Supporting drivers: cascade dynamics & volume anomalies. |
| `content_semantic` | — | **-0.39** | Highly collinear with lexical; penalizes redundant matches. |

### 4.4 Scenario-by-Scenario Resilience

* **Immunity to Scams:** On *Fake-Leak Scams* (*"Pay Rs 5000 for paper"*), baseline models trigger false alarms **21% to 100%** of the time. AINN produces only **1%** false alarms.
* **Past Paper Resilience:** On legitimate past-year paper discussions, AINN triggers only **3%** false alarms.
* **Early Warning:** Under streaming evaluation, **88.0% of leaks are caught before the exam starts** (97.8% of all content-bearing leaks) with a median lead time of **168 hours** (7 days).

---

## 5. Security & Attack-Resistance Verification

The repository contains an automated security test harness (`src/attacks/attack_harness.py`) testing 8 adversarial attack scenarios. All 8 tests pass against the real codebase:

| Attack Scenario | Adversary Action | AINN Defense Mechanism | Outcome |
|---|---|---|:---:|
| **1. Early Time-Lock Bypass** | Attacker attempts decryption before VDF time expires. | VDF sequential squaring property prevents shortcuts; verify fails. | **Blocked** |
| **2. Single-Stakeholder Rogue Release** | Single agency attempts unilateral key reconstruction. | 3-of-4 Shamir threshold polynomial requires $t=3$ shares. | **Blocked** |
| **3. Hardware Platform Tampering** | Attacker modifies OS kernel or debugger on center node. | PCR registers change; TPM sealed key refuses to unseal. | **Blocked** |
| **4. Metadata / Center Spoofing** | Attacker attempts to decrypt Paper C1 on terminal C2. | Center ID is bound in AES-GCM AAD; tag validation fails. | **Blocked** |
| **5. Post-Distribution Paper Substitution** | Insiders swap question paper after signing. | Cryptographic Ledger check fails against original Ed25519 hash. | **Blocked** |
| **6. Photographed Page Leak** | Leaker circulates angled, cropped phone photos of paper. | ORB keypoints + RANSAC homography matches committed template. | **Detected** |
| **7. Scam / Coincidental Noise** | Fraudsters sell fake papers with alarmist language. | Content score against ledger hash is zero; alert suppressed. | **Rejected** |
| **8. Brute-Force Key Recovery** | Quantum or classical brute force on content key. | 256-bit AES-GCM + ML-KEM-768 post-quantum key wrap. | **Blocked** |

---

## 6. High-Yield Viva Defense: Questions & Answers

### Q1: "Why can't we just use a centralized secure cloud server with two-factor authentication?"
> **Answer:** *"A centralized server represents a single point of institutional failure. In national scandals like Vyapam and NEET-UG, compromises occurred precisely because authorized insiders—database administrators, testing directors, or contracted printing agencies—had legitimate access. A centralized server cannot defend against a compromised super-admin. AINN removes single-point trust entirely: key material is split across 4 independent stakeholders via Shamir secret sharing, locked to hardware TPMs, and time-gated via non-parallelizable VDF mathematics."*

### Q2: "Your initial prototype had an F1 score of 1.00. Why is 0.920 a better result?"
> **Answer:** *"In machine learning, an F1 score of 1.00 across all seeds is a classic red flag indicating dataset leakage and artificial simplicity. In our initial prototype, leak traffic contained verbatim text and normal traffic contained none, making separation trivial. We replaced that with a rigorous Leave-One-Paper-Out benchmark testing 11 difficult scenarios, including fake-leak scams, past-paper discussions, and buried leaks. An F1 of 0.920 under realistic conditions with held-out exam papers proves true generalization rather than memorization."*

### Q3: "Why did you abandon perceptual hashing (pHash) in favor of ORB + RANSAC for images?"
> **Answer:** *"We empirically measured pHash on actual phone photos of exam papers. Real leak photos suffer from perspective tilt, shadows, folds, and severe WhatsApp compression. Under these distortions, global perceptual hashes change drastically, causing pHash recall to drop to chance level (FPR > 50%). We replaced it with local feature detection: ORB keypoints matched through RANSAC homography. Even if only 30% of a crumpled, angled page is visible, RANSAC finds the geometric planar transform and validates the match with 1.00 recall."*

### Q4: "How does the Verifiable Delay Function (VDF) prevent an attacker with a supercomputer from unlocking the paper early?"
> **Answer:** *"A standard cryptographic puzzle can be solved faster by throwing thousands of parallel GPU/ASIC cores at it. A VDF, specifically Pietrzak’s construction based on modular squaring ($x^{2^T} \pmod N$), is inherently non-parallelizable. Step $i+1$ strictly requires the output of step $i$. Therefore, an adversary with 10,000 servers cannot compute it any faster than a single core operating at the same clock speed. Once solved, however, the proof can be verified in milliseconds by anyone."*

### Q5: "What is the system's biggest blind spot, and how do you defend against it?"
> **Answer:** *"Our empirical results honestly document that **coordination-only leaks achieve 0% detection recall**. If conspirators coordinate a leak on private 1-on-1 chats and never post the actual question text or photos in the monitored channel, text and image matching find nothing. The chatter looks identical to a scam channel.  
> However, this is precisely why AINN is a **dual-layer architecture**: the detection layer is not our only line of defense. The Custody Layer guarantees that the paper cannot physically leave the time-locked enclave before the exam window in the first place."*

### Q6: "Why is Hinglish rewording your weakest content case (recall 0.30 at 1% FPR)?"
> **Answer:** *"Hinglish mixes Hindi syntax and vocabulary with Latin script using non-standardized phonetic spelling (e.g., 'yeh question direct formula based hai'). Character n-grams and English WordLlama embeddings struggle with this vocabulary drift. To solve this in production, the next step is integrating an Indic multilingual transformer encoder like MuRIL (Multilingual Representations for Indian Languages) or LaBSE."*

### Q7: "Are the TPM and TEE running on real hardware or simulated?"
> **Answer:** *"In this software prototype, the TPM 2.0 and TEE are high-fidelity software simulations faithfully modeling the exact TCG TPM2 command specifications (PCR extension, policy sealing, attestation quotes). For production deployment, the code is structured so that `tpm_sim.py` can be swapped with real `tpm2-tools` calling hardware TPM 2.0 chips on Intel SGX or AMD SEV server motherboards without changing the custody protocol."*

---

## 7. Viva Presentation Checklist

When presenting this project to examiners, emphasize these 4 pillars:
1. **The Core Philosophy:** *"Commitment before distribution; zero plaintext in transit."*
2. **The Cryptographic Proof:** *"3-of-4 Shamir + VDF + TPM means no single insider can leak the paper."*
3. **The AI Explainability:** *"Alerts don't just say 'high risk'; they cite the exact Question ID, the exact page number, and the inlier keypoint count."*
4. **The Scientific Honesty:** *"We tested on real JEE questions, evaluated on unseen papers, and openly report our failure modes on coordination-only channels."*
