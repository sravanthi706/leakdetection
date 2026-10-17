# AINN — Tamper-Evident, Time-Locked Exam Paper Distribution System
## Master Document (Project State + Patent Strategy + Full Technical Specification)

*Single source of truth. Read this file first in any new session — no other doc needs to be
opened unless you want the original patent-strategy phrasing verbatim (superseded content below).*

---

# PART A — PROJECT STATE

## What this is

A patent-strength architecture, **now fully built as a working prototype**, for distributing exam
papers so that no single person, device, or server — including authorized personnel — can access
the plaintext before the exam window opens, with a parallel AI/NN layer detecting leaks in real
time on public channels. This is an upgrade of an original academic proposal that was assessed as
"a strong project proposal but a weak patent application" (see Part B for why).

## Status: 18/18 build items complete + ML upgrade (v2) complete

Every module below is real, working code — no pseudo-code, no placeholders — and every module has
its own self-test (`python3 -m src.<package>.<module>` runs it directly). All of it lives under
`src/`; see `README.md` at the project root for the exact folder layout and how to run each piece.

**Custody chain** (the 9-step flow in Part C.6, wired together in `src/orchestrator.py`):
- AES-256-GCM content encryption with AAD-bound metadata (`src/crypto/aes_gcm.py`)
- ML-KEM-768 post-quantum key wrapping (`src/crypto/ml_kem_wrap.py`)
- Shamir's Secret Sharing, real GF(2⁸) arithmetic (`src/crypto/shamir.py`)
- Hash-chained, Ed25519 co-signed commitment ledger (`src/custody/commitment_ledger.py`)
- Simulated TPM 2.0 (PCR extend/seal/unseal/quote) (`src/custody/tpm_sim.py`)
- Simulated TEE enclave boundary (`src/custody/tee_enclave.py`)
- Verifiable Delay Function, Pietrzak's construction (`src/custody/vdf.py`)
- Conditional release AND-gate (VDF ∧ TPM/TEE) (`src/custody/conditional_release.py`)
- Active tamper response / dead-man's switch (`src/custody/tamper_response.py`)

**Detection layer** (four independent signals fused in `src/detection/fusion.py`):
- Legal, cited-and-calibrated synthetic dataset (`src/detection/synthetic_data.py`)
- Propagation-graph GNN, hand-rolled NumPy GraphSAGE (`src/detection/gnn.py`)
- TF-IDF character n-gram content matching (`src/detection/content_match.py`)
- Dense autoencoder anomaly detector (`src/detection/autoencoder.py`)
- Perceptual-hash (pHash) image matching (`src/detection/phash_match.py`)

**Integration and validation:**
- End-to-end orchestrator running the real 9-step custody flow + detection layer together
  (`src/orchestrator.py`)
- Attack-resistance harness — 8 adversarial scenarios mirroring Part C.7's table, all currently
  passing against the real modules (`src/attacks/attack_harness.py`)
- Report/dashboard — one CLI + PNG summary per run (`src/report/dashboard.py`)

**ML upgrade (v2) of the detection layer.** Full details are in `docs/ML_REPORT.md`.

- **Real data**:
  - real JEE-Advanced papers (JEEBench, 515 questions) as the committed papers, with 12 past
    papers as legitimate look-alikes;
  - real Twitter15/16 rumour cascades for the GNN benchmark;
  - MSRP for paraphrase;
  - 80 hand-written paraphrases (English and Hinglish) of real questions.
- **Hard session generator** with 11 scenarios. It includes scams, coaching broadcasts and past-paper
  discussion on the normal side, and quiet, buried, paraphrased, answer-key-only, photo-only and
  coordination-only leaks on the leak side.
- **Content branch**: per-question lexical + WordLlama semantic + answer-key matching against the
  committed paper, calibrated on legitimate traffic.
- **Image branch**: realistic phone-photo simulation and verified ORB/RANSAC page matching. pHash was
  measured at chance on phone photos and removed from fusion.
- **Fusion and training**: learned stacking fusion, a validation-chosen threshold, streaming-aware
  training, and explanations naming the matched question, page and signal contributions.
- **Optional PyTorch Geometric GNN** with GNNExplainer.
- **Evaluation**:
  - leave-one-paper-out CV, 8 folds, 2,400 test sessions;
  - 4 baselines, ablations, label noise, calibration and streaming time-to-detection;
  - a held-out harsh-photo pool;
  - a BiGCN re-implementation on real cascades.
- **Headline results**:
  - F1 0.920 ± 0.016 against 0.735 for the best baseline;
  - 97.8% of content-bearing leaks flagged before the exam, with a 1.5% streaming false-alarm rate;
  - BiGCN 0.820 / 0.844 on Twitter15/16.
- **Live demo**: `python -m src.demo.live_demo` commits a real paper to the ledger, replays a leak
  and a scam channel, and prints an explained alert.

## Honest, documented limitations (state these if asked — don't let them go unmentioned)

- TPM and TEE are software models of the real interfaces, not hardware.
- The VDF's RSA modulus has no multi-party trusted-setup ceremony. A production deployment needs one,
  or a class-group construction that needs none (e.g. Chia's `chiavdf`).
- **Detection sessions are simulated.** The question text, answer keys and Twitter cascades are
  real, but who-posts-what-when is modelled. The shape is calibrated to cited NEET-UG 2026 reporting,
  and no real leak-channel data was used or scraped. Numbers on real monitoring data will be lower.
- **Coordination-only leaks are not detected (0% recall).** If the paper itself never appears in a
  monitored channel, the talk around it looks like a scam.
- **Hinglish rewording is the weakest content case** (recall 0.30 at 1% FPR).
- **The semantic model is small.** Hugging Face was blocked in the build environment, so the semantic
  model is WordLlama rather than a transformer encoder. MiniLM is supported if it can be downloaded
  locally.
- **The structure-only GNN is a weak signal**: +0.003 F1 in ablation, and near chance on real
  Twitter cascades. It is kept as a supporting signal, not a primary detector.
- **Phone photos are simulated.** `scripts/eval_user_photos.py` tests real ones.
- The NumPy GNN/autoencoder remain as dependency-free defaults. PyTorch Geometric is an optional
  backend (`requirements-ml.txt`).
- No patent has been filed. No real institutional deployment exists — getting four real,
  mutually-distrusting stakeholders to run enrolled, attested infrastructure is a governance
  problem, not a coding one.

## Working style for this project

- Brief, directive answers. Minimal unnecessary explanation.
- When teaching concepts, one at a time with a check-in — not a full dump — *unless* explicitly
  asked for the full picture at once.
- User will personally defend this work — accuracy and the user's own understanding matter more
  than speed of output. Every honest substitution above must be something the user can explain, not
  just something that runs.

---

# PART B — PATENTABILITY ASSESSMENT & STRATEGY

## B.0 Direct answer

As originally written, the proposal was **a strong project/research proposal but a weak patent
application**. Not because the idea is bad — the diagnosis (insider access, not external hacking,
is the real failure mode in NEET/Vyapam/SSC) is accurate and well-argued. The problem: the three
original technical pillars — AES-256/RSA/ECC encryption, TPM 2.0 attestation, LSTM/Autoencoder
anomaly detection — are individually well-established, decades-old techniques. A patent examiner's
first move would be an **obviousness objection**: each component is known prior art; applying known
components to a new domain (exam papers) is not by itself an inventive step.

## B.1 Why the original version was patent-fragile

India's Patent Office released updated **Computer-Related Invention (CRI) guidelines in 2025**. The
operative test is not "does software touch hardware" — it's whether the invention produces a
**concrete, measurable technical effect** (the guidelines name examples: faster processing, reduced
memory use, improved throughput, lower latency, enhanced encryption, real-time hardware control),
and separately whether combining known elements clears **Section 2(1)(ja) inventive step** (not
merely obvious to a person skilled in the art).

Two concrete risks in the original draft:

- **"Use AES-256 + RSA/ECC + TPM"** — naming standard algorithms without a novel protocol structure
  around them does not by itself show inventive step. Time-locked decryption gated on a trusted
  timestamp is known (time-lock puzzles date to 1996; TPM-sealed keys are a standard TPM use case).
  A *specific new protocol combination* is needed, not a list of standard primitives.
- **"Use an LSTM/Autoencoder for anomaly detection"** — the 2025 guidelines explicitly raise the bar
  for AI/ML claims: architecture detail, dataset characteristics, training parameters, and
  validation results are required, not a use-case description.

None of this means abandon the idea — it means add at least one genuinely novel mechanism so the
claims have something concrete and non-obvious to point at. That is what Part C now reflects in full.

## B.2 Six upgrades, ranked by inventive-step impact (adopted into the design in Part C)

### Tier 1 — adopted; each is a real inventive step, buildable with current, real technology

**(A) Threshold (t-of-n) secret-shared key release — replaces a single trusted server.**
Highest-impact change, targeting the actual documented failure mode: NEET 2026 and Vyapam were
institutional collusion, not "one rogue printer." A design with one Time-Lock Key Server, however
well-secured, is still a single point of institutional trust — exactly what failed in Vyapam.
Splitting the key via Shamir's Secret Sharing across independent, mutually distrusting stakeholders,
each requiring independent TPM-attested release, is a specific, non-obvious combination and maps
directly onto the case-study evidence. Technical-effect claim: reduces single-point insider
compromise risk from N potential actors to requiring collusion among t independent stakeholders.

**(B) Blockchain-anchored commit–reveal for provable paper integrity.**
Publishing `hash(paper)` to an append-only, timestamped ledger before distribution gives something
NEET 2026 investigators did not have: a mathematically verifiable answer to "does this leaked paper
match the real one, or is it a coincidental guess-paper overlap." Proof, not testimony — a concrete
technical mechanism tied directly to the case study.

**(C) Tamper-response "self-erasing" sealed key (active dead-man's switch).**
Standard TPM sealing passively refuses to unseal on a PCR mismatch. Adding an explicit policy — `k`
failed attestation attempts or detected mid-session drift triggers active destruction of the sealed
key plus automatic fallback to a secondary channel — is a specific, claimable mechanism beyond "use
TPM sealing."

### Tier 2 — adopted where noted; stronger differentiation, higher implementation cost, good continuation-patent material

**(D) Multi-modal detection — extend the NN layer from text to photographed-page matching. ADOPTED.**
Most real leaks eventually involve someone photographing the physical paper; text-only detection
can't catch that. A perceptual-hash (pHash) layer, matched against a committed image hash from (B),
is a specific, novel combination — most literature treats social-media leak detection and
image-similarity detection as separate problems.

**(E) Geo-fenced device attestation. Noted, not adopted in the current design.**
Binding unsealing to proof of physical presence at the registered exam-center coordinates would
close the "insider takes the device/key material off-site" gap that pure TPM attestation doesn't
address. Flagged as future work.

**(F) Zero-knowledge request proofs. Noted, not adopted in the current design.**
Would let center devices prove "I am an enrolled, valid share-holder" without revealing which
`paper_id` they're requesting, closing a metadata-leak vector. Flagged as the heaviest lift of the
six and lowest priority — future work.

## B.3 Titles

- **For patent filing (short, examiner-conventional):**
  *"System and Method for Threshold-Secured, Hardware-Attested, Time-Locked Distribution of
  Confidential Examination Content with Real-Time Leak Detection"*
- **For the research paper / thesis (descriptive, keeps the case-study framing):**
  *"A Threshold-Cryptographic, Blockchain-Anchored, Hardware-Attested Time-Lock Architecture with
  Multi-Modal Neural Anomaly Detection for Tamper-Evident Examination Paper Distribution —
  Addressing Insider-Driven Leaks in India's National Examination System"*

Use the short version on the patent application; keep the descriptive version for the research
report/thesis cover page.

## B.4 Time-sensitive filing note — read before submitting or publishing anything

India follows an **absolute novelty** standard: public disclosure before filing (a published paper,
a submitted project report placed in a public repository, a public presentation) can itself destroy
patentability, subject only to narrow statutory exceptions (Section 31 — certain
government-recognized exhibitions/learned-society disclosures, generally requiring the application
within 12 months of that specific disclosure). If this is going to be submitted as a college
research project, published, or uploaded anywhere public, the safer sequence is: file at least a
**provisional application** capturing the Tier-1 upgrades first, *then* submit/publish the research
report. A provisional does not require a working prototype — it requires a clear written description
of the mechanism. This is general information, not legal advice — a registered patent agent should
review the actual claims before filing.

## B.5 Phased roadmap (patent + build, combined)

- **Phase 0 (before wider disclosure):** file a provisional patent application capturing the Tier-1
  combination (threshold release + TPM/TEE attestation + commitment ledger) to lock a priority date.
- **Phase 1:** reference software prototype — encryption + threshold split/reconstruct + simulated
  per-node TPM/TEE attestation + hash-chained commitment ledger + baseline autoencoder anomaly
  detector on synthetic data.
- **Phase 2:** swap simulated TPM for real `tpm2-tools`-based attestation on actual hardware for at
  least one node; run multi-stakeholder threshold reconstruction across physically separate machines.
- **Phase 3:** add perceptual-hash image detection; pilot commitment-ledger anchoring; tune anomaly
  thresholds against a larger synthetic corpus.
- **Phase 4:** complete specification and file the full (non-provisional) patent application within
  12 months of the provisional, with architecture diagrams, quantified technical-effect claims, and
  full NN disclosure (architecture, dataset characteristics, training protocol, validation results)
  per the 2025 CRI guidelines.

---

# PART C — COMPLETE TECHNICAL SPECIFICATION

## C.1 Overview

This specifies a complete system for distributing exam question papers such that no party in the
distribution chain — including officially authorized personnel — can access the plaintext paper
before the exam window opens, and any leak attempt is detected in real time. The system is built
around three independent layers that must each fail for a leak to succeed before the exam window: a
cryptographic layer, a hardware- and protocol-rooted custody layer, and a neural-network-based
detection layer that runs in parallel and does not depend on the other two.

The design responds directly to documented insider-driven failures in India's national examination
system — the NEET-UG 2026 cancellation (2.27 million candidates, arrests including National Testing
Agency personnel), the Vyapam scandal (2013), and recurring SSC/RRB paper leaks — where the common
failure point was not external hacking but legitimate personnel in the setting, printing, or
distribution chain gaining early access and disclosing content before the exam.

**Core design principle:** no single person, device, or server is ever in a position to produce the
plaintext paper early. Release requires independent agreement across multiple mutually-distrusting
parties, a hardware-attested device state, and a cryptographically verifiable proof that time has
genuinely elapsed — none of which any one participant, however senior or trusted, can produce alone.

## C.2 Threat model

| Actor | Access level | Motivation |
|---|---|---|
| Paper setter / reviewer | Sees plaintext during authoring, before packaging | Bribery, coercion, ideology |
| Printing press staff | Handles encrypted copies before distribution | Bribery, organized leak rings |
| Distribution / logistics personnel | Transports encrypted packets or devices to centers | Bribery, opportunistic leak |
| Exam center staff | Holds device/key material immediately before exam start | Time-pressure leak (sell before window opens) |
| External attacker | No legitimate access | Network/endpoint compromise, social engineering |

The system assumes the attacker already holds a valid role, valid device, and valid credentials — it
does not rely on keeping outsiders out; it removes the technical *capability* of an insider to
produce plaintext early, even when willing to act against their role.

Out of scope: coercion of proctors after the window has legitimately opened (the paper is plaintext
by design at that point); post-leak criminal investigation beyond what the audit trail and detector
provide; availability/DDoS attacks on distribution infrastructure.

## C.3 Cryptographic layer

**C.3.1 Content encryption.** The finalized paper is authored and encrypted on an **air-gapped
system** — plaintext never touches a networked device at any point before the exam window. Content
is encrypted with **AES-256-GCM**, producing ciphertext, a nonce, and an authentication tag bound to
the paper's metadata (paper ID, exam window start, center ID). AES-GCM is authenticated encryption,
so tampering with the ciphertext is detected, not merely prevented.

**C.3.2 Post-quantum key wrapping.** The 256-bit AES content key is wrapped using **ML-KEM
(FIPS 203)**, NIST's finalized post-quantum key-encapsulation standard, rather than classical RSA or
ECC. This is deliberate given the multi-year operational lifetime of exam infrastructure: RSA/ECC-
wrapped keys captured today remain vulnerable to "harvest-now, decrypt-later" attacks once
cryptographically relevant quantum hardware matures; ML-KEM does not carry that exposure.

**C.3.3 Commitment ledger.** Before distribution begins, a cryptographic hash of the finalized
paper — `hash(paper)` — is published to an append-only, hash-chained commitment ledger and
timestamped. This is not encryption; it reveals nothing about the paper's content. Its purpose is
post-incident verification: if a leaked paper surfaces after the fact, anyone (an auditor, a court,
the public) can hash the disputed document and check it against the pre-committed value, turning
"does this leaked paper match the real one" into a checkable fact rather than a matter of testimony.
The ledger does not require a public blockchain — a permissioned, hash-chained log co-signed by the
independent custody stakeholders (C.4) is sufficient, with the option to periodically anchor Merkle
roots to a public chain for external, court-admissible verifiability.

## C.4 Custody layer — threshold, hardware-attested, time-locked

**C.4.1 Threshold key custody.** The wrapped content key is not held by any single server or
operator. It is split via **threshold secret sharing (t-of-n, Shamir's Secret Sharing)** across `n`
independent, mutually-distrusting stakeholders — for example, NTA HQ, the relevant State Exam
Authority, an independent audit body, and a custodian. Reconstructing the key requires shares from
at least `t` of these `n` parties. No single bribed official, coerced employee, or compromised
server can release the paper early — this directly targets the documented failure mode in Vyapam
and NEET 2026, which were institutional collusion, not a single rogue actor.

**C.4.2 Hardware-rooted device trust.** Each stakeholder's share is sealed to that stakeholder's own
device state using two complementary hardware mechanisms:

- **TPM 2.0** provides device identity and boot-integrity attestation — PCR values captured at boot
  prove the device is running unmodified firmware and boot chain, and sealing
  (`TPM2_Seal`/`TPM2_Unseal`) means the share is only unsealable when current PCRs match the
  enrolled policy.
- **TEE (Trusted Execution Environment)** — Intel TDX, AMD SEV-SNP, or ARM CCA — provides an
  encrypted, isolated runtime enclave. Where TPM proves *what booted*, the TEE additionally protects
  the decrypted key material *while it is in use*, so that even someone with root/administrator
  access to the host OS cannot dump memory to recover it. TPM and TEE address different points in
  the attack surface and are used together, not as substitutes for one another.

**C.4.3 Verifiable Delay Function time-lock.** Rather than trusting any single server's clock to
decide "has the exam window opened," each stakeholder node independently computes a **Verifiable
Delay Function (VDF)** proof — a computation that is provably sequential (no amount of parallel
hardware can shorten it) and cheaply verifiable by anyone. The VDF proof demonstrates that a genuine
amount of elapsed time has passed since a reference point recorded in the commitment ledger, without
requiring trust in any operator's reported clock. (In practice, each node's VDF computation is
anchored to the ledger's timestamped commitment as its starting reference — this removes trust in
any single party's *real-time* clock, though the system as a whole still relies on the initial
commitment timestamp being honest, which is why that timestamp is itself published and co-signed
across the independent stakeholders rather than set by one party.)

**C.4.4 Conditional release.** A stakeholder node releases its share only when **both** conditions
hold simultaneously: the VDF proof confirms elapsed time, and the node's own TPM+TEE attestation
succeeds. Either condition failing blocks release — an insider with a valid credential on a tampered
device is refused by the TPM/TEE check even if the time condition is satisfied; a request before the
window opens is refused by the VDF check even from an untampered, fully legitimate device.

**C.4.5 Reconstruction and decryption.** At the exam center, `t`-of-`n` released shares are combined
**inside a TEE enclave** — the reconstructed content key never exists in ordinary, unprotected
memory at any point. The AES-256-GCM content key is recovered and the paper is decrypted. Plaintext
exists only from this moment onward, and only for the duration of the exam window.

**C.4.6 Tamper response.** TPM sealing is passive by default — it simply refuses to unseal on a
state mismatch. This system adds an active response: after `k` failed attestation or unseal
attempts, or a detected mid-session PCR/state drift, the locally sealed share is **actively
destroyed**, and the node automatically triggers a fallback release through a secondary channel with
fresh key material. Tampering becomes a logged event with a defined recovery path, not a silent,
unresolved failure.

## C.5 AI / Neural Network detection layer

This layer runs continuously and independently of the custody chain in C.3–C.4 — it does not wait
on, and is not weakened by, anything happening there. It exists to catch a leak in progress on
monitored public and semi-public channels (principally Telegram; WhatsApp's end-to-end encryption
limits equivalent visibility there without separate legal/access mechanisms outside this system's
scope), including leaks that originate outside the formal distribution chain — a photographed page,
for instance.

**C.5.1 Text pipeline.** Monitored message text feeds three parallel branches, all reading from the
same input stream:

- **Propagation graph construction.** Messages are modeled as a graph — nodes are accounts/groups,
  edges are share/forward events — because a leak spreading is structurally a network-propagation
  phenomenon (who shared with whom, how fast, across how many groups), not a simple time series of
  message counts.
- **Sentence embedding.** A pretrained transformer encoder (used frozen, not trained on project
  data) produces a semantic embedding of each message, enabling comparison against the committed
  paper's own embedding — this catches paraphrased or reworded leaks, not just exact keyword matches.
- **Feature extraction.** Classical signal features — keyword density, message-rate burst,
  hour-of-day, sender-account novelty — feed the reconstruction-based baseline detector.

**C.5.2 Models.** Two components are literal trained neural networks; everything else in this layer
is classical signal processing feeding them or combining their outputs.

- **Graph Neural Network** (2–3 GraphSAGE/GCN layers) consumes the propagation graph and produces a
  cascade embedding, scored for anomalous spread patterns — the primary novel-signal detector, since
  it models the actual shape of a leak spreading rather than aggregate volume.
- **Autoencoder** (dense encoder–decoder, ~8-dimensional bottleneck) is trained only on normal
  traffic; reconstruction error spikes on out-of-distribution (leak-pattern) input. This suits the
  real deployment constraint that confirmed leak examples are rare, while normal traffic is abundant.
- **Cosine similarity** between a message's embedding and the committed paper's embedding produces a
  content match score (classical computation, not a trained model).

Both trained models are trained offline by a **synthetic data generator**, since no public dataset
of real exam-leak chat traffic exists. The generator produces labeled sessions: a normal baseline
(steady low-volume exam-prep chatter) and injected leak-pattern anomalies (late-night bursts, high
specificity, high cross-group spread velocity) matching the pattern described in the original threat
analysis. This is a documented stand-in for real deployment telemetry, which would replace it once a
monitoring program with appropriate legal authorization exists.

**C.5.3 Image pipeline.** Monitored images are processed with **perceptual hashing (pHash,
DCT-based)** and compared by Hamming distance against the paper's committed image hash. This is
classical signal processing, not a trained model, and it closes the gap that text-only monitoring
cannot: it catches a photographed physical page circulating online, which is the dominant
real-world leak vector in practice.

**C.5.4 Fusion and alerting.** Four independent scores — cascade anomaly score (GNN), content match
score (embedding similarity), reconstruction error score (autoencoder), and image match score
(pHash) — combine in a weighted ensemble fusion into a single anomaly score. Scores above a tuned
threshold trigger an alert to the monitoring team with per-signal feature attribution, so the output
feeds a human decision rather than an automated takedown.

## C.6 End-to-end data flow

Nine sequential steps carry a paper from authoring to exam-time decryption; the detection layer runs
in parallel from step 3 onward, independent of everything after it.

1. **Air-gapped authoring.** The paper is finalized on a system that is never networked; plaintext
   never leaves it.
2. **AES-256-GCM encryption.** Produces ciphertext and the content key `K`.
3. **Commit to ledger.** `hash(paper)` is published and timestamped, before distribution begins.
4. **Wrap and split the key.** `K` is wrapped with ML-KEM and split via threshold secret sharing
   into `share₁…shareₙ`.
5. **Seal to stakeholder nodes.** Each share is bound to its stakeholder's TPM+TEE state.
6. **VDF time proof.** Each node independently computes a proof that the exam window has genuinely
   elapsed.
7. **Conditional release.** A share is released only if the VDF proof and the node's TPM/TEE
   attestation both hold.
8. **Threshold reconstruction.** `t`-of-`n` released shares combine inside a TEE enclave at the exam
   center; `K` is recovered only in protected memory.
9. **Decrypt.** The paper becomes plaintext — existing only from this point, only for the exam
   window.

**Parallel, from step 3 onward:** the detection layer continuously scores monitored channels against
the committed hash and alerts independently of whether steps 4–9 ever run to completion or are ever
breached.

## C.7 Attack resistance

| Scenario | Mechanism exercised | Result |
|---|---|---|
| Early access attempt | Request release before the window opens | VDF proof fails → no node releases its share |
| Unauthorized device | Present an attestation from a device/key never enrolled | Stakeholder verification fails → release refused |
| Tampered platform | Modified exam-client binary changes measured PCR/TEE state | TPM/TEE refuses to unseal even with a valid VDF proof and a valid release request — demonstrates the AND condition in C.4.4 |
| Key extraction / replay | Reuse a previously valid attestation or VDF proof | Nonce/freshness checks reject the replay |
| Repeated tamper attempts | `k` failed attestations at one node | Local share is actively destroyed; fallback channel triggers automatically (C.4.6) |
| Social-media leak | Leak-pattern content appears on monitored channels | Ensemble anomaly score exceeds threshold → alert, independent of custody-chain state |
| Photographed page | Image of the physical paper circulates | pHash match against the committed image hash → alert, catching what text monitoring alone would miss |
| Post-incident dispute | A leaked document surfaces after an exam | `hash(leaked document)` checked against the pre-committed ledger value → verifiable match/non-match, not testimony |

## C.8 Technology stack

Every primitive below is available and implementable today; the substantial remaining work is the
protocol that connects them and the institutional agreement to operate it, not invention at the
component level.

| Component | Concrete technology |
|---|---|
| Content encryption | AES-256-GCM (standard library / `pycryptodome`) |
| Key wrapping | ML-KEM — FIPS 203 (post-quantum KEM implementations, e.g. liboqs bindings) |
| Threshold secret sharing | Shamir's Secret Sharing (`pycryptodome`, dedicated SSS libraries, or HashiCorp Vault's implementation as a reference model) |
| Device identity / boot integrity | TPM 2.0 via `tpm2-tools` / `tpm2-pytss` (Linux TSS2 stack) or Windows Platform Crypto Provider / TBS API |
| Runtime enclave | Intel TDX, AMD SEV-SNP, or ARM CCA (shipping confidential-computing hardware) |
| Time-lock proof | Verifiable Delay Function — Wesolowski or Pietrzak construction |
| Commitment ledger | Hash-chained append-only log, co-signed by custody stakeholders; optional periodic anchoring to a public chain (OpenTimestamps-style) |
| Graph Neural Network | PyTorch Geometric or DGL, 2–3 GraphSAGE/GCN layers |
| Sentence embedding | Pretrained transformer encoder (e.g. a MiniLM-class sentence-transformer), used frozen |
| Autoencoder | Dense encoder–decoder in PyTorch or scikit-learn-based, ~8-d bottleneck |
| Perceptual hashing | `imagehash` (Python, pHash/DCT-based) |
| Synthetic training data | Custom generator producing labeled normal / leak-pattern message-stream sessions |

Further upgrade path, noted but not required for this specification: replace Shamir reconstruction
(which briefly assembles `K` in protected memory) with true **MPC-based threshold decryption**,
where the key is never reconstructed in any single location at all.

## C.9 Deployment considerations

- **Connectivity.** Many exam centers cannot rely on real-time connectivity to stakeholder nodes at
  national exam-day scale. An offline variant — a pre-signed, time-boxed release sealed to a local
  device's monotonic counter/measured-boot state — is the field-mode default, with online release as
  the high-assurance path where connectivity exists.
- **Hardware baseline.** TPM 2.0 is standard on most post-2016 business/education-grade hardware but
  not universal; TEE-capable hardware is less widespread still. A phased rollout starting with the
  highest-stakes exams and centers already meeting the hardware baseline is more realistic than a
  simultaneous national deployment.
- **Legal basis for channel monitoring.** The detection layer's data source requires a defined legal
  and privacy framework — what is monitored, under what authorization, what is retained — before
  field deployment. This is a policy prerequisite, not a technical one, and is the hardest
  non-technical blocker in the whole system.
- **Institutional adoption.** Threshold custody only works if independent stakeholders genuinely
  agree to run enrolled, attested infrastructure and cede unilateral control — a governance
  negotiation, not an engineering task, and realistically the largest source of schedule risk in an
  actual deployment.
- **Non-goals.** The system does not prevent leaks after the exam window has legitimately opened
  (the paper is necessarily plaintext at that point), and it does not replace personnel integrity
  vetting — it removes the technical *opportunity* for early leak, which is the documented common
  factor across NEET 2026, Vyapam, and SSC/RRB.

---

# PART D — DIAGRAMS (shipped as separate image files, not embedded here)

Three diagrams exist as standalone PNG/SVG files and should be uploaded to the same Project
knowledge base alongside this document:

- `threshold-locked-architecture-diagram.png` / `.svg` — full system architecture: authoring →
  encryption → ML-KEM → commitment ledger, four-stakeholder threshold custody row (each with
  TPM/TEE/VDF sub-boxes and dead-man's-switch note), reconstruction, plaintext output, and the
  detection layer row.
- `ai-nn-detection-layer-diagram.png` / `.svg` — detailed AI/NN pipeline: input ingestion → four
  parallel branches (propagation graph/GNN, sentence embedding/cosine similarity, classical
  features/autoencoder, pHash) → score fusion & attribution. Amber = literal trained neural network
  (GNN, autoencoder); blue = classical/pipeline component.
- `data-flow-diagram.png` / `.svg` — the 9-step data flow from C.6 above, with the parallel
  "ground truth" branch into the detection layer, dark theme to match the presentation deck.

---

*End of master document. This file supersedes `PROJECT_STATE.md`, `COMPLETE_IMPLEMENTATION.md`,
`PATENT_STRATEGY.md`, and `THRESHOLD_LOCKED_ARCHITECTURE.md` as the single upload target — those
four files are now redundant with this one and don't need separate uploading.*
