"""
Attack simulation harness -- exercises every row of the spec's Attack
Resistance table (master doc section C.7) against the REAL modules
built in items 1-23, not a mocked or pseudo version of them.

Each scenario function below calls directly into src.crypto /
src.custody / src.detection / src.orchestrator -- this file's only job
is constructing the adversarial input or condition for each row and
checking the real module's response against what the spec claims
should happen. No scenario asserts a result from a config flag: every
"attack" is actually attempted against the real primitive (a
deliberately wrong VDF proof, a forged signature, a tampered PCR
state, a replayed nonce, k genuinely failed unseal attempts, a
text-neutralized leak session, a fabricated document hash, ...).
"""

import os
import random

from src.crypto.aes_gcm import decrypt_paper
from src.custody.commitment_ledger import CommitmentLedger, generate_stakeholder_keypair, hash_paper
from src.custody.tpm_sim import SoftwareTPM, verify_quote
from src.custody.vdf import (
    generate_vdf_parameters, derive_vdf_input,
    evaluate as vdf_evaluate, prove as vdf_prove, verify as vdf_verify,
)
from src.custody.conditional_release import check_release
from src.custody.tamper_response import TamperResponseGuard, ShareDestroyedError
from src.orchestrator import run_custody_flow, build_trained_fusion_engine


class ScenarioResult:
    def __init__(self, scenario, mechanism, expected, passed, detail):
        self.scenario = scenario
        self.mechanism = mechanism
        self.expected = expected
        self.passed = passed
        self.detail = detail


# ---------------------------------------------------------------------
# Shared VDF parameters -- generated ONCE and reused across scenarios
# that need one, since RSA modulus generation is the slow part and no
# test below depends on distinct moduli across scenarios.
# ---------------------------------------------------------------------
_VDF_PARAMS = None


def _vdf_params():
    global _VDF_PARAMS
    if _VDF_PARAMS is None:
        _VDF_PARAMS = generate_vdf_parameters(modulus_bits=1024)  # smaller modulus, fine at demo speed
    return _VDF_PARAMS


def _fresh_sealed_node(device_id="TEST_NODE", secret=None):
    tpm = SoftwareTPM(device_id=device_id)
    tpm.extend(0, f"firmware-hash-{device_id}".encode())
    tpm.extend(1, f"os-kernel-hash-{device_id}".encode())
    secret = secret if secret is not None else os.urandom(48)
    sealed = tpm.seal(secret, pcr_indices=[0, 1])
    return tpm, sealed, secret


# ---------------------------------------------------------------------
# Row 1: Early access attempt
# ---------------------------------------------------------------------
def scenario_early_access_attempt():
    params = _vdf_params()
    N = params["N"]
    T = 2 ** 12
    anchor = b"ledger_entry:exam_window_A"
    x = derive_vdf_input(anchor, N)

    tpm, sealed, secret = _fresh_sealed_node("EARLY_ACCESS_NODE")

    # Attacker claims elapsed time T, but genuinely only computed T//2
    # sequential squarings -- i.e. tries to request release before the
    # window has actually elapsed.
    y_early = vdf_evaluate(x, T // 2, N)
    proof_early = vdf_prove(x, y_early, T // 2, N)

    result = check_release(tpm, sealed, N, x, y_early, T, proof_early)  # claims T, proof only covers T//2

    passed = (result["released"] is False and result["vdf_ok"] is False and result["tpm_ok"] is True)
    detail = (f"claimed T={T}, proof only genuinely covers T//2={T // 2} squarings -> "
              f"vdf_ok={result['vdf_ok']} tpm_ok={result['tpm_ok']} released={result['released']}")
    return ScenarioResult(
        "Early access attempt", "Request release before the window opens",
        "VDF proof fails -> no node releases its share", passed, detail,
    )


# ---------------------------------------------------------------------
# Row 2: Unauthorized device
# ---------------------------------------------------------------------
def scenario_unauthorized_device():
    # (a) Ledger enrollment check: an id never enrolled as a stakeholder
    # cannot co-sign a commitment at all.
    _legit_priv, legit_pub = generate_stakeholder_keypair()
    ledger = CommitmentLedger({"NTA_HQ": legit_pub})
    attacker_priv, _attacker_pub = generate_stakeholder_keypair()  # never added to the ledger's known keys

    ledger_rejected = False
    try:
        ledger.append(data={"paper_id": "TEST"}, cosigners={"ATTACKER_NODE": attacker_priv})
    except ValueError:
        ledger_rejected = True

    # (b) Attestation forgery: an attacker runs its OWN device (its own
    # AIK keypair, generated fresh in SoftwareTPM.__init__), and tries
    # to have its quote accepted as if it came from the legitimate,
    # enrolled stakeholder node -- even reproducing the same device_id
    # string and the same boot measurements.
    legit_tpm = SoftwareTPM(device_id="NTA_HQ")
    legit_tpm.extend(0, b"firmware-hash-NTA_HQ")
    attacker_tpm = SoftwareTPM(device_id="NTA_HQ")
    attacker_tpm.extend(0, b"firmware-hash-NTA_HQ")

    nonce = os.urandom(16)
    forged_quote = attacker_tpm.quote(nonce)  # signed with the ATTACKER's own AIK private key
    quote_ok, quote_problem = verify_quote(forged_quote, legit_tpm.aik_public, nonce)  # checked against the real node's public key

    passed = ledger_rejected and (quote_ok is False)
    detail = (f"ledger rejected an unenrolled cosigner: {ledger_rejected} | "
              f"forged attestation (same device_id, same PCR values, different AIK) verified "
              f"against the real node's public key: accepted={quote_ok} ({quote_problem})")
    return ScenarioResult(
        "Unauthorized device", "Present an attestation from a device/key never enrolled",
        "Stakeholder verification fails -> release refused", passed, detail,
    )


# ---------------------------------------------------------------------
# Row 3: Tampered platform
# ---------------------------------------------------------------------
def scenario_tampered_platform():
    params = _vdf_params()
    N = params["N"]
    T = 2 ** 10
    anchor = b"ledger_entry:exam_window_B"
    x = derive_vdf_input(anchor, N)
    y = vdf_evaluate(x, T, N)
    proof = vdf_prove(x, y, T, N)  # GENUINELY valid VDF proof -- the window really has elapsed

    tpm, sealed, _secret = _fresh_sealed_node("TAMPERED_NODE")
    tpm.extend(1, b"MALICIOUS-modified-exam-client-binary")  # tamper AFTER sealing

    result = check_release(tpm, sealed, N, x, y, T, proof)

    passed = (result["released"] is False and result["vdf_ok"] is True and result["tpm_ok"] is False)
    detail = (f"VDF proof is genuinely valid (vdf_ok={result['vdf_ok']}) but the platform was modified after "
              f"sealing (tpm_ok={result['tpm_ok']}) -> released={result['released']}. "
              f"This is the AND-gate proof point: a valid time proof alone was NOT enough.")
    return ScenarioResult(
        "Tampered platform", "Modified exam-client binary changes measured PCR/TEE state",
        "TPM/TEE refuses to unseal even with a valid VDF proof -- demonstrates the AND condition", passed, detail,
    )


# ---------------------------------------------------------------------
# Row 4: Key extraction / replay
# ---------------------------------------------------------------------
def scenario_key_extraction_replay():
    # (a) TPM quote replay: an old, genuinely valid quote is replayed
    # against a fresh challenge nonce it was never issued for.
    tpm, _sealed, _secret = _fresh_sealed_node("REPLAY_NODE")
    original_nonce = os.urandom(16)
    old_quote = tpm.quote(original_nonce)
    replay_nonce = os.urandom(16)  # verifier later issues a NEW, different challenge
    replay_ok, replay_problem = verify_quote(old_quote, tpm.aik_public, replay_nonce)

    # (b) VDF proof replay across a DIFFERENT exam window's anchor: a
    # genuinely valid proof for window A's commitment is replayed to
    # try to unlock window B's release.
    params = _vdf_params()
    N = params["N"]
    T = 2 ** 10
    anchor_a = b"ledger_entry:exam_window_A_REPLAY_TEST"
    anchor_b = b"ledger_entry:exam_window_B_REPLAY_TEST"
    x_a = derive_vdf_input(anchor_a, N)
    x_b = derive_vdf_input(anchor_b, N)
    y_a = vdf_evaluate(x_a, T, N)
    proof_a = vdf_prove(x_a, y_a, T, N)
    vdf_replay_ok = vdf_verify(x_b, y_a, T, N, proof_a)  # a valid proof for A, checked against B's input

    passed = (replay_ok is False) and (vdf_replay_ok is False)
    detail = (f"stale TPM quote replayed against a fresh nonce: accepted={replay_ok} ({replay_problem}) | "
              f"VDF proof from exam window A replayed against window B's anchor: accepted={vdf_replay_ok}")
    return ScenarioResult(
        "Key extraction / replay", "Reuse a previously valid attestation or VDF proof",
        "Nonce/freshness and anchor-binding checks reject the replay", passed, detail,
    )


# ---------------------------------------------------------------------
# Row 5: Repeated tamper attempts
# ---------------------------------------------------------------------
def scenario_repeated_tamper_attempts():
    priv, pub = generate_stakeholder_keypair()
    ledger = CommitmentLedger({"CUSTODIAN": pub})

    tpm, sealed, _secret = _fresh_sealed_node("CUSTODIAN")
    fallback_fired = []

    def fallback(node, reason):
        fallback_fired.append((node, reason))

    guard = TamperResponseGuard(
        tpm, sealed, max_failed_attempts=3, fallback_callback=fallback,
        ledger=ledger, stakeholder_id="CUSTODIAN", signer_priv=priv,
    )
    tpm.extend(1, b"MALICIOUS-repeated-tamper")  # tamper once; every subsequent unseal attempt will now fail

    for _ in range(3):
        try:
            guard.attempt_unseal()
        except Exception:
            pass

    destroyed_after_k = guard.destroyed
    fallback_fired_once = len(fallback_fired) == 1

    # Further attempts must be refused WITHOUT even touching the TPM again.
    post_destruction_refused = False
    try:
        guard.attempt_unseal()
    except ShareDestroyedError:
        post_destruction_refused = True

    # Prove destruction is REAL, not a flag: restore a MATCHING boot
    # state, then try to unseal the (destroyed) sealed blob directly
    # against the TPM, bypassing the guard entirely.
    tpm.pcrs = [b"\x00" * 32 for _ in range(8)]
    tpm.extend(0, "firmware-hash-CUSTODIAN".encode())
    tpm.extend(1, "os-kernel-hash-CUSTODIAN".encode())
    genuinely_unrecoverable = False
    try:
        tpm.unseal(guard.sealed_share)
    except ValueError:
        genuinely_unrecoverable = True

    ledger_valid, _problem = ledger.verify_chain()
    tamper_logged = any(e["data"].get("event") == "tamper_response_triggered" for e in ledger.entries)

    passed = all([destroyed_after_k, fallback_fired_once, post_destruction_refused,
                  genuinely_unrecoverable, ledger_valid, tamper_logged])
    detail = (f"destroyed_after_3_failed_attempts={destroyed_after_k} fallback_fired_once={fallback_fired_once} "
              f"post_destruction_refused_without_touching_tpm={post_destruction_refused} "
              f"genuinely_unrecoverable_even_with_boot_state_restored={genuinely_unrecoverable} "
              f"ledger_still_valid={ledger_valid} tamper_event_permanently_logged={tamper_logged}")
    return ScenarioResult(
        "Repeated tamper attempts", "k failed attestations at one node",
        "Local share is actively destroyed; fallback channel triggers automatically", passed, detail,
    )


# ---------------------------------------------------------------------
# Row 6: Social-media leak (detection layer, custody chain untouched)
# ---------------------------------------------------------------------
def scenario_social_media_leak(engine, test_sessions):
    leak_session = next((s for s in test_sessions if s["label"] == "leak"), None)
    if leak_session is None:
        return ScenarioResult(
            "Social-media leak", "Leak-pattern content appears on monitored channels",
            "Ensemble anomaly score exceeds threshold -> alert", False,
            "no leak session available in this held-out test split",
        )

    result = engine.score_session(leak_session)
    passed = result["alert"] is True
    detail = (f"session={leak_session['session_id']} combined_score={result['combined_score']:.3f} "
              f"alert={result['alert']} -- scored purely by the detection layer; the custody chain "
              f"was never invoked for this check, demonstrating the two layers run independently")
    return ScenarioResult(
        "Social-media leak", "Leak-pattern content appears on monitored channels",
        "Ensemble anomaly score exceeds threshold -> alert, independent of custody-chain state", passed, detail,
    )


# ---------------------------------------------------------------------
# Row 7: Photographed page -- isolates the image branch from the text branch
# ---------------------------------------------------------------------
def scenario_photographed_page(engine, seed):
    from src.detection.synthetic_data import generate_leak_session, _make_true_image_hash, NORMAL_CHATTER_TEMPLATES

    rng = random.Random(seed)
    true_image_hash = _make_true_image_hash(rng)
    session = generate_leak_session("photo_only_leak_test", window_start=1_800_000_000.0,
                                     rng=rng, true_image_hash=true_image_hash)

    # Neutralize the TEXT signal specifically: replace every text event's
    # content with ordinary, unrelated exam-prep chatter, so the content-
    # match branch has nothing paper-specific to find. Image events (the
    # photographed-page shares, already generated near the true hash) are
    # left untouched. This isolates exactly the spec row's claim: text
    # monitoring alone would miss this: the image branch is what catches it.
    for e in session["events"]:
        if e["event_type"] == "text":
            e["text"] = NORMAL_CHATTER_TEMPLATES[rng.randrange(len(NORMAL_CHATTER_TEMPLATES))].format(n=rng.randint(1, 20))

    result = engine.score_session(session)
    top_signal = result["attribution_ranked"][0][0]

    passed = (result["scores"]["content"] < 0.3) and (result["scores"]["image"] > 0.7) and (result["alert"] is True)
    detail = (f"content_score={result['scores']['content']:.3f} (text neutralized to generic chatter) | "
              f"image_score={result['scores']['image']:.3f} (photographed-page pHash match) | "
              f"combined_score={result['combined_score']:.3f} alert={result['alert']} | "
              f"top contributing signal={top_signal}")
    return ScenarioResult(
        "Photographed page", "Image of the physical paper circulates",
        "pHash match against the committed image hash -> alert, catching what text monitoring alone would miss",
        passed, detail,
    )


# ---------------------------------------------------------------------
# Row 8: Post-incident dispute
# ---------------------------------------------------------------------
def scenario_post_incident_dispute():
    paper = b"Q1. ... [the real finalized exam paper content -- attack-harness copy] ..."
    result = run_custody_flow(
        paper, paper_id="ATTACK-TEST-A1", window_start_label="2026-09-20T09:00:00Z",
        center_id="CTR-TEST", vdf_bits=10, verbose=False,  # small T -- this test is about the LEDGER, not VDF timing
    )
    ledger = result["ledger"]

    genuine_match = ledger.find_commitment(hash_paper(paper))
    fabricated_paper = b"a completely different, fabricated document falsely claimed to be the leak"
    fabricated_match = ledger.find_commitment(hash_paper(fabricated_paper))

    passed = (genuine_match is not None) and (fabricated_match is None) and result["ledger_valid"]
    detail = (f"genuine leaked copy matches the pre-committed hash: {genuine_match is not None} | "
              f"fabricated document falsely matches: {fabricated_match is not None} (must be False) | "
              f"ledger chain still valid: {result['ledger_valid']}")
    return ScenarioResult(
        "Post-incident dispute", "A leaked document surfaces after an exam",
        "hash(leaked document) checked against the pre-committed ledger value -> verifiable match/non-match",
        passed, detail,
    )


def run_all_scenarios(seed=99, verbose=True):
    results = []
    if verbose:
        print("Running attack-resistance scenarios against the real, already-built modules...\n")

    results.append(scenario_early_access_attempt())
    results.append(scenario_unauthorized_device())
    results.append(scenario_tampered_platform())
    results.append(scenario_key_extraction_replay())
    results.append(scenario_repeated_tamper_attempts())

    engine, test_sessions = build_trained_fusion_engine(seed=seed, verbose=False)
    results.append(scenario_social_media_leak(engine, test_sessions))
    results.append(scenario_photographed_page(engine, seed=seed + 1))

    results.append(scenario_post_incident_dispute())

    if verbose:
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"[{status}] {r.scenario}")
            print(f"       mechanism exercised: {r.mechanism}")
            print(f"       expected:            {r.expected}")
            print(f"       measured:            {r.detail}")
            print()

    return results


if __name__ == "__main__":
    results = run_all_scenarios(verbose=True)

    print("=" * 105)
    print(f"{'Scenario':<26} | {'Mechanism exercised':<50} | Result")
    print("=" * 105)
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"{r.scenario:<26} | {r.mechanism:<50} | {mark}")
    print("=" * 105)

    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} attack-resistance scenarios passed "
          f"(mirrors spec section C.7 -- all 8 rows of the Attack Resistance table).")

    if n_pass != n_total:
        print("FAILED scenarios:", [r.scenario for r in results if not r.passed])
    assert n_pass == n_total, "not all attack-resistance scenarios passed -- see failures above"
    print("All 8 rows of the Attack Resistance table verified against the real, working system.")
