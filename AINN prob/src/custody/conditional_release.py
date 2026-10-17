"""
Conditional release -- the AND gate from spec section 4.4.

A stakeholder node releases its share ONLY when BOTH hold at once:
  1. The VDF proof verifies (the exam window has genuinely elapsed).
  2. The node's own TPM/TEE attestation succeeds (unseal works, meaning
     the boot/platform state matches what the share was sealed to).

This module doesn't reimplement either check -- it wires together the
real vdf.py and tpm_sim.py modules already built, and its only job is
the gating logic and surfacing WHICH condition(s) failed, since that
matters for the attack-resistance demo: "tampered platform" and "early
access" are different failure reasons that should be visibly different,
not just a single opaque "denied".
"""

from src.custody.vdf import verify as vdf_verify


def check_release(node_tpm, sealed_share: dict, N: int, vdf_x: int, vdf_y: int, vdf_T: int, vdf_proof: list) -> dict:
    """
    Evaluates BOTH conditions independently (does not short-circuit),
    so a demo can show cases where both fail at once, not just one.
    """
    vdf_ok = False
    vdf_reason = None
    try:
        vdf_ok = vdf_verify(vdf_x, vdf_y, vdf_T, N, vdf_proof)
        if not vdf_ok:
            vdf_reason = "VDF proof invalid, or does not correspond to a genuinely elapsed exam window"
    except Exception as e:
        vdf_reason = f"VDF verification raised an error: {e}"

    tpm_ok = False
    tpm_reason = None
    share = None
    try:
        share = node_tpm.unseal(sealed_share)
        tpm_ok = True
    except ValueError as e:
        tpm_reason = str(e)

    released = vdf_ok and tpm_ok
    return {
        "released": released,
        "share": share if released else None,
        "vdf_ok": vdf_ok,
        "vdf_reason": vdf_reason,
        "tpm_ok": tpm_ok,
        "tpm_reason": tpm_reason,
    }


def release_share(node_tpm, sealed_share: dict, N: int, vdf_x: int, vdf_y: int, vdf_T: int, vdf_proof: list) -> bytes:
    """Raises PermissionError naming every failing condition; returns the share only if both hold."""
    result = check_release(node_tpm, sealed_share, N, vdf_x, vdf_y, vdf_T, vdf_proof)
    if not result["released"]:
        reasons = []
        if not result["vdf_ok"]:
            reasons.append(f"VDF check failed ({result['vdf_reason']})")
        if not result["tpm_ok"]:
            reasons.append(f"TPM/TEE attestation failed ({result['tpm_reason']})")
        raise PermissionError("Release refused: " + "; ".join(reasons))
    return result["share"]


if __name__ == "__main__":
    import os
    from src.custody.vdf import generate_vdf_parameters, derive_vdf_input, evaluate, prove
    from src.custody.tpm_sim import SoftwareTPM

    params = generate_vdf_parameters(1024)  # smaller modulus for a fast self-test
    N = params["N"]
    T = 2 ** 10  # small, just needs to be genuinely computed and verified, not realistic timing here

    anchor = b"ledger_entry_hash_for_this_test"
    x = derive_vdf_input(anchor, N)
    y_correct = evaluate(x, T, N)
    proof_correct = prove(x, y_correct, T, N)

    def fresh_node_with_sealed_share():
        tpm = SoftwareTPM(device_id="TEST_NODE")
        tpm.extend(0, b"firmware-hash")
        tpm.extend(1, b"os-hash")
        share = os.urandom(48)
        sealed = tpm.seal(share, pcr_indices=[0, 1])
        return tpm, sealed, share

    # --- Case 1: both conditions hold -> release succeeds ---
    tpm1, sealed1, real_share1 = fresh_node_with_sealed_share()
    released = release_share(tpm1, sealed1, N, x, y_correct, T, proof_correct)
    assert released == real_share1
    print("Case 1 (VDF ok, TPM ok): release succeeded, share matches. OK")

    # --- Case 2: VDF fails (early access attempt), TPM is fine ---
    tpm2, sealed2, _ = fresh_node_with_sealed_share()
    early_y = evaluate(x, T // 2, N)  # someone claims a SHORTER elapsed computation
    early_proof = prove(x, early_y, T // 2, N)
    result2 = check_release(tpm2, sealed2, N, x, early_y, T, early_proof)  # claims T but proof is for T//2
    print("Case 2 (early access attempt):", "released=" + str(result2["released"]),
          "| vdf_ok=" + str(result2["vdf_ok"]), "| tpm_ok=" + str(result2["tpm_ok"]))
    assert result2["released"] is False and result2["tpm_ok"] is True and result2["vdf_ok"] is False

    # --- Case 3: TPM fails (tampered platform), VDF is genuinely valid ---
    tpm3, sealed3, _ = fresh_node_with_sealed_share()
    tpm3.extend(1, b"MALICIOUS-modified-os")  # tamper AFTER sealing
    result3 = check_release(tpm3, sealed3, N, x, y_correct, T, proof_correct)
    print("Case 3 (tampered platform):", "released=" + str(result3["released"]),
          "| vdf_ok=" + str(result3["vdf_ok"]), "| tpm_ok=" + str(result3["tpm_ok"]))
    assert result3["released"] is False and result3["vdf_ok"] is True and result3["tpm_ok"] is False
    print("  -> This is the AND-condition proof point: a VALID VDF proof alone was not enough.")

    # --- Case 4: both fail ---
    tpm4, sealed4, _ = fresh_node_with_sealed_share()
    tpm4.extend(1, b"MALICIOUS-modified-os")
    result4 = check_release(tpm4, sealed4, N, x, early_y, T, early_proof)
    print("Case 4 (both fail):", "released=" + str(result4["released"]),
          "| vdf_ok=" + str(result4["vdf_ok"]), "| tpm_ok=" + str(result4["tpm_ok"]))
    assert result4["released"] is False and result4["vdf_ok"] is False and result4["tpm_ok"] is False

    # --- release_share() raises with BOTH reasons named when both fail ---
    try:
        release_share(tpm4, sealed4, N, x, early_y, T, early_proof)
        print("FAIL: release_share() did not raise for case 4")
    except PermissionError as e:
        print("release_share() correctly raised:", e)

    print("All 4 AND-gate truth-table cases verified.")
