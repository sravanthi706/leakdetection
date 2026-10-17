"""
Active tamper response -- spec section 4.6.

Standard TPM sealing is PASSIVE: it just refuses to unseal on a
mismatch, forever, silently. This module adds an ACTIVE response on
top of the real tpm_sim.py module (no reimplementation of TPM logic
here -- this wraps it):

  - After `k` failed unseal attempts at one node, the locally sealed
    share is genuinely destroyed (its ciphertext bytes are zeroed in
    place, not just flagged), and a fallback callback fires.
  - A detected mid-session PCR/state DRIFT (noticed without even
    attempting an unseal) triggers the same response immediately,
    on the first detection -- drift is a stronger, more specific
    signal than a single failed unseal, so it doesn't wait for a
    count of attempts.
  - The event is logged to the commitment ledger when one is wired in,
    so tampering becomes a permanent, co-signed audit record rather
    than something only the local node knows happened.
"""


class ShareDestroyedError(PermissionError):
    """Raised when an operation is attempted on a share that has already been actively destroyed."""


class TamperResponseGuard:
    def __init__(self, node_tpm, sealed_share: dict, max_failed_attempts: int,
                 fallback_callback=None, ledger=None, stakeholder_id=None, signer_priv=None):
        self.node_tpm = node_tpm
        self.sealed_share = sealed_share  # mutated in place on destruction
        self.max_failed_attempts = max_failed_attempts
        self.fallback_callback = fallback_callback
        self.ledger = ledger
        self.stakeholder_id = stakeholder_id
        self.signer_priv = signer_priv

        self.failed_attempts = 0
        self.destroyed = False
        self.destroy_reason = None
        self.ledger_log_error = None

    def attempt_unseal(self) -> bytes:
        if self.destroyed:
            raise ShareDestroyedError(
                f"[{self.stakeholder_id}] Share was already actively destroyed "
                f"({self.destroy_reason}) -- use the fallback channel, not this node."
            )

        try:
            share = self.node_tpm.unseal(self.sealed_share)
            self.failed_attempts = 0  # a clean unseal resets the counter
            return share
        except ValueError as e:
            self.failed_attempts += 1
            if self.failed_attempts >= self.max_failed_attempts:
                self._destroy_and_fallback(
                    reason=f"{self.failed_attempts} failed unseal attempts (limit {self.max_failed_attempts}): {e}"
                )
            raise

    def check_drift(self) -> bool:
        """
        Proactive check, independent of anyone attempting an unseal.
        Returns True if drift was detected (and triggers destruction
        immediately, on this first detection -- no attempt counting).
        """
        if self.destroyed:
            return True  # already handled

        current_ok = self.node_tpm.matches_policy(
            self.sealed_share["pcr_indices"], self.sealed_share["policy_digest"]
        )
        if not current_ok:
            self._destroy_and_fallback(reason="mid-session PCR/state drift detected (proactive check, no unseal attempted)")
            return True
        return False

    def _destroy_and_fallback(self, reason: str):
        # Active destruction: actually zero the ciphertext/tag bytes in place,
        # not just set a flag -- so even if an attacker later restores a
        # matching boot state, there is nothing left to decrypt.
        if isinstance(self.sealed_share.get("ciphertext"), (bytes, bytearray)):
            self.sealed_share["ciphertext"] = b"\x00" * len(self.sealed_share["ciphertext"])
        if isinstance(self.sealed_share.get("tag"), (bytes, bytearray)):
            self.sealed_share["tag"] = b"\x00" * len(self.sealed_share["tag"])

        self.destroyed = True
        self.destroy_reason = reason

        # Destruction and fallback notification are the security-critical
        # actions and must happen regardless of what follows. Ledger logging
        # is important for the audit trail but its failure must never be
        # allowed to silently swallow the fallback notification -- so it's
        # explicitly best-effort here, not a precondition for the rest.
        self.ledger_log_error = None
        if self.ledger is not None and self.stakeholder_id is not None and self.signer_priv is not None:
            try:
                self.ledger.append(
                    data={
                        "event": "tamper_response_triggered",
                        "node": self.stakeholder_id,
                        "reason": reason,
                        "failed_attempts": self.failed_attempts,
                    },
                    cosigners={self.stakeholder_id: self.signer_priv},
                )
            except Exception as e:
                self.ledger_log_error = str(e)

        if self.fallback_callback is not None:
            self.fallback_callback(node=self.stakeholder_id, reason=reason)


if __name__ == "__main__":
    import os
    from src.custody.tpm_sim import SoftwareTPM
    from src.custody.commitment_ledger import CommitmentLedger, generate_stakeholder_keypair

    stakeholder_ids = ["STATE_AUTHORITY", "INDEPENDENT_AUDIT", "CUSTODIAN"]
    keys = {sid: generate_stakeholder_keypair() for sid in stakeholder_ids}  # each stakeholder gets ITS OWN keypair
    ledger = CommitmentLedger({sid: pub for sid, (priv, pub) in keys.items()})

    fallback_events = []

    def fallback(node, reason):
        fallback_events.append((node, reason))
        print(f"  >> FALLBACK CHANNEL TRIGGERED for {node}: {reason}")

    # --- Scenario 1: normal repeated use, no tampering -- never trips ---
    tpm1 = SoftwareTPM(device_id="STATE_AUTHORITY")
    tpm1.extend(0, b"firmware-hash")
    tpm1.extend(1, b"os-hash")
    share1 = os.urandom(48)
    sealed1 = tpm1.seal(share1, pcr_indices=[0, 1])
    guard1 = TamperResponseGuard(tpm1, sealed1, max_failed_attempts=3, fallback_callback=fallback,
                                  ledger=ledger, stakeholder_id="STATE_AUTHORITY", signer_priv=keys["STATE_AUTHORITY"][0])

    for _ in range(5):
        recovered = guard1.attempt_unseal()
        assert recovered == share1
    print("Scenario 1: 5 clean unseals, guard never tripped. destroyed =", guard1.destroyed)
    assert guard1.destroyed is False

    # --- Scenario 2: k failed unseal attempts -> destruction on the k-th ---
    tpm2 = SoftwareTPM(device_id="INDEPENDENT_AUDIT")
    tpm2.extend(0, b"firmware-hash")
    tpm2.extend(1, b"os-hash")
    share2 = os.urandom(48)
    sealed2 = tpm2.seal(share2, pcr_indices=[0, 1])
    guard2 = TamperResponseGuard(tpm2, sealed2, max_failed_attempts=3, fallback_callback=fallback,
                                  ledger=ledger, stakeholder_id="INDEPENDENT_AUDIT", signer_priv=keys["INDEPENDENT_AUDIT"][0])

    tpm2.extend(1, b"MALICIOUS-modified-os")  # tamper once, before any attempts

    for i in range(1, 4):
        try:
            guard2.attempt_unseal()
        except Exception as e:
            print(f"Scenario 2, attempt {i}: failed as expected ({type(e).__name__}). destroyed =", guard2.destroyed)

    assert guard2.destroyed is True
    assert len(fallback_events) == 1 and fallback_events[0][0] == "INDEPENDENT_AUDIT"
    print("Scenario 2: destroyed after 3 failed attempts, fallback fired exactly once.")

    # Further attempts must be refused WITHOUT even touching the TPM again
    try:
        guard2.attempt_unseal()
        print("FAIL: attempt succeeded after destruction")
    except ShareDestroyedError as e:
        print("Scenario 2: post-destruction attempt correctly refused:", e)

    # Prove destruction is REAL, not just a flag: restore matching boot state,
    # then try to unseal the (destroyed) sealed blob directly against the TPM,
    # bypassing the guard entirely.
    tpm2.pcrs = [b"\x00" * 32 for _ in range(8)]
    tpm2.extend(0, b"firmware-hash")
    tpm2.extend(1, b"os-hash")  # boot state "restored" to the original good measurement
    try:
        tpm2.unseal(guard2.sealed_share)
        print("FAIL: destroyed share was still recoverable after restoring boot state")
    except ValueError as e:
        print("Scenario 2: destroyed share correctly UNRECOVERABLE even with matching boot state restored:", e)

    # --- Scenario 3: proactive drift detection, no unseal attempt needed ---
    tpm3 = SoftwareTPM(device_id="CUSTODIAN")
    tpm3.extend(0, b"firmware-hash")
    tpm3.extend(1, b"os-hash")
    share3 = os.urandom(48)
    sealed3 = tpm3.seal(share3, pcr_indices=[0, 1])
    guard3 = TamperResponseGuard(tpm3, sealed3, max_failed_attempts=3, fallback_callback=fallback,
                                  ledger=ledger, stakeholder_id="CUSTODIAN", signer_priv=keys["CUSTODIAN"][0])

    assert guard3.check_drift() is False
    print("Scenario 3: no drift on clean state.")

    tpm3.extend(1, b"MALICIOUS-mid-session-change")
    drift_detected = guard3.check_drift()
    print("Scenario 3: drift detected on first check:", drift_detected, "| destroyed =", guard3.destroyed)
    assert drift_detected is True and guard3.destroyed is True
    assert len(fallback_events) == 2

    # --- Ledger check: both tamper events are permanently, verifiably recorded ---
    valid, problem = ledger.verify_chain()
    tamper_events = [e for e in ledger.entries if e["data"].get("event") == "tamper_response_triggered"]
    print(f"Ledger chain valid: {valid}. Tamper events recorded: {len(tamper_events)} (expected 2).")
    assert valid and len(tamper_events) == 2

    print("All tamper-response scenarios verified.")
