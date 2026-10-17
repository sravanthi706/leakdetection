"""
Simulated TPM 2.0: PCR measured-boot state, sealing, and quoting.

This is a SOFTWARE MODEL of TPM semantics, not real hardware -- it
implements the same interface (extend / seal / unseal / quote) a real
TPM 2.0 chip exposes via tpm2-tools, so the rest of this codebase (and
a later swap to `tpm2-pytss` against real hardware, per the roadmap)
doesn't need to change shape, only its backing implementation.

Key concept -- PCR extend is one-way and cumulative:
    PCR_new = SHA256(PCR_old || measurement)
This is exactly how a real boot chain works: each stage (firmware,
bootloader, OS, application) measures the NEXT stage before handing
off control, extending the PCR. You cannot "unextend" or set a PCR to
an arbitrary value directly -- the only way to reach a given PCR value
is to have actually run the exact same sequence of measurements. This
is what "boot integrity attestation" rests on.

Sealing binds a secret to a *policy*: a specific set of expected PCR
values. Unsealing only succeeds if the TPM's CURRENT PCR state matches
that policy -- i.e., nothing changed since the secret was sealed.
"""

import hashlib
import os

from Crypto.Cipher import AES
from Crypto.PublicKey import ECC
from Crypto.Signature import eddsa

NUM_PCRS = 8
PCR_SIZE = 32  # SHA-256 output


class SoftwareTPM:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.pcrs = [b"\x00" * PCR_SIZE for _ in range(NUM_PCRS)]
        # Attestation Identity Key -- signs quotes, proving they came from THIS device.
        self._aik_private = ECC.generate(curve='ed25519')
        self.aik_public = self._aik_private.public_key()

    def extend(self, pcr_index: int, measurement: bytes):
        """Called once per boot stage (firmware measures bootloader, bootloader measures OS, ...)."""
        if not (0 <= pcr_index < NUM_PCRS):
            raise ValueError(f"pcr_index must be 0..{NUM_PCRS - 1}")
        self.pcrs[pcr_index] = hashlib.sha256(self.pcrs[pcr_index] + measurement).digest()

    def _policy_digest(self, pcr_indices: list) -> bytes:
        """Digest of the PCR values a seal policy cares about, in a fixed order."""
        h = hashlib.sha256()
        for i in pcr_indices:
            h.update(self.pcrs[i])
        return h.digest()

    def seal(self, data: bytes, pcr_indices: list) -> dict:
        """
        Seal `data` to the CURRENT values of the given PCRs. Only a TPM
        (real or this simulated one) whose PCRs later match this exact
        policy digest can unseal it.
        """
        policy_digest = self._policy_digest(pcr_indices)
        seal_key = hashlib.sha256(b"seal_key|" + policy_digest).digest()  # KDF stand-in

        nonce = os.urandom(12)
        cipher = AES.new(seal_key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(data)

        return {
            "pcr_indices": list(pcr_indices),
            "policy_digest": policy_digest,   # not secret -- knowing it doesn't help without matching PCRs
            "ciphertext": ciphertext,
            "nonce": nonce,
            "tag": tag,
        }

    def unseal(self, sealed: dict) -> bytes:
        """Succeeds only if this TPM's CURRENT PCR state matches the policy recorded at seal time."""
        current_digest = self._policy_digest(sealed["pcr_indices"])
        if current_digest != sealed["policy_digest"]:
            raise ValueError(
                "Unseal refused: current PCR state does not match the sealing policy "
                "(boot chain has changed since the secret was sealed -- possible tampering)"
            )

        seal_key = hashlib.sha256(b"seal_key|" + sealed["policy_digest"]).digest()
        cipher = AES.new(seal_key, AES.MODE_GCM, nonce=sealed["nonce"])
        return cipher.decrypt_and_verify(sealed["ciphertext"], sealed["tag"])

    def matches_policy(self, pcr_indices: list, policy_digest: bytes) -> bool:
        """
        Read-only check: does the CURRENT PCR state match a given
        policy digest, without attempting to unseal anything? Used for
        drift detection -- noticing tampering has happened even before
        anyone tries (and fails) an unseal.
        """
        return self._policy_digest(pcr_indices) == policy_digest

    def quote(self, nonce: bytes) -> dict:
        """
        A signed statement: "as of right now, my PCRs are X, and I was
        asked to prove this with YOUR nonce" -- the nonce is what stops
        a replay of an old quote (freshness).
        """
        message = self.device_id.encode() + nonce + b"".join(self.pcrs)
        signature = eddsa.new(self._aik_private, 'rfc8032').sign(message)
        return {
            "device_id": self.device_id,
            "pcrs": list(self.pcrs),
            "nonce": nonce,
            "signature": signature,
        }


def verify_quote(quote: dict, aik_public_key, expected_nonce: bytes) -> tuple:
    """Returns (is_valid, problem)."""
    if quote["nonce"] != expected_nonce:
        return False, "nonce mismatch -- this quote is stale or a replay"

    message = quote["device_id"].encode() + quote["nonce"] + b"".join(quote["pcrs"])
    try:
        eddsa.new(aik_public_key, 'rfc8032').verify(message, quote["signature"])
    except ValueError:
        return False, "signature invalid -- quote did not come from the claimed device"

    return True, None


if __name__ == "__main__":
    tpm = SoftwareTPM(device_id="STATE_AUTHORITY_NODE_01")

    # Simulate a normal, legitimate boot: firmware -> bootloader -> OS -> app
    tpm.extend(0, b"firmware-v3.2-hash")
    tpm.extend(1, b"bootloader-v1.0-hash")
    tpm.extend(2, b"os-kernel-5.15-hash")
    tpm.extend(3, b"exam-client-app-v1.0-hash")
    print("Boot complete. PCR[0..3] set.")

    # Seal a Shamir share to this exact boot state
    share = os.urandom(48)
    sealed = tpm.seal(share, pcr_indices=[0, 1, 2, 3])
    print("Share sealed to PCR[0..3].")

    recovered = tpm.unseal(sealed)
    assert recovered == share
    print("Unseal on unchanged boot state: OK, share recovered correctly.")

    # Quote: prove current state, bound to a fresh nonce from the verifier
    challenge_nonce = os.urandom(16)
    q = tpm.quote(challenge_nonce)
    ok, problem = verify_quote(q, tpm.aik_public, challenge_nonce)
    print("Quote verification (fresh nonce):", ok, problem)

    # Replay test: reuse an old quote against a NEW expected nonce
    stale_nonce_check = verify_quote(q, tpm.aik_public, os.urandom(16))
    print("Quote verification against a different nonce (must fail):", stale_nonce_check)

    # --- Tamper scenario: modified exam-client binary after sealing ---
    tpm.extend(3, b"MALICIOUS-modified-app-hash")
    try:
        tpm.unseal(sealed)
        print("FAIL: unseal succeeded after boot-state tamper")
    except ValueError as e:
        print("Tamper correctly detected on unseal:", e)

    # A quote taken now also reflects the changed PCR[3] -- and would not match
    # any attestation policy that expected the original app hash.
    q2 = tpm.quote(os.urandom(16))
    print("PCR[3] after tamper differs from original:", q2["pcrs"][3] != q["pcrs"][3])
