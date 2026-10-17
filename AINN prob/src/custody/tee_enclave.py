"""
Simulated TEE (Trusted Execution Environment) enclave boundary.

Honest scope note up front: Intel TDX / AMD SEV-SNP / ARM CCA give a
HARDWARE-enforced guarantee -- encrypted memory that not even a
compromised OS or hypervisor can read. Python has no equivalent
mechanism; this module cannot reproduce that guarantee. What it CAN
reproduce, and what actually matters for the protocol design, is the
*API boundary*: a reconstructed secret must never be returned to, or
retained by, code outside this enclave -- only the RESULT of an
operation performed on it may leave. That's an architectural property
this module enforces by construction (the secret never appears in a
return value or a public attribute), backed by a best-effort memory
wipe. It is a simulation of the guarantee for prototype/demo purposes,
not a substitute for real confidential-computing hardware in a
production deployment.
"""

from src.crypto.shamir import reconstruct_secret


class TEEEnclave:
    """
    Use as a context manager. Reconstruction and use of a secret may
    only happen inside `with TEEEnclave(...) as enclave: ...`, via
    `run_protected()`, which never returns the secret itself.
    """

    def __init__(self, name: str):
        self.name = name
        self._active = False

    def __enter__(self):
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self._active = False
        return False  # never suppress exceptions

    def run_protected(self, shares: list, operation):
        """
        Reconstructs the secret from `shares` INSIDE this call, passes
        it to `operation(secret_bytes) -> result`, then wipes the local
        copy before returning. Only `result` -- whatever `operation`
        computed FROM the secret -- ever leaves this method.
        """
        if not self._active:
            raise RuntimeError(
                f"[{self.name}] run_protected() called outside an active enclave context "
                "(use 'with TEEEnclave(...) as enclave:')"
            )

        secret = bytearray(reconstruct_secret(shares))
        try:
            result = operation(bytes(secret))
            return result
        finally:
            # Best-effort wipe. Note this does NOT guarantee the plaintext
            # bytes are gone from process memory -- Python's `bytes()` copy
            # made above and any CPython internal buffering are outside our
            # control. Real memory-safety for this comes from the TEE
            # hardware's memory encryption in a production deployment, not
            # from this line.
            for i in range(len(secret)):
                secret[i] = 0


if __name__ == "__main__":
    # Self-test / demo -- ties together Shamir reconstruction + AES-GCM
    # decryption, exactly as step 8 -> 9 of the data flow describes.
    from src.crypto.shamir import split_secret
    from src.crypto.aes_gcm import generate_content_key, encrypt_paper, decrypt_paper

    # Setup: encrypt a paper, split the key into shares (mirrors steps 2 and 4)
    K = generate_content_key()
    paper = b"Q1. ... [the real finalized exam paper] ..."
    sealed_paper = encrypt_paper(paper, K, paper_id="PHY-2026-A1", window_start="2026-09-20T09:00:00Z", center_id="CTR-0042")
    shares = split_secret(K, threshold=3, total_shares=4)

    print("Setup done: paper encrypted, key split into 4 shares (t=3).")

    # At the exam center: reconstruct + decrypt, ALL inside the enclave.
    # The exam center code never sees the raw key K -- only the plaintext
    # `operation()` hands back.
    enclave = TEEEnclave(name="EXAM_CENTER_CTR-0042")
    with enclave:
        plaintext = enclave.run_protected(
            shares=shares[:3],  # any 3 of the 4 -- threshold met
            operation=lambda recovered_key: decrypt_paper(sealed_paper, recovered_key),
        )

    assert plaintext == paper
    print("Reconstruction + decryption inside enclave: OK, plaintext matches original.")
    print("Plaintext (first 40 bytes):", plaintext[:40])

    # Boundary test: run_protected() must refuse to run outside an active context
    enclave2 = TEEEnclave(name="STANDALONE_TEST")
    try:
        enclave2.run_protected(shares[:3], lambda k: k)
        print("FAIL: run_protected() executed outside an active enclave context")
    except RuntimeError as e:
        print("Enclave boundary correctly enforced:", e)

    # API-shape check: nothing on the enclave object ever held the raw key
    leaked = any(isinstance(v, (bytes, bytearray)) for v in vars(enclave).values())
    print("Raw secret retained as an enclave attribute (must be False):", leaked)
