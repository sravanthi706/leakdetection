"""
AES-256-GCM content encryption for the exam paper.

Binds ciphertext to metadata (paper_id, window_start, center_id) as
Associated Authenticated Data (AAD) -- tampering with either the
ciphertext OR the metadata is detected on decrypt.
"""

import json
import os
from Crypto.Cipher import AES

KEY_BYTES = 32   # AES-256
NONCE_BYTES = 12 # GCM standard nonce size
TAG_BYTES = 16


def generate_content_key() -> bytes:
    """Fresh random 256-bit AES key. This is K in the spec."""
    return os.urandom(KEY_BYTES)


def _aad_bytes(paper_id: str, window_start: str, center_id: str) -> bytes:
    # Deterministic encoding so encrypt/decrypt always agree on AAD bytes.
    return json.dumps(
        {"paper_id": paper_id, "window_start": window_start, "center_id": center_id},
        sort_keys=True,
    ).encode("utf-8")


def encrypt_paper(plaintext: bytes, key: bytes, paper_id: str, window_start: str, center_id: str) -> dict:
    """
    Encrypt paper content. Returns a dict with everything needed to decrypt
    later EXCEPT the key itself (the key is never stored alongside the
    ciphertext -- it goes through wrap/split in later modules).
    """
    if len(key) != KEY_BYTES:
        raise ValueError(f"key must be {KEY_BYTES} bytes, got {len(key)}")

    nonce = os.urandom(NONCE_BYTES)
    aad = _aad_bytes(paper_id, window_start, center_id)

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(aad)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)

    return {
        "ciphertext": ciphertext,
        "nonce": nonce,
        "tag": tag,
        "paper_id": paper_id,
        "window_start": window_start,
        "center_id": center_id,
    }


def decrypt_paper(sealed: dict, key: bytes) -> bytes:
    """
    Decrypt and verify. Raises ValueError if the ciphertext OR any piece
    of the bound metadata was tampered with -- this is the "tamper with
    the ciphertext, not the key" property described in the spec.
    """
    aad = _aad_bytes(sealed["paper_id"], sealed["window_start"], sealed["center_id"])

    cipher = AES.new(key, AES.MODE_GCM, nonce=sealed["nonce"])
    cipher.update(aad)
    try:
        plaintext = cipher.decrypt_and_verify(sealed["ciphertext"], sealed["tag"])
    except ValueError as e:
        raise ValueError("Integrity check failed: ciphertext or metadata was tampered with") from e

    return plaintext


if __name__ == "__main__":
    # Self-test / demo
    K = generate_content_key()
    paper = b"Q1. What is the time complexity of ... [full exam paper content]"

    sealed = encrypt_paper(paper, K, paper_id="PHY-2026-A1", window_start="2026-09-20T09:00:00Z", center_id="CTR-0042")
    print("Encrypted OK. Ciphertext bytes:", len(sealed["ciphertext"]))

    recovered = decrypt_paper(sealed, K)
    assert recovered == paper
    print("Decrypted OK. Plaintext matches original.")

    # Tamper test: flip one byte of ciphertext, decryption must fail
    tampered = dict(sealed)
    tampered["ciphertext"] = bytes([sealed["ciphertext"][0] ^ 0x01]) + sealed["ciphertext"][1:]
    try:
        decrypt_paper(tampered, K)
        print("FAIL: tampered ciphertext was not detected")
    except ValueError as e:
        print("Tamper correctly detected:", e)

    # Tamper test: change bound metadata (center_id) without touching ciphertext
    tampered2 = dict(sealed)
    tampered2["center_id"] = "CTR-0099"
    try:
        decrypt_paper(tampered2, K)
        print("FAIL: tampered metadata was not detected")
    except ValueError as e:
        print("Metadata tamper correctly detected:", e)
