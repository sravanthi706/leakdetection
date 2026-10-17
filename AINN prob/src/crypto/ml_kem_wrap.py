"""
Post-quantum key wrapping for the AES content key K, using ML-KEM
(FIPS 203) instead of RSA/ECC.

Uses the pure-Python `kyber-py` implementation (ML-KEM-768, NIST
security category 3) -- this is a reference/demonstration-grade
implementation, not a hardened production one, but it implements the
real FIPS 203 algorithm rather than a simplified stand-in.

Flow:
  1. Each stakeholder node generates its own ML-KEM keypair (ek, dk)
     once, ahead of time -- ek (public) can be shared, dk (private)
     never leaves the node.
  2. To wrap K for a specific node: ML-KEM encapsulation against that
     node's ek produces a fresh 32-byte shared secret + a kem
     ciphertext. The shared secret is used as an AES-256-GCM key to
     encrypt K itself.
  3. To unwrap: the node uses its own dk to decapsulate the kem
     ciphertext, recovering the same shared secret, then AES-GCM
     decrypts to recover K.

This means K is never sent in a form that classical RSA/ECC "harvest
now, decrypt later" attacks could exploit -- the security of the
wrapping rests on a lattice problem believed hard even for quantum
computers.
"""

import os
from kyber_py.ml_kem import ML_KEM_768
from Crypto.Cipher import AES

NONCE_BYTES = 12


def generate_node_keypair():
    """One stakeholder node's long-term ML-KEM keypair. ek is public, dk is secret."""
    ek, dk = ML_KEM_768.keygen()
    return ek, dk


def wrap_content_key(content_key: bytes, node_ek: bytes) -> dict:
    """
    Wrap K for a specific stakeholder node, using only that node's
    public encapsulation key. Only the matching dk can unwrap it.
    """
    shared_secret, kem_ciphertext = ML_KEM_768.encaps(node_ek)

    nonce = os.urandom(NONCE_BYTES)
    cipher = AES.new(shared_secret, AES.MODE_GCM, nonce=nonce)
    wrapped_key, tag = cipher.encrypt_and_digest(content_key)

    return {
        "kem_ciphertext": kem_ciphertext,   # goes to the node; safe to store/transmit
        "wrapped_key": wrapped_key,          # K, encrypted under the shared secret
        "nonce": nonce,
        "tag": tag,
    }


def unwrap_content_key(wrapped: dict, node_dk: bytes) -> bytes:
    """Recover K at the stakeholder node, using its private dk."""
    shared_secret = ML_KEM_768.decaps(node_dk, wrapped["kem_ciphertext"])

    cipher = AES.new(shared_secret, AES.MODE_GCM, nonce=wrapped["nonce"])
    try:
        content_key = cipher.decrypt_and_verify(wrapped["wrapped_key"], wrapped["tag"])
    except ValueError as e:
        raise ValueError("Key unwrap failed: wrong dk, or wrapped_key was tampered with") from e

    return content_key


if __name__ == "__main__":
    # Self-test / demo
    ek, dk = generate_node_keypair()
    print("Node keypair generated. ek bytes:", len(ek), "dk bytes:", len(dk))

    K = os.urandom(32)  # stand-in for the real AES content key from aes_gcm.py
    wrapped = wrap_content_key(K, ek)
    print("Wrapped OK. kem_ciphertext bytes:", len(wrapped["kem_ciphertext"]))

    recovered = unwrap_content_key(wrapped, dk)
    assert recovered == K
    print("Unwrapped OK. Recovered key matches original K.")

    # Wrong dk test: a different node's dk must NOT be able to unwrap
    _, wrong_dk = generate_node_keypair()
    try:
        unwrap_content_key(wrapped, wrong_dk)
        print("FAIL: wrong node's dk was able to unwrap")
    except Exception as e:
        print("Wrong-dk correctly rejected:", type(e).__name__)

    # Tamper test: corrupt the wrapped_key ciphertext
    tampered = dict(wrapped)
    tampered["wrapped_key"] = bytes([wrapped["wrapped_key"][0] ^ 0x01]) + wrapped["wrapped_key"][1:]
    try:
        unwrap_content_key(tampered, dk)
        print("FAIL: tampered wrapped_key was not detected")
    except ValueError as e:
        print("Tamper correctly detected:", e)
