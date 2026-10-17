"""
Verifiable Delay Function -- Pietrzak's construction (2018), over an
RSA modulus group.

WHAT THIS PROVES: "at least T sequential squarings genuinely happened
between committing to x and producing y" -- and this can be checked in
roughly log2(T) steps instead of redoing all T squarings. That
asymmetry (slow to compute, fast to verify, and NOT parallelizable no
matter how much hardware you throw at it) is what removes the need to
trust any single stakeholder node's reported clock: the VDF proof
itself is the evidence that time elapsed, not an operator's word for it.

HONEST SCOPE NOTE (documented here because it matters for how this
gets defended, not just how it runs): this needs a group of UNKNOWN
ORDER to be secure -- nobody should know phi(N) = (p-1)(q-1), because
knowing it lets you skip the sequential squaring entirely via fast
exponentiation. A real deployment gets this either from a multi-party
trusted-setup ceremony for the RSA modulus (so no single party ever
knows the factorization), or from class groups of imaginary quadratic
order (what Chia's `chiavdf` uses), which need no trusted setup at
all. Neither is something a solo prototype can stand up. Here, this
module generates its own RSA modulus locally and never uses the
factorization again after generating N -- which demonstrates the real
protocol and the real verification-speed asymmetry correctly, but does
NOT reproduce the "nobody knows the order" security property a
production system needs. That gap is explicit, not hidden.
"""

import hashlib
import time

from Crypto.PublicKey import RSA


def generate_vdf_parameters(modulus_bits: int = 2048) -> dict:
    """
    Generates the public RSA modulus N. The private factorization is
    used only transiently inside this function and is never returned,
    stored, or used anywhere else in this module -- evaluate/prove/verify
    below operate on N alone, exactly as a real deployment's stakeholder
    nodes would (they'd only ever see N from the trusted-setup ceremony,
    never a factorization).
    """
    key = RSA.generate(modulus_bits)
    return {"N": key.n, "modulus_bits": modulus_bits}


def _sequential_square(base: int, num_squarings: int, N: int) -> int:
    result = base % N
    for _ in range(num_squarings):
        result = pow(result, 2, N)
    return result


def _fiat_shamir_challenge(x: int, y: int, mu: int, N: int, challenge_bits: int = 128) -> int:
    """Turns the interactive Pietrzak protocol into a non-interactive proof (Fiat-Shamir heuristic)."""
    data = f"{x}|{y}|{mu}|{N}".encode()
    digest = hashlib.sha256(data).digest()
    r = int.from_bytes(digest, "big") % (1 << challenge_bits)
    return r if r != 0 else 1


def derive_vdf_input(anchor_bytes: bytes, N: int) -> int:
    """
    Turns an arbitrary anchor (in this system: the commitment ledger
    entry's hash, per the spec's "each node's VDF computation is
    anchored to the ledger's timestamped commitment") into a valid
    group element x in [2, N-1] to run the VDF on.
    """
    x = int.from_bytes(hashlib.sha256(anchor_bytes).digest(), "big") % N
    return x if x >= 2 else x + 2


def evaluate(x: int, T: int, N: int) -> int:
    """y = x^(2^T) mod N, via T genuinely sequential squarings. This is the slow, unparallelizable step."""
    if T <= 0 or (T & (T - 1)) != 0:
        raise ValueError("T must be a positive power of two")
    return _sequential_square(x, T, N)


def prove(x: int, y: int, T: int, N: int) -> list:
    """
    Builds a proof of log2(T) intermediate values, via recursive
    halving. Prover cost is bounded by roughly one extra full pass of
    squarings (still linear in T, not exponential) -- there are known
    optimizations that reuse the evaluate() trace to reduce this
    further, not applied here for clarity; they don't change what's
    being proved, only how cheaply the prover computes it.
    """
    if T <= 0 or (T & (T - 1)) != 0:
        raise ValueError("T must be a positive power of two")

    proof = []
    cur_x, cur_y, cur_T = x, y, T
    while cur_T > 1:
        half = cur_T // 2
        mu = _sequential_square(cur_x, half, N)
        proof.append(mu)
        r = _fiat_shamir_challenge(cur_x, cur_y, mu, N)
        cur_x = (pow(cur_x, r, N) * mu) % N
        cur_y = (pow(mu, r, N) * cur_y) % N
        cur_T = half
    return proof


def verify(x: int, y: int, T: int, N: int, proof: list) -> bool:
    """
    Checks the proof in log2(T) rounds -- each round is one small
    modular exponentiation (by a ~128-bit challenge), not a T-length
    squaring chain. This is the fast side of the asymmetry.
    """
    if T <= 0 or (T & (T - 1)) != 0:
        return False
    if len(proof) != T.bit_length() - 1:
        return False  # wrong number of proof elements for this T

    cur_x, cur_y, cur_T = x, y, T
    for mu in proof:
        half = cur_T // 2
        r = _fiat_shamir_challenge(cur_x, cur_y, mu, N)
        cur_x = (pow(cur_x, r, N) * mu) % N
        cur_y = (pow(mu, r, N) * cur_y) % N
        cur_T = half

    return cur_T == 1 and cur_y == pow(cur_x, 2, N)


if __name__ == "__main__":
    print("Generating VDF public parameters (RSA-2048 modulus, factorization discarded after this call)...")
    params = generate_vdf_parameters(2048)
    N = params["N"]

    # T = 2^17 -- chosen from a real benchmark on this machine (~16 microseconds/squaring
    # at this modulus size), so evaluate() takes a few real seconds, not an instant no-op.
    T = 2 ** 17

    anchor = b"ledger_entry_hash:f1bae7aa39b319e2...:window_start=2026-09-20T09:00:00Z"
    x = derive_vdf_input(anchor, N)
    print(f"VDF input x derived from commitment-ledger anchor. T = {T} sequential squarings.")

    t0 = time.time()
    y = evaluate(x, T, N)
    eval_time = time.time() - t0
    print(f"evaluate(): {round(eval_time, 2)}s  <- this is the genuinely sequential, unparallelizable step")

    t0 = time.time()
    proof = prove(x, y, T, N)
    prove_time = time.time() - t0
    print(f"prove():    {round(prove_time, 2)}s  ({len(proof)} intermediate values)")

    t0 = time.time()
    ok = verify(x, y, T, N, proof)
    verify_time = time.time() - t0
    print(f"verify():   {round(verify_time, 4)}s  <- {round(eval_time / verify_time, 0)}x faster than evaluate()")
    print("Proof valid:", ok)
    assert ok

    # --- Negative tests: verification must reject anything that isn't genuine ---

    # 1. Claiming a result without having done the work (guess a y)
    fake_y = (y + 1) % N
    assert verify(x, fake_y, T, N, proof) is False
    print("Forged y correctly rejected.")

    # 2. Reusing a genuinely valid proof for a DIFFERENT claimed output
    other_y = evaluate(x, T // 2, N)  # a real but wrong-length computation
    assert verify(x, other_y, T, N, proof) is False
    print("Proof/output mismatch correctly rejected.")

    # 3. Claiming an elapsed time (T) shorter than what actually happened -- i.e.
    #    trying to skip ahead: verify a smaller T than the proof was built for
    assert verify(x, y, T // 2, N, proof) is False
    print("Under-length T (early-access attempt) correctly rejected.")

    # 4. Tampered proof element
    tampered_proof = list(proof)
    tampered_proof[0] = (tampered_proof[0] + 1) % N
    assert verify(x, y, T, N, tampered_proof) is False
    print("Tampered proof element correctly rejected.")
