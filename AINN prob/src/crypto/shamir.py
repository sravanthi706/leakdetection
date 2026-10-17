"""
Shamir's Secret Sharing (t-of-n), byte-oriented over GF(2^8).

Splits any byte string (in this system: the wrapped content key) into n
shares such that any t of them reconstruct it exactly, but any t-1 or
fewer reveal *zero* information about it (information-theoretic, not
just computationally hard -- this is the actual guarantee threshold
custody in the spec relies on).

This is the same field arithmetic real tools use (e.g. the `ssss`
utility, and conceptually HashiCorp Vault's unseal-key scheme): work
one byte at a time, treat each byte as a constant term of a random
polynomial over GF(2^8), evaluate the polynomial at n distinct points
for the n shares, and use Lagrange interpolation at x=0 to recover the
constant term (the original byte) from any t evaluations.
"""

import os

# --- GF(2^8) arithmetic, same field AES uses (irreducible poly 0x11B) ---
# Generator must be 3, not 2: 2 only has multiplicative order 51 under this
# reduction polynomial (it is NOT a primitive element), which silently
# corrupts a log/antilog table built on the assumption of a full 255-cycle.
# 3 is confirmed primitive (order 255) for 0x11B -- this is also the
# generator AES's own reference tables use.

_EXP = [0] * 512
_LOG = [0] * 256


def _xtime(a: int) -> int:
    """Multiply by x (i.e. by 2) in GF(2^8) mod 0x11B."""
    a <<= 1
    if a & 0x100:
        a ^= 0x11B
    return a & 0xFF


def _init_tables():
    x = 1
    for i in range(255):
        _EXP[i] = x
        _LOG[x] = i
        x = _xtime(x) ^ x  # x * 3 = (x * 2) XOR x
    for i in range(255, 512):
        _EXP[i] = _EXP[i - 255]


_init_tables()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def _gf_div(a: int, b: int) -> int:
    if a == 0:
        return 0
    if b == 0:
        raise ZeroDivisionError("GF(256) division by zero")
    return _EXP[(_LOG[a] - _LOG[b]) % 255]


# --- Shamir split / reconstruct ---

def split_secret(secret: bytes, threshold: int, total_shares: int) -> list:
    """
    Returns a list of `total_shares` shares. Each share is (x, y_bytes):
    x is the share's index (1..total_shares, never 0), y_bytes is the
    same length as `secret`. Any `threshold` of these shares reconstruct
    the secret exactly.
    """
    if not (1 <= threshold <= total_shares <= 255):
        raise ValueError("require 1 <= threshold <= total_shares <= 255")

    shares = [bytearray(len(secret)) for _ in range(total_shares)]

    for byte_index, secret_byte in enumerate(secret):
        # Random polynomial of degree (threshold - 1):
        #   f(x) = secret_byte + c1*x + c2*x^2 + ... + c_{t-1}*x^{t-1}
        coeffs = [secret_byte] + [os.urandom(1)[0] for _ in range(threshold - 1)]

        for share_index in range(total_shares):
            x = share_index + 1  # x = 0 would leak the secret byte directly -- never used
            y = 0
            x_power = 1  # x^0
            for coeff in coeffs:
                y ^= _gf_mul(coeff, x_power)
                x_power = _gf_mul(x_power, x)
            shares[share_index][byte_index] = y

    return [(i + 1, bytes(shares[i])) for i in range(total_shares)]


def reconstruct_secret(shares: list) -> bytes:
    """
    Reconstruct the secret from a list of (x, y_bytes) shares.
    Needs >= threshold correct shares from the *same* split_secret call;
    the function itself doesn't know what the threshold was, so passing
    too few shares silently returns garbage -- that's the honest
    information-theoretic property, not a bug: there is no way to
    detect "not enough shares" from the shares alone without an
    additional integrity check (see the commitment ledger module).
    """
    if not shares:
        raise ValueError("no shares provided")

    secret_length = len(shares[0][1])
    for x, y_bytes in shares:
        if len(y_bytes) != secret_length:
            raise ValueError("all shares must be the same length")

    secret = bytearray(secret_length)

    for byte_index in range(secret_length):
        # Lagrange interpolation at x = 0
        result = 0
        for i, (xi, yi_bytes) in enumerate(shares):
            yi = yi_bytes[byte_index]
            # Lagrange basis polynomial L_i(0) = product over j != i of (0 - xj) / (xi - xj)
            # In GF(2^8), subtraction == addition == XOR.
            num = 1
            den = 1
            for j, (xj, _) in enumerate(shares):
                if j == i:
                    continue
                num = _gf_mul(num, xj)          # (0 - xj) == xj in GF(2^8)
                den = _gf_mul(den, xi ^ xj)      # (xi - xj) == xi ^ xj
            term = _gf_mul(yi, _gf_div(num, den))
            result ^= term
        secret[byte_index] = result

    return bytes(secret)


if __name__ == "__main__":
    # Self-test / demo -- mirrors the 4-stakeholder, t=3 design from the spec
    secret = os.urandom(48)  # stand-in for a wrapped key blob
    threshold, total = 3, 4

    shares = split_secret(secret, threshold, total)
    print(f"Split into {total} shares, threshold {threshold}.")
    print("Share x-values:", [x for x, _ in shares])

    recovered = reconstruct_secret(shares[:threshold])
    assert recovered == secret
    print("Reconstruction from t=3 shares: OK, matches original secret.")

    recovered2 = reconstruct_secret([shares[0], shares[2], shares[3]])
    assert recovered2 == secret
    print("Reconstruction from a different subset of 3: OK.")

    recovered3 = reconstruct_secret(shares)
    assert recovered3 == secret
    print("Reconstruction from all 4 shares: OK.")

    wrong = reconstruct_secret(shares[:threshold - 1])
    print("Reconstruction attempt from only 2 shares (below threshold):")
    print("  Matches real secret:", wrong == secret, "(must be False)")
    assert wrong != secret
    print("All checks passed.")
