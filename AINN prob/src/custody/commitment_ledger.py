"""
Append-only, hash-chained commitment ledger, co-signed by the
independent custody stakeholders.

Two separate things happen here, and it's worth keeping them distinct:
  1. Hash chaining -- each entry embeds the hash of the previous entry,
     so altering or removing any past entry breaks every entry after it.
     This is what makes the ledger tamper-evident.
  2. Co-signing -- each entry is signed (Ed25519) by multiple independent
     stakeholders, so no single party (not even whoever runs the ledger
     server) can forge or backdate an entry unilaterally. This is what
     makes the *initial timestamp* trustworthy despite point 4.3 in the
     spec: "the system still relies on the initial commitment timestamp
     being honest" -- co-signing is the mitigation for that.
"""

import hashlib
import json
import time

from Crypto.PublicKey import ECC
from Crypto.Signature import eddsa


def generate_stakeholder_keypair():
    """One stakeholder's long-term signing identity for the ledger."""
    key = ECC.generate(curve='ed25519')
    return key, key.public_key()


def _sign(private_key, message: bytes) -> bytes:
    return eddsa.new(private_key, 'rfc8032').sign(message)


def _verify(public_key, message: bytes, signature: bytes) -> bool:
    try:
        eddsa.new(public_key, 'rfc8032').verify(message, signature)
        return True
    except ValueError:
        return False


def _canonical_bytes(obj) -> bytes:
    """Deterministic JSON encoding so hashing/signing always agree byte-for-byte."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


class CommitmentLedger:
    """
    entries[i] = {
        "index": i,
        "timestamp": <float>,
        "data": {...},                 # e.g. {"paper_id": ..., "paper_hash": "..."}
        "prev_hash": "<hex>",          # hash of entries[i-1], or "0"*64 for the genesis entry
        "entry_hash": "<hex>",         # hash of everything above
        "signatures": {stakeholder_id: "<hex signature>", ...}
    }
    """

    GENESIS_PREV_HASH = "0" * 64

    def __init__(self, stakeholder_public_keys: dict):
        # stakeholder_id -> ECC public key. Known/enrolled signers only.
        self.stakeholder_public_keys = stakeholder_public_keys
        self.entries = []

    def _entry_hash(self, index, timestamp, data, prev_hash) -> str:
        payload = {"index": index, "timestamp": timestamp, "data": data, "prev_hash": prev_hash}
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()

    def append(self, data: dict, cosigners: dict) -> dict:
        """
        cosigners: {stakeholder_id: private_key} for every stakeholder
        co-signing this specific entry. Must be a subset of
        self.stakeholder_public_keys (by id).
        """
        for sid in cosigners:
            if sid not in self.stakeholder_public_keys:
                raise ValueError(f"unknown stakeholder id: {sid}")

        index = len(self.entries)
        timestamp = time.time()
        prev_hash = self.entries[-1]["entry_hash"] if self.entries else self.GENESIS_PREV_HASH
        entry_hash = self._entry_hash(index, timestamp, data, prev_hash)

        signatures = {
            sid: _sign(priv, bytes.fromhex(entry_hash)).hex()
            for sid, priv in cosigners.items()
        }

        entry = {
            "index": index,
            "timestamp": timestamp,
            "data": data,
            "prev_hash": prev_hash,
            "entry_hash": entry_hash,
            "signatures": signatures,
        }
        self.entries.append(entry)
        return entry

    def verify_chain(self, required_signers: set = None) -> tuple:
        """
        Returns (is_valid: bool, problem: str | None).
        Checks: (a) hash chain integrity, (b) every signature on every
        entry is cryptographically valid, (c) if required_signers is
        given, every entry has signatures from at least that set.
        """
        prev_hash = self.GENESIS_PREV_HASH

        for entry in self.entries:
            if entry["prev_hash"] != prev_hash:
                return False, f"chain broken at index {entry['index']}: prev_hash mismatch"

            recomputed = self._entry_hash(entry["index"], entry["timestamp"], entry["data"], entry["prev_hash"])
            if recomputed != entry["entry_hash"]:
                return False, f"entry {entry['index']} hash does not match its own contents (tampered)"

            for sid, sig_hex in entry["signatures"].items():
                pub = self.stakeholder_public_keys.get(sid)
                if pub is None:
                    return False, f"entry {entry['index']} signed by unknown stakeholder {sid}"
                if not _verify(pub, bytes.fromhex(entry["entry_hash"]), bytes.fromhex(sig_hex)):
                    return False, f"entry {entry['index']} has an invalid signature from {sid}"

            if required_signers is not None:
                missing = required_signers - set(entry["signatures"].keys())
                if missing:
                    return False, f"entry {entry['index']} missing required signers: {missing}"

            prev_hash = entry["entry_hash"]

        return True, None

    def find_commitment(self, paper_hash_hex: str):
        """Post-incident verification: does a leaked-document hash match any committed entry?"""
        for entry in self.entries:
            if entry["data"].get("paper_hash") == paper_hash_hex:
                return entry
        return None


def hash_paper(paper_bytes: bytes) -> str:
    return hashlib.sha256(paper_bytes).hexdigest()


if __name__ == "__main__":
    # Self-test / demo -- 4 stakeholders co-sign the pre-exam commitment
    stakeholders = {}
    for sid in ["NTA_HQ", "STATE_AUTHORITY", "INDEPENDENT_AUDIT", "CUSTODIAN"]:
        priv, pub = generate_stakeholder_keypair()
        stakeholders[sid] = {"priv": priv, "pub": pub}

    ledger = CommitmentLedger({sid: s["pub"] for sid, s in stakeholders.items()})

    real_paper = b"Q1. ... [the real finalized exam paper] ..."
    entry = ledger.append(
        data={"paper_id": "PHY-2026-A1", "paper_hash": hash_paper(real_paper), "window_start": "2026-09-20T09:00:00Z"},
        cosigners={sid: s["priv"] for sid, s in stakeholders.items()},  # all 4 co-sign the commitment
    )
    print("Committed entry 0, hash:", entry["entry_hash"][:16], "... signed by", list(entry["signatures"].keys()))

    # A second, unrelated ledger event (e.g. tamper-response log entry from a later module)
    ledger.append(
        data={"event": "node_enrolled", "node": "CTR-0042"},
        cosigners={"NTA_HQ": stakeholders["NTA_HQ"]["priv"]},
    )

    valid, problem = ledger.verify_chain(required_signers={"NTA_HQ"})
    print("Chain valid:", valid, problem)

    # Post-incident check: does a leaked document match the committed hash?
    found = ledger.find_commitment(hash_paper(real_paper))
    print("Leaked-document match found:", found is not None)

    fake_paper = b"a completely different, fabricated paper"
    found_fake = ledger.find_commitment(hash_paper(fake_paper))
    print("Fabricated document match found (must be False):", found_fake is not None)

    # Tamper test: alter an already-committed entry's data after the fact
    ledger.entries[0]["data"]["paper_hash"] = "0" * 64
    valid2, problem2 = ledger.verify_chain()
    print("Chain valid after tampering with entry 0 (must be False):", valid2, "--", problem2)
