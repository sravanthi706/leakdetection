"""
Fast automated checks for the whole project (about 1-2 minutes on a laptop).

    pip install pytest
    python -m pytest -q

Custody tests need only the core requirements. Detection tests need the
datasets (`python scripts/download_data.py`) and skip cleanly if they are
missing. The PyTorch test skips if PyTorch Geometric is not installed.
"""

import importlib.util
import os
import random

import numpy as np
import pytest

from src.detection.exam_corpus import JEEBENCH_JSON

HAS_DATA = JEEBENCH_JSON.exists()
HAS_PYG = importlib.util.find_spec("torch_geometric") is not None
needs_data = pytest.mark.skipif(not HAS_DATA, reason="run scripts/download_data.py first")


# ------------------------------------------------------------------ custody layer

def test_aes_gcm_roundtrip_and_tamper():
    from src.crypto.aes_gcm import generate_content_key, encrypt_paper, decrypt_paper
    k = generate_content_key()
    sealed = encrypt_paper(b"paper", k, "P1", "2026-05-03T09:00", "C1")
    assert decrypt_paper(sealed, k) == b"paper"
    bad = dict(sealed, center_id="C2")
    with pytest.raises(ValueError):
        decrypt_paper(bad, k)


def test_shamir_any_threshold_subset_reconstructs():
    from src.crypto.shamir import split_secret, reconstruct_secret
    secret = os.urandom(32)
    shares = split_secret(secret, threshold=3, total_shares=5)
    for _ in range(20):
        assert reconstruct_secret(random.sample(shares, 3)) == secret


def test_ledger_detects_tampering():
    from src.custody.commitment_ledger import CommitmentLedger, generate_stakeholder_keypair, hash_paper
    priv, pub = generate_stakeholder_keypair()
    led = CommitmentLedger({"A": pub})
    led.append({"paper_hash": hash_paper(b"x")}, {"A": priv})
    assert led.verify_chain()[0]
    led.entries[0]["data"]["paper_hash"] = "0" * 64
    assert not led.verify_chain()[0]


# ------------------------------------------------------------------ detection layer

@needs_data
def test_latex_conversion_leaves_no_backslashes():
    from src.detection.exam_corpus import load_papers
    papers = load_papers()
    assert sum(len(v) for v in papers.values()) == 515
    assert not [q for v in papers.values() for q in v if "\\" in q["plain"]]


@needs_data
def test_answer_key_parser_formats():
    from src.detection.paper_matcher import _PAIR
    assert _PAIR.findall("Q3-B, Q4-AC, Q11-281") == [("3", "B"), ("4", "AC"), ("11", "281")]
    assert _PAIR.findall("Q.11) 281  Q12) 45") == [("11", "281"), ("12", "45")]
    assert _PAIR.findall("anyone have notes for chapter 5?") == []


@needs_data
def test_content_matcher_separates_committed_from_past_paper():
    from src.detection.exam_corpus import load_papers
    from src.detection.paper_matcher import matcher_for
    from src.detection import leak_text
    rng = random.Random(0)
    m = matcher_for("JEE Adv 2023 Paper 1")
    q = load_papers()["JEE Adv 2023 Paper 1"][10]
    past = load_papers()["JEE Adv 2019 Paper 1"][4]
    r_leak, r_past = m.score_messages([leak_text.verbatim(q, rng), leak_text.verbatim(past, rng)])
    assert r_leak["score"] > 0.9 and r_leak["matched_qid"] == q["qid"]
    assert r_past["score"] < 0.3
    key = leak_text.answer_key(load_papers()["JEE Adv 2023 Paper 1"][5:12], rng)
    assert m.score_messages([key])[0]["answer_key"] > 0.9


@needs_data
def test_orb_matcher_finds_photographed_page_and_rejects_other_paper():
    from src.detection.exam_corpus import load_papers
    from src.detection.image_match import CommittedPaperImages
    from src.detection.photo_sim import paper_pages, camera_photo
    papers = load_papers()
    pages = paper_pages("JEE Adv 2023 Paper 1", papers["JEE Adv 2023 Paper 1"])
    idx = CommittedPaperImages("JEE Adv 2023 Paper 1", pages)
    rng = random.Random(2)
    hit = idx.match(camera_photo(pages[4][0], rng))
    miss = idx.match(camera_photo(paper_pages("JEE Adv 2019 Paper 1", papers["JEE Adv 2019 Paper 1"])[3][0], rng))
    # the right page, and clearly above the other paper (99% of negatives in the benchmark pool score < 0.15)
    assert hit["page"] == 5 and hit["orb_score"] > 0.3
    assert miss["orb_score"] < 0.15 and hit["orb_score"] > miss["orb_score"] + 0.25


@needs_data
def test_generator_labels_match_event_ground_truth():
    from src.detection.realistic_data import generate_session, SCENARIOS, LEAK_MIX
    from src.detection.image_pool import build_image_pool
    pool = build_image_pool(verbose=False)
    for i, scen in enumerate(SCENARIOS):
        s = generate_session(f"t{i}", scen, random.Random(i), pool)
        has_leak = any(e["is_leak_event"] for e in s["events"])
        assert has_leak == (scen in LEAK_MIX), scen
        assert not any("broker" in e["sender_id"] or "leak" in e["group_id"] for e in s["events"]), "ids must not reveal roles"


@needs_data
def test_detector_trains_and_beats_chance_on_unseen_paper():
    from sklearn.metrics import roc_auc_score
    from src.detection.realistic_data import generate_hard_dataset
    from src.detection.engine import LeakDetector, annotate_events
    ds = generate_hard_dataset(n_sessions=240, seed=4)
    annotate_events(ds["sessions"], {r["image_id"]: r for r in ds["pool"]})
    test = [s for s in ds["sessions"] if s["exam_paper"] == "JEE Adv 2023 Paper 2"]
    rest = [s for s in ds["sessions"] if s["exam_paper"] != "JEE Adv 2023 Paper 2"]
    det = LeakDetector(seed=0, gnn_epochs=4, ae_epochs=10).fit(rest[:130], rest[130:])
    y = [s["label"] == "leak" for s in test]
    assert roc_auc_score(y, det.predict_proba(test)) > 0.8
    ex = det.explain(next(s for s in test if s["label"] == "leak"))
    assert set(ex["contributions"]) == set(det.branches)


@needs_data
@pytest.mark.skipif(not HAS_PYG, reason="PyTorch Geometric not installed")
def test_torch_gnn_fits_and_predicts():
    from src.detection.realistic_data import generate_hard_dataset
    from src.detection.gnn_torch import TorchCascadeGNN
    ds = generate_hard_dataset(n_sessions=80, seed=6)
    y = [s["label"] == "leak" for s in ds["sessions"]]
    m = TorchCascadeGNN(epochs=3).fit(ds["sessions"][:60], y[:60])
    p = m.predict_proba(ds["sessions"][60:])
    assert p.shape == (20,) and np.all((p >= 0) & (p <= 1))
