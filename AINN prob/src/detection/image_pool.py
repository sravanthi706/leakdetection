"""
A fixed pool of realistic images that the session generator draws from,
with every image pre-scored against every protected (target) paper by
BOTH image matchers. Built once, cached to data/cache/, reused by every
experiment -- so the image branch in the end-to-end evaluation is using
real matcher outputs on real-looking photos, not a number made up by the
data generator (which is what the old `image_phash_hamming_to_true`
field was).

Pool contents (all rendered from REAL question text):
  target_photo    phone photos of pages of a protected paper   (leak evidence)
  target_digital  clean digital copy (screenshot/PDF render)   (leak evidence)
  past_photo      phone photos of PAST papers, same layout     (normal)
  coaching_photo  mock-test sheets built from past questions   (normal)
  notes_photo     photos of handwritten-style formula notes    (normal)
"""

import hashlib
import io
import json
import random
from multiprocessing import Pool
from pathlib import Path

import imagehash
from PIL import Image

from src.detection.exam_corpus import load_papers, TARGET_PAPERS, past_papers
from src.detection.image_match import CommittedPaperImages, orb_features
from src.detection.photo_sim import paper_pages, camera_photo, notes_page

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
POOL_VERSION = "v2"

_INDEXES = None


def _init_indexes():
    global _INDEXES
    papers = load_papers()
    _INDEXES = {t: CommittedPaperImages(t, paper_pages(t, papers[t])) for t in TARGET_PAPERS}


def _digital_copy(page: Image.Image, rng: random.Random) -> Image.Image:
    """A clean digital copy: downscaled + JPEG re-encoded, like a PDF page screenshot forwarded on chat."""
    f = rng.uniform(0.45, 0.8)
    img = page.convert("RGB").resize((int(page.width * f), int(page.height * f)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=rng.randint(50, 85))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _plan(seed: int, stress: bool = False) -> list:
    """
    Decide WHAT to render (cheap); rendering + scoring happens in worker processes.
    stress=True builds a HELD-OUT pool with harsher conditions than anything the
    matcher was tuned on: stronger distortion (1.0-1.8) and low-resolution
    forwards (long side 900 px).
    """
    rng = random.Random(seed)
    lo, hi = (1.0, 1.8) if stress else (0.6, 1.35)
    papers = load_papers()
    plan = []
    for t in TARGET_PAPERS:
        n_pages = len(paper_pages(t, papers[t]))
        for pg in range(n_pages):
            for k in range(2):
                plan.append({"kind": "target_photo", "paper": t, "page_idx": pg, "strength": rng.uniform(lo, hi)})
            if rng.random() < 0.4:
                plan.append({"kind": "target_digital", "paper": t, "page_idx": pg})
    for pp in past_papers():
        n_pages = len(paper_pages(pp, papers[pp]))
        for pg in rng.sample(range(n_pages), min(5, n_pages)):
            plan.append({"kind": "past_photo", "paper": pp, "page_idx": pg, "strength": rng.uniform(lo, hi)})
    for i in range(30):
        plan.append({"kind": "coaching_photo", "paper": rng.choice(past_papers()), "mock_no": rng.randint(1, 40),
                     "strength": rng.uniform(lo, hi)})
    for i in range(30):
        plan.append({"kind": "notes_photo", "strength": rng.uniform(lo, hi)})
    for i, item in enumerate(plan):
        item["image_id"] = f"{'stress' if stress else 'img'}{i:04d}"
        item["seed"] = seed * 100003 + i
        item["max_side"] = 900 if stress else 1600
    return plan


def render(item: dict) -> Image.Image:
    papers = load_papers()
    rng = random.Random(item["seed"])
    kind = item["kind"]
    if kind in ("target_photo", "target_digital", "past_photo"):
        page = paper_pages(item["paper"], papers[item["paper"]])[item["page_idx"]][0]
        return _digital_copy(page, rng) if kind == "target_digital" else camera_photo(page, rng, item["strength"], item.get("max_side", 1600))
    if kind == "coaching_photo":
        qs = rng.sample(papers[item["paper"]], min(6, len(papers[item["paper"]])))
        qs = [dict(q, number=i + 1) for i, q in enumerate(qs)]
        page = paper_pages("mock", qs, header_override=f"Coaching Mock Test {item['mock_no']} (JEE Advanced pattern)")[0][0]
        return camera_photo(page, rng, item["strength"], item.get("max_side", 1600))
    return camera_photo(notes_page(rng), rng, item["strength"], item.get("max_side", 1600))


def _score_item(item: dict) -> dict:
    img = render(item)
    ph, feats = imagehash.phash(img), orb_features(img)
    scores = {}
    for t, index in _INDEXES.items():
        r = index.match(photo_phash=ph, photo_orb=feats)
        scores[t] = {k: r[k] for k in ("phash_hamming", "phash_score", "inliers", "second_inliers", "ncc",
                                        "orb_raw_score", "orb_score", "score", "page", "qids")}
    out = dict(item)
    out["scores"] = scores
    return out


def build_image_pool(seed: int = 7, workers: int = 2, verbose: bool = True, stress: bool = False) -> list:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{POOL_VERSION}|{seed}|{TARGET_PAPERS}".encode()).hexdigest()[:10]
    cache = CACHE_DIR / f"image_pool_{POOL_VERSION}{'_stress' if stress else ''}_{key}.json"
    if cache.exists():
        return json.load(open(cache))
    plan = _plan(seed, stress)
    if verbose:
        print(f"Building image pool: rendering + scoring {len(plan)} images against "
              f"{len(TARGET_PAPERS)} committed papers (one-off, cached to {cache.relative_to(ROOT)}) ...", flush=True)
    with Pool(workers, initializer=_init_indexes) as pool:
        records = []
        for i, rec in enumerate(pool.imap(_score_item, plan, chunksize=4)):
            records.append(rec)
            if verbose and (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(plan)}", flush=True)
    json.dump(records, open(cache, "w"))
    return records


def image_evidence(record: dict, target_paper: str) -> dict:
    """The matcher outputs for one pool image against one committed paper."""
    return record["scores"][target_paper]


if __name__ == "__main__":
    import collections
    import sys
    stress = "--stress" in sys.argv
    pool = build_image_pool(seed=99 if stress else 7, stress=stress)
    print(collections.Counter(r["kind"] for r in pool))
