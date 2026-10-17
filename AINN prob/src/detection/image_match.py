"""
Image branch, upgraded: does this photo show a page of the COMMITTED paper?

Two matchers, both classical computer vision (no training, so no
training-data leakage and nothing to overfit):

1. pHash (the original method, kept for comparison)
   A 64-bit DCT fingerprint of the WHOLE image, compared by Hamming
   distance. Excellent for digital near-duplicates (re-compressed or
   resized copies of the same file). Expected weakness, measured in
   src/evaluation/image_benchmark.py rather than assumed: a phone photo
   changes the whole frame -- background, perspective, lighting -- so the
   global fingerprint of the photo no longer resembles the clean page.

2. ORB keypoints + RANSAC homography (the upgrade)
   Finds hundreds of small local features (corners of glyphs, rule lines)
   in the photo and in each committed page, matches them, and then asks
   RANSAC for ONE consistent perspective transform that maps many of the
   photo's features onto the page. A different page may share letter
   shapes, but it cannot share a consistent *geometric arrangement* of
   hundreds of them -- that is what the inlier count measures. This is
   the standard technique behind document-photo retrieval, and it is
   invariant to exactly the distortions a phone introduces: perspective,
   rotation, scale, partial crop, lighting.

Score = combined evidence in [0, 1], with the page that matched and the
number of geometric inliers kept for the explanation shown to a human.
"""

import math
from functools import lru_cache

import cv2
import imagehash
import numpy as np
from PIL import Image

from src.detection.phash_match import image_match_score as phash_score_from_hamming

ORB_FEATURES = 1500
WORK_LONG_SIDE = 1200
RATIO = 0.75
INLIER_SCALE = 18.0      # inliers at which the ORB score reaches ~0.63 (1 - 1/e); calibrated in image_benchmark


def _to_gray(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        img = np.array(img.convert("L"))
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    f = WORK_LONG_SIDE / max(h, w)
    if f < 1.0:
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return img


@lru_cache(maxsize=1)
def _orb():
    return cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2, nlevels=8, fastThreshold=12)


def orb_features(img):
    """Returns (keypoint coords, ORB descriptors, contrast-normalised grayscale used for verification)."""
    gray = _to_gray(img)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)  # normalise uneven lighting
    kp, des = _orb().detectAndCompute(gray, None)
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    return pts, des, gray


def orb_inliers(photo_feats, page_feats):
    """Returns (number of RANSAC inliers, homography photo->page or None)."""
    (p_pts, p_des, _), (q_pts, q_des, _) = photo_feats, page_feats
    if p_des is None or q_des is None or len(p_des) < 8 or len(q_des) < 8:
        return 0, None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(p_des, q_des, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < RATIO * n.distance]
    if len(good) < 8:
        return 0, None
    src = p_pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = q_pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0, maxIters=2000, confidence=0.995)
    if H is None or mask is None:
        return 0, None
    # reject degenerate homographies (a real photo of a page never mirrors or collapses it)
    det = np.linalg.det(H[:2, :2])
    if not (0.05 < abs(det) < 20) or det < 0:
        return 0, None
    return int(mask.sum()), H


def photometric_agreement(photo_feats, page_feats, H) -> float:
    """
    Pixel-level check after geometry: warp the photo onto the page and
    measure normalised cross-correlation over the page BODY (header and
    footer excluded, since those are identical on every page and every
    year's paper). Same page -> the same words sit in the same places ->
    high correlation. A different page that merely shares the layout
    template -> near zero.
    """
    if H is None:
        return 0.0
    g_photo, g_page = photo_feats[2], page_feats[2]
    h, w = g_page.shape
    warped = cv2.warpPerspective(g_photo, H, (w, h))
    valid = cv2.warpPerspective(np.full(g_photo.shape, 255, np.uint8), H, (w, h)) > 0
    body = np.zeros((h, w), bool)
    body[int(0.08 * h): int(0.92 * h), :] = True
    m = valid & body
    if m.sum() < 0.05 * h * w:
        return 0.0
    a = cv2.GaussianBlur(warped.astype(np.float32), (0, 0), 2.0)[m]
    b = cv2.GaussianBlur(g_page.astype(np.float32), (0, 0), 2.0)[m]
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


def orb_score_from_inliers(inliers: int) -> float:
    """Naive score from the raw inlier count alone (kept only so the benchmark can show why it is not enough)."""
    return float(1.0 - math.exp(-max(0, inliers - 6) / INLIER_SCALE))


def orb_score(best_inliers: int, second_inliers: int, ncc: float) -> float:
    """
    Verified local-feature score in [0, 1].

    Raw inlier counts are NOT enough -- found the hard way, see
    image_benchmark.py: exam pages share one layout template (header,
    footer, option indentation), so a clean page of a DIFFERENT paper can
    collect 100+ inliers against every committed page at once. What
    identifies the actual page is (a) how much better the best page
    matches than the runner-up (template matches are equally good on all
    pages; a real match is specific to one), and (b) whether the pixels
    agree once the photo is warped onto that page.
    """
    specific = max(0, best_inliers - second_inliers)
    s_geo = 1.0 - math.exp(-max(0, specific - 4) / 12.0)
    s_pix = float(np.clip((ncc - 0.2) / 0.4, 0.0, 1.0))
    return float(0.5 * s_geo + 0.5 * s_pix)


class CommittedPaperImages:
    """
    Pre-computes pHash + ORB features for every page of the committed
    paper once, so each incoming photo is compared against all pages
    quickly.
    """

    def __init__(self, paper_name: str, pages: list):
        """pages: list of (PIL page image, [qids on page]) from photo_sim.paper_pages."""
        self.paper_name = paper_name
        self.pages = []
        for idx, (img, qids) in enumerate(pages):
            self.pages.append({
                "page": idx + 1,
                "qids": qids,
                "phash": imagehash.phash(img),
                "orb": orb_features(img),
            })

    def match(self, photo=None, photo_phash=None, photo_orb=None) -> dict:
        """Pass the photo, or its pre-computed (phash, orb features) to avoid recomputing them per paper."""
        if photo_phash is None or photo_orb is None:
            pil = photo if isinstance(photo, Image.Image) else Image.fromarray(photo)
            photo_phash, photo_orb = imagehash.phash(pil), orb_features(pil)
        hd = min(int(photo_phash - p["phash"]) for p in self.pages)
        geo = []
        for p in self.pages:
            inl, H = orb_inliers(photo_orb, p["orb"])
            geo.append((inl, p, H))
        geo.sort(key=lambda g: -g[0])
        best_inl, best_page, best_H = geo[0]
        second_inl = geo[1][0] if len(geo) > 1 else 0
        ncc = photometric_agreement(photo_orb, best_page["orb"], best_H)
        result = {
            "phash_hamming": hd,
            "phash_score": float(phash_score_from_hamming(hd)),
            "inliers": best_inl,
            "second_inliers": second_inl,
            "ncc": round(ncc, 4),
            "page": best_page["page"] if best_inl > 0 else None,
            "qids": best_page["qids"] if best_inl > 0 else [],
        }
        result["orb_raw_score"] = orb_score_from_inliers(best_inl)
        result["orb_score"] = orb_score(best_inl, second_inl, ncc)
        result["score"] = max(result["phash_score"], result["orb_score"])
        return result


if __name__ == "__main__":
    import random
    import time
    from src.detection.exam_corpus import load_papers
    from src.detection.photo_sim import paper_pages, camera_photo, notes_page

    papers = load_papers()
    target = "JEE Adv 2023 Paper 1"
    t0 = time.time()
    index = CommittedPaperImages(target, paper_pages(target, papers[target]))
    print(f"Indexed {len(index.pages)} committed pages in {time.time() - t0:.1f}s")

    rng = random.Random(1)
    past = paper_pages("JEE Adv 2019 Paper 1", papers["JEE Adv 2019 Paper 1"])
    cases = [("photo of committed page", camera_photo(paper_pages(target, papers[target])[4][0], rng))]
    cases += [("photo of a PAST paper page (same format)", camera_photo(past[2][0], rng))]
    cases += [("photo of formula notes", camera_photo(notes_page(rng), rng))]
    for name, img in cases:
        t0 = time.time()
        r = index.match(img)
        print(f"{name:42s} pHash_hd={r['phash_hamming']:2d} phash={r['phash_score']:.2f} | inliers={r['inliers']:4d} "
              f"runner-up={r['second_inliers']:4d} ncc={r['ncc']:.2f} -> verified_orb={r['orb_score']:.2f} page={r['page']} "
              f"({time.time() - t0:.2f}s)")
