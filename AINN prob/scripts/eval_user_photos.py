"""
Test the image branch on YOUR OWN phone photos -- the one thing a
simulator can never fully replace.

Step 1 -- export the committed paper's pages so you can print them:
    python scripts/eval_user_photos.py --export-pages
  -> writes data/photos/print_me/page_01.png ... and print_me.pdf

Step 2 -- print one or two pages, photograph them with your phone (any
angle, any lighting, part of the page, a WhatsApp-forwarded copy, a
screenshot of the photo...), and copy the photos into data/photos/.
For a fair test, also photograph something that is NOT the paper
(your notes, a textbook page, a different printed paper).

Step 3 -- score them:
    python scripts/eval_user_photos.py
  Each photo is matched against every committed page. You get the
  pHash score, the ORB inlier count, the verified ORB score (the one the
  detector uses), and which page it thinks the photo shows.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from src.detection.exam_corpus import load_papers, TARGET_PAPERS  # noqa: E402
from src.detection.image_match import CommittedPaperImages  # noqa: E402
from src.detection.photo_sim import paper_pages  # noqa: E402

PHOTOS = ROOT / "data" / "photos"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="JEE Adv 2023 Paper 1", choices=TARGET_PAPERS)
    ap.add_argument("--export-pages", action="store_true")
    args = ap.parse_args()
    pages = paper_pages(args.paper, load_papers()[args.paper])

    if args.export_pages:
        out = PHOTOS / "print_me"
        out.mkdir(parents=True, exist_ok=True)
        imgs = []
        for i, (img, qids) in enumerate(pages, 1):
            img.save(out / f"page_{i:02d}.png")
            imgs.append(img.convert("RGB"))
        Image.init()                     # make sure every PIL codec is registered (PDF embeds pages as JPEG)
        imgs[0].save(out / "print_me.pdf", save_all=True, append_images=imgs[1:], resolution=150)
        print(f"Wrote {len(pages)} pages of '{args.paper}' to {out.relative_to(ROOT)} (PNG + print_me.pdf).")
        return

    photos = sorted(p for p in PHOTOS.glob("*") if p.suffix.lower() in EXTS)
    if not photos:
        print(f"No photos found in {PHOTOS.relative_to(ROOT)}/ -- see the instructions at the top of this script.")
        return
    index = CommittedPaperImages(args.paper, pages)
    print(f"Matching {len(photos)} photo(s) against the {len(pages)} committed pages of '{args.paper}'\n")
    print(f"{'photo':32s} {'pHash':>6s} {'inliers':>8s} {'runner-up':>9s} {'pixel r':>8s} {'verified':>9s}  verdict")
    for p in photos:
        r = index.match(Image.open(p).convert("RGB"))
        verdict = f"COMMITTED PAGE {r['page']}" if r["orb_score"] >= 0.5 else "no match"
        print(f"{p.name[:32]:32s} {r['phash_score']:6.2f} {r['inliers']:8d} {r['second_inliers']:9d} {r['ncc']:8.2f} "
              f"{r['orb_score']:9.2f}  {verdict}")
    print("\nverified >= 0.5 is treated as a match. 'inliers' = geometrically consistent keypoints on the best page;\n"
          "'runner-up' = the same on the second-best page (a real match is specific to ONE page).")


if __name__ == "__main__":
    main()
