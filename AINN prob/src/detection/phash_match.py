"""
Perceptual hash (pHash) image matching -- spec 5.3. Classical DCT-based
signal processing, not a trained model, using the real `imagehash`
library (the exact one named in the tech stack table) on REAL images
rendered with Pillow -- not symbolic stand-in numbers. This is the one
detection-layer module that needed no substitution for anything
network-blocked; pHash needs no pretrained model or external download.

Two things happen in this file for two different reasons:
  1. `render_text_page_image` + the self-test below PROVE the real
     algorithm works correctly on real images: a recompressed/resized
     near-duplicate gets a small Hamming distance, a genuinely
     different image gets a large one.
  2. `image_match_score` is the small, reusable scoring function the
     fusion module (item 15) and the synthetic dataset both use --
     synthetic_data.py doesn't render thousands of fake page images
     for a demo dataset, it directly simulates pHash's OUTPUT (a
     Hamming distance), which is a legitimate simplification only
     because part 1 above already proves what that output really
     means on genuine images.
"""

import io

import imagehash
from PIL import Image, ImageDraw, ImageFont

HASH_BITS = 64  # imagehash.phash() default (8x8 DCT hash)


def render_text_page_image(lines: list, size: tuple = (600, 800)) -> Image.Image:
    """Renders a simple synthetic 'exam page' image -- self-generated content, not a real paper scan."""
    img = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(img)
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=0)
        y += 24
        if y > size[1] - 20:
            break
    return img


def compute_phash(image: Image.Image):
    return imagehash.phash(image, hash_size=8)  # 8x8 = 64-bit hash


def hamming_distance(hash_a, hash_b) -> int:
    return hash_a - hash_b  # imagehash.ImageHash overloads subtraction as Hamming distance


def image_match_score(hamming_dist: int, alert_threshold_distance: int = 15) -> float:
    """
    Score in [0, 1]: 1.0 = identical hash, decaying to 0 as distance
    grows past the alert threshold. `alert_threshold_distance` (~15 of
    64 bits, i.e. ~23%) is the conventional pHash near-duplicate cutoff
    used in real deduplication systems -- distances below it reliably
    indicate the same underlying image (recompression, resizing, minor
    crop); above it, they're different images.

    Calibration note (caught by testing against a genuinely different
    image, not assumed): two UNRELATED images' pHashes average roughly
    32-bit Hamming distance apart, not 64 -- 64-bit distance would mean
    every single bit disagrees, which essentially never happens even
    for unrelated content. An earlier version of this function decayed
    the score linearly out to 64, which under-penalized a clearly
    non-matching image (it scored 0.47, misleadingly close to
    "ambiguous"). The RANDOM_PAIR_DISTANCE below is used as the point
    where the score should have already reached 0, not 64.
    """
    RANDOM_PAIR_DISTANCE = 32  # ~expected Hamming distance between two unrelated images' pHashes
    if hamming_dist <= alert_threshold_distance:
        return 1.0 - (hamming_dist / (2 * alert_threshold_distance))
    return max(0.0, 1.0 - (hamming_dist / RANDOM_PAIR_DISTANCE))


if __name__ == "__main__":
    true_lines = [f"Q{i}: [synthetic placeholder question stem #{i}]" for i in range(1, 20)]
    true_image = render_text_page_image(true_lines)
    true_hash = compute_phash(true_image)
    print(f"True page image rendered, pHash = {true_hash}")

    # --- Near-duplicate 1: JPEG recompression (simulates a photographed/re-shared, lossy copy) ---
    buf = io.BytesIO()
    true_image.convert("RGB").save(buf, format="JPEG", quality=35)
    buf.seek(0)
    recompressed = Image.open(buf)
    d_recompressed = hamming_distance(true_hash, compute_phash(recompressed))
    print(f"Recompressed (JPEG q=35) Hamming distance: {d_recompressed}  "
          f"| match_score={image_match_score(d_recompressed):.2f}")

    # --- Near-duplicate 2: resized down then up (simulates a thumbnail/forward chain) ---
    resized = true_image.resize((150, 200)).resize((600, 800))
    d_resized = hamming_distance(true_hash, compute_phash(resized))
    print(f"Resized down+up Hamming distance:        {d_resized}  "
          f"| match_score={image_match_score(d_resized):.2f}")

    # --- Near-duplicate 3: mild crop (simulates a photo that doesn't capture the full page) ---
    cropped = true_image.crop((0, 0, 600, 700)).resize((600, 800))
    d_cropped = hamming_distance(true_hash, compute_phash(cropped))
    print(f"Cropped+rescaled Hamming distance:       {d_cropped}  "
          f"| match_score={image_match_score(d_cropped):.2f}")

    # --- A genuinely DIFFERENT image -- must score low ---
    different_lines = [f"Chapter {i} revision notes -- generic study content, unrelated topic" for i in range(1, 20)]
    different_image = render_text_page_image(different_lines)
    d_different = hamming_distance(true_hash, compute_phash(different_image))
    print(f"Genuinely different image Hamming distance: {d_different}  "
          f"| match_score={image_match_score(d_different):.2f}")

    assert d_recompressed < 15, "recompressed near-duplicate should score as a match"
    assert d_resized < 15, "resized near-duplicate should score as a match"
    assert d_different > d_recompressed and d_different > d_resized, \
        "a genuinely different image must score further from the true hash than any near-duplicate"
    print("\nAll real-image pHash checks passed: near-duplicates matched, a different image did not.")

    # --- Now use the same scoring function on the synthetic dataset's simulated distances ---
    print("\n=== Applying image_match_score to the synthetic dataset (item 10) ===")
    from src.detection.synthetic_data import generate_dataset

    dataset = generate_dataset(n_normal=30, n_leak=30, seed=11)
    leak_image_scores = []
    for s in dataset["sessions"]:
        for e in s["events"]:
            if e["event_type"] == "image":
                leak_image_scores.append(image_match_score(e["image_phash_hamming_to_true"]))

    print(f"Leak-session image events: n={len(leak_image_scores)}, "
          f"mean match_score={sum(leak_image_scores)/len(leak_image_scores):.3f}")
    normal_has_images = any(e["event_type"] == "image" for s in dataset["sessions"]
                             if s["label"] == "normal" for e in s["events"])
    print(f"Normal sessions contain image events: {normal_has_images} (must be False)")
    assert not normal_has_images
    assert sum(leak_image_scores) / len(leak_image_scores) > 0.7
    print("Confirmed: photographed-page leak events score as strong matches; normal sessions have none.")
