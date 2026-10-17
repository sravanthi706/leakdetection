"""
Realistic "phone photo of the question paper" generator.

The dominant real-world leak vector is a phone photo of a printed page
(NEET-UG 2026 CBI reporting: "photos of the paper" circulated on
WhatsApp/Telegram -- see synthetic_data.py citations). The old pHash
test only used clean digital near-duplicates (JPEG recompression,
resize, a straight crop). A real phone photo is much harder:

  * the page is shot at an angle        -> perspective warp
  * the phone is tilted                 -> rotation
  * the page sits on a desk / bed       -> cluttered background, page
                                           fills only part of the frame
  * room lighting is uneven             -> brightness gradient + shadow
  * hand shake / focus                  -> blur
  * cheap sensor                        -> noise, colour cast
  * WhatsApp re-encodes it              -> downscale to 1600 px + JPEG q~60-80
  * sometimes only part of the page     -> partial crop
  * sometimes it's a screenshot of a    -> phone UI bars, scaled copy
    forwarded photo

This module renders REAL question-paper pages (text from exam_corpus.py,
in an exam-style layout) and applies those distortions with OpenCV. It
is still a simulation of a camera, not a real camera -- which is why
data/photos/ exists: drop your own phone photos of a printed page there
and scripts/eval_user_photos.py scores them with the same matchers.
"""

import os
import random
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAGE_W, PAGE_H = 1240, 1754   # A4 at 150 dpi
MARGIN = 90


def _font(size: int, bold: bool = False):
    import matplotlib
    ttf_dir = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(ttf_dir / name), size)
    except OSError:
        return ImageFont.load_default()


def _question_lines(q: dict) -> list:
    lines = textwrap.wrap(f"Q.{q['number']}  " + q["stem_plain"], width=70)[:14]
    opts = []
    for k, v in list(q.get("options_plain", {}).items())[:4]:
        opts += ["    " + ln for ln in textwrap.wrap(f"({k}) {v}", width=64)[:2]]
    return lines + opts


def _draw_header(d, header: str, page_no: int):
    d.text((MARGIN, 50), header, font=_font(34, bold=True), fill=0)
    d.text((PAGE_W - MARGIN - 160, 58), f"Page {page_no}", font=_font(24), fill=60)
    d.line((MARGIN, 105, PAGE_W - MARGIN, 105), fill=0, width=3)


LINE_H = 40
BODY_TOP = 135
BODY_BOTTOM = PAGE_H - 120


def paper_pages(paper_name: str, questions: list, header_override: str = None) -> list:
    """
    Lays the paper out like a real exam booklet: questions are packed onto
    a page until it is full, then continue on the next page. Returns a list
    of (PIL.Image, [qids on that page]).
    """
    header = header_override or paper_name.replace("JEE Adv", "JEE (Advanced)")
    body = _font(28)
    pages, cur, cur_ids, y, page_no = [], None, [], BODY_TOP, 0

    def new_page():
        nonlocal cur, y, page_no, cur_ids
        page_no += 1
        cur = Image.new("L", (PAGE_W, PAGE_H), 255)
        _draw_header(ImageDraw.Draw(cur), header, page_no)
        y, cur_ids = BODY_TOP, []

    def close_page():
        d = ImageDraw.Draw(cur)
        d.line((MARGIN, PAGE_H - 90, PAGE_W - MARGIN, PAGE_H - 90), fill=120, width=2)
        d.text((MARGIN, PAGE_H - 75), "Space for rough work", font=_font(20), fill=120)
        pages.append((cur, list(cur_ids)))

    new_page()
    for q in questions:
        lines = _question_lines(q)
        need = len(lines) * LINE_H + 26
        if y + need > BODY_BOTTOM and cur_ids:
            close_page()
            new_page()
        d = ImageDraw.Draw(cur)
        for ln in lines:
            if y > BODY_BOTTOM:
                break
            d.text((MARGIN, y), ln, font=body, fill=0)
            y += LINE_H
        y += 26
        cur_ids.append(q["qid"])
    close_page()
    return pages


# ---------------------------------------------------------------------
# Camera simulation
# ---------------------------------------------------------------------

def _background(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Desk / bed / floor-like background: one muted surface colour with soft luminance blotches and grain."""
    np_rng = np.random.default_rng(rng.randrange(1 << 30))
    surfaces = [(62, 92, 128), (40, 60, 90), (95, 110, 120), (150, 150, 150), (70, 70, 72),
                (120, 150, 175), (30, 30, 35), (110, 125, 140), (160, 175, 190)]   # BGR: woods, greys, bedsheet blue
    base = np.array(rng.choice(surfaces), dtype=np.float32) * rng.uniform(0.8, 1.15)
    low = cv2.resize(np_rng.normal(1.0, 0.12, (6, 6)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    grain = np_rng.normal(0, 6, (h, w, 1)).astype(np.float32)
    return np.clip(base[None, None, :] * low[..., None] + grain, 0, 255).astype(np.uint8)


def camera_photo(page: Image.Image, rng: random.Random, strength: float = 1.0, max_side: int = 1600) -> Image.Image:
    """
    Returns a simulated phone photo (RGB) of `page`. `strength` in [0, 1.5]
    scales how aggressive the distortions are (0 ~ a flat scanner copy).
    """
    np_rng = np.random.default_rng(rng.randrange(1 << 30))
    src = cv2.cvtColor(np.array(page.convert("L")), cv2.COLOR_GRAY2BGR)
    h, w = src.shape[:2]

    # 1. partial capture: sometimes only a band of the page is photographed
    if rng.random() < 0.35 * strength:
        keep = rng.uniform(0.55, 0.9)
        top = int(rng.uniform(0, 1 - keep) * h)
        src = src[top: top + int(keep * h)]
        h = src.shape[0]

    # 2. place on a larger canvas (page occupies 60-92% of the frame)
    fill = rng.uniform(0.6, 0.92)
    H = int(max(h, w * 1.33) / fill)
    W = int(H * 0.75)
    canvas = _background(H, W, rng)

    # 3. perspective: jitter the 4 page corners
    scale = min(W * fill / w, H * fill / h)
    pw, ph = w * scale, h * scale
    cx, cy = W / 2 + rng.uniform(-0.05, 0.05) * W, H / 2 + rng.uniform(-0.05, 0.05) * H
    corners = np.float32([[cx - pw / 2, cy - ph / 2], [cx + pw / 2, cy - ph / 2],
                          [cx + pw / 2, cy + ph / 2], [cx - pw / 2, cy + ph / 2]])
    jitter = 0.09 * strength
    corners += np.float32([[rng.uniform(-jitter, jitter) * pw, rng.uniform(-jitter, jitter) * ph] for _ in range(4)])
    angle = np.deg2rad(rng.uniform(-8, 8) * strength)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    corners = (corners - [cx, cy]) @ R.T + [cx, cy]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), corners.astype(np.float32))
    warped = cv2.warpPerspective(src, M, (W, H), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    paper_tint = np.array([rng.uniform(0.9, 1.0), rng.uniform(0.92, 1.0), rng.uniform(0.93, 1.0)], np.float32)
    warped = (warped.astype(np.float32) * paper_tint)
    img = np.where(mask[..., None] > 0, warped, canvas.astype(np.float32))

    # 4. uneven lighting: linear gradient in a random direction + a soft shadow blob
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    theta = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(theta) * xx / W + np.sin(theta) * yy / H)
    grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    light = 1.0 - 0.35 * strength * grad
    if rng.random() < 0.5 * strength:
        sx, sy = rng.uniform(0, W), rng.uniform(0, H)
        rad = rng.uniform(0.2, 0.5) * W
        blob = np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * rad ** 2)))
        light *= 1.0 - 0.35 * blob
    img = img * light[..., None]

    # 5. blur (focus / hand shake) and sensor noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        k = rng.choice([3, 5]) if strength > 0.5 else 3
        img = cv2.GaussianBlur(img, (k, k), rng.uniform(0.3, 1.4) * strength + 0.1)
    if rng.random() < 0.25 * strength:
        L = rng.choice([5, 7, 9])
        kern = np.zeros((L, L), np.float32)
        kern[L // 2, :] = 1.0 / L
        rot = cv2.getRotationMatrix2D((L / 2, L / 2), rng.uniform(0, 180), 1.0)
        kern = cv2.warpAffine(kern, rot, (L, L))
        kern /= kern.sum() + 1e-6
        img = cv2.filter2D(img, -1, kern)
    noise = np_rng.normal(0, rng.uniform(2, 8) * strength, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 6. WhatsApp-style re-encode: cap the long side (1600 px by default), JPEG q 55-85
    long_side = max(img.shape[:2])
    if long_side > max_side:
        f = max_side / long_side
        img = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 85)])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # 7. sometimes it's a screenshot of the forwarded photo: phone UI bars + another re-encode
    if rng.random() < 0.2:
        bar = int(img.shape[0] * 0.06)
        ui = np.full((img.shape[0] + 2 * bar, img.shape[1], 3), rng.randint(10, 40), np.uint8)
        ui[bar: bar + img.shape[0]] = img
        cv2.putText(ui, "10:4" + str(rng.randint(0, 9)), (20, int(bar * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
        img = cv2.resize(ui, (int(ui.shape[1] * 0.75), int(ui.shape[0] * 0.75)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(50, 75)])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def notes_page(rng: random.Random) -> Image.Image:
    """A non-exam image people share constantly: handwritten-style formula notes on ruled paper."""
    img = Image.new("L", (PAGE_W, PAGE_H), 245)
    d = ImageDraw.Draw(img)
    for y in range(120, PAGE_H, 48):
        d.line((0, y, PAGE_W, y), fill=200, width=2)
    formulas = ["F = ma", "v^2 = u^2 + 2as", "PV = nRT", "E = hc / lambda", "sin^2 x + cos^2 x = 1",
                "d/dx (x^n) = n x^(n-1)", "pH = -log[H+]", "KE = 1/2 m v^2", "Ohm: V = IR",
                "Area = 1/2 ab sin C", "Gibbs: dG = dH - T dS", "lens: 1/f = 1/v - 1/u"]
    y = 130
    for _ in range(rng.randint(10, 22)):
        d.text((MARGIN + rng.randint(-20, 60), y), rng.choice(formulas), font=_font(rng.randint(30, 44)), fill=rng.randint(10, 80))
        y += 48 * rng.randint(1, 2)
        if y > PAGE_H - 60:
            break
    return img


if __name__ == "__main__":
    from src.detection.exam_corpus import load_papers
    out = Path(__file__).resolve().parents[2] / "reports" / "photo_examples"
    out.mkdir(parents=True, exist_ok=True)
    qs = load_papers()["JEE Adv 2023 Paper 1"]
    pages = paper_pages("JEE Adv 2023 Paper 1", qs)
    print(f"JEE Adv 2023 Paper 1 laid out on {len(pages)} pages:", [len(ids) for _, ids in pages])
    page = pages[3][0]
    page.save(out / "clean_page.png")
    rng = random.Random(4)
    for i in range(4):
        camera_photo(page, rng).save(out / f"phone_photo_{i}.jpg")
    notes_page(rng).save(out / "notes_page.png")
    print(f"Wrote a clean page, 4 simulated phone photos and a notes page to {out}")
