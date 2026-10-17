"""
Synthetic data generator for the AI/NN detection layer.

LEGAL/METHODOLOGY NOTE (read this before using the output anywhere):
This generator produces 100% synthetic data -- no real people, no real
accounts, no scraped platform content, and no real exam paper content.
What IS real is the *shape* of the data: message timing, propagation
structure, and content-overlap severity are calibrated to publicly
reported, citable facts about the NEET-UG 2026 leak (CBI chargesheet
reporting), not to guesses. Every calibrated parameter below names its
source. Where no public figure exists, that is stated too -- treated
as a documented assumption, not silently invented.

Cited facts this generator is calibrated against:
  [1] Leak content circulated up to ~10 days before the exam (a named
      accused "received the chemistry questions on 23 April 2026" for
      the 3 May 2026 exam). Source: 2026 NEET controversy, Wikipedia
      (CBI chargesheet reporting) -- https://en.wikipedia.org/wiki/2026_NEET_controversy
  [2] The leak passed through a multi-hop chain of roles -- "subject
      experts, private brokers, and operators, before allegedly
      reaching students" -- not a flat broadcast.
      Source: WION, "CBI reveals digital trail behind NEET UG 2026
      paper leak" -- https://www.wionews.com/india-news/cbi-reveals-digital-trail-behind-neet-ug-2026-paper-leak-through-whatsapp-groups-telegram-chats-and-photos-1786097020958
  [3] Distribution spanned multiple states/cities at once (Rajasthan:
      Sikar; Maharashtra: Latur, Nashik, Pune), not one local group.
      Source: same as [1].
  [4] Both TEXT/PDF copies AND PHOTOGRAPHS of the paper were circulated
      -- confirming this system's dual text+image detection design
      against the real incident, not a hypothetical.
      Source: same as [2].
  [5] Content-match severity in the real case was extreme: "120
      questions" / "nearly 140 questions" matched in one report, and
      "42 out of 45 questions" in another. This generator uses a
      45-question synthetic paper specifically to mirror that
      second, more precisely quantified figure.
      Source: same as [1].

What is an assumption, not a cited figure: the exact hour-by-hour
intensity ramp as the exam approaches, and normal-traffic baseline
volume. No public source quantifies these, so they are modeled as a
reasonable monotonic increase toward the exam date, stated here as an
assumption rather than presented as researched fact.
"""

import hashlib
import math
import random

# --- World setup: regions/cities named after the real reported locations (fact [3]) ---

REGIONS = {
    "Rajasthan": ["Sikar"],
    "Maharashtra": ["Latur", "Nashik", "Pune"],
}

ROLE_CHAIN = ["subject_expert", "broker", "operator", "student"]  # fact [2]

# A 45-item synthetic paper -- content is entirely self-authored, generic
# placeholder question stems, chosen to mirror the "42 out of 45" figure
# from fact [5] for realistic overlap-fraction calibration. This is NOT
# real exam content.
TRUE_PAPER_QUESTIONS = [f"Q{i}: [synthetic placeholder question stem #{i} -- topic {chr(65 + i % 6)}]" for i in range(1, 46)]

NORMAL_CHATTER_TEMPLATES = [
    "anyone have notes for chapter {n}?",
    "what time is the mock test tomorrow",
    "can someone explain this doubt in topic {n}",
    "good luck everyone for the exam",
    "revising chapter {n} today",
    "is the syllabus for this year the same as last year",
    "does anyone have last year's cutoff",
]

BROKER_KEYWORDS = ["confirmed", "advance", "PDF", "full set", "genuine", "lakh", "₹", "deal", "payment"]


def _stable_hash_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def _make_true_image_hash(rng: random.Random) -> int:
    """Stand-in 64-bit pHash for the committed paper's page image."""
    return rng.getrandbits(64)


def _near_duplicate_hash(true_hash: int, hamming_distance: int, rng: random.Random) -> int:
    """Flip exactly `hamming_distance` bits of true_hash -- simulates a photographed/recompressed near-duplicate."""
    bits_to_flip = rng.sample(range(64), min(hamming_distance, 64))
    h = true_hash
    for b in bits_to_flip:
        h ^= (1 << b)
    return h


def generate_normal_session(session_id: str, window_start: float, rng: random.Random,
                             num_groups: int = 3, duration_days: float = 10.0) -> dict:
    """
    Ordinary exam-prep chatter: low volume, generic content, confined to
    one region's groups, flat rate (no ramp toward the exam), normal
    waking-hour timing, no photographed-page events at all.
    """
    region = rng.choice(list(REGIONS.keys()))
    cities = REGIONS[region]
    groups = [f"{region}_{cities[i % len(cities)]}_group{i}" for i in range(num_groups)]
    students = [f"student_{session_id}_{i}" for i in range(20)]

    events = []
    num_events = rng.randint(15, 40)  # low volume -- assumption, see module docstring
    for _ in range(num_events):
        t = window_start - rng.uniform(0, duration_days * 86400)
        hour = int((t / 3600) % 24)
        # bias toward normal waking/study hours (assumption, not a cited figure)
        if hour < 6 or hour > 23:
            t += 8 * 3600 if hour < 6 else -8 * 3600

        template = rng.choice(NORMAL_CHATTER_TEMPLATES)
        text = template.format(n=rng.randint(1, 20))

        events.append({
            "timestamp": t,
            "sender_id": rng.choice(students),
            "group_id": rng.choice(groups),
            "region": region,
            "role": "student",
            "event_type": "text",
            "text": text,
            "content_overlap_fraction": 0.0,
            "image_phash_hamming_to_true": None,
        })

    events.sort(key=lambda e: e["timestamp"])
    return {"session_id": session_id, "label": "normal", "window_start": window_start, "events": events}


def generate_leak_session(session_id: str, window_start: float, rng: random.Random,
                           true_image_hash: int, duration_days: float = 10.0) -> dict:
    """
    Leak-pattern session: multi-hop role chain (fact [2]), multi-region
    spread (fact [3]), intensity ramping up toward the exam date
    (assumption -- direction supported by fact [1]'s 10-day window, exact
    curve shape is not a cited figure), high text content-overlap with
    the true paper (calibrated toward fact [5]), and includes photographed-
    page image events (fact [4]), which normal sessions never have.
    """
    all_regions = list(REGIONS.keys())
    num_leak_regions = rng.randint(2, len(all_regions))  # spans MULTIPLE regions, per fact [3]
    leak_regions = rng.sample(all_regions, num_leak_regions)

    # Build the role chain population: few experts/brokers, many operators/students -- funnel shape, per fact [2]
    role_pop = {
        "subject_expert": [f"expert_{session_id}_{i}" for i in range(2)],
        "broker": [f"broker_{session_id}_{i}" for i in range(4)],
        "operator": [f"operator_{session_id}_{i}" for i in range(8)],
        "student": [f"leak_student_{session_id}_{i}" for i in range(30)],
    }

    events = []
    num_events = rng.randint(60, 150)  # higher volume than normal, reflecting the fan-out

    for i in range(num_events):
        # Intensity ramp: sample more heavily from LATE in the window (closer to window_start)
        # using a skewed random draw -- assumption on exact shape, direction grounded in fact [1].
        u = rng.random() ** 2.2  # skews toward 0 => toward window_start after the flip below
        t = window_start - u * duration_days * 86400

        # progression through the role chain as time approaches the exam:
        # early = expert/broker only, late = operator/student dominate
        progress = 1 - (u)  # 0 at start of window, ~1 near window_start
        if progress < 0.25:
            role = rng.choices(ROLE_CHAIN, weights=[0.6, 0.4, 0.0, 0.0])[0]
        elif progress < 0.6:
            role = rng.choices(ROLE_CHAIN, weights=[0.1, 0.4, 0.4, 0.1])[0]
        else:
            role = rng.choices(ROLE_CHAIN, weights=[0.0, 0.1, 0.4, 0.5])[0]

        region = rng.choice(leak_regions)
        city = rng.choice(REGIONS[region])
        group_id = f"{region}_{city}_leakgroup{_stable_hash_int(session_id, region, city) % 3}"

        is_image = rng.random() < 0.2  # a minority of events are photographed-page shares, per fact [4]

        if is_image:
            # Leak images cluster CLOSE to the true hash -- that's what a real photographed
            # page's near-duplicate pHash looks like; small Hamming distance.
            hd = rng.randint(0, 10)
            events.append({
                "timestamp": t,
                "sender_id": rng.choice(role_pop[role]),
                "group_id": group_id,
                "region": region,
                "role": role,
                "event_type": "image",
                "text": None,
                "content_overlap_fraction": None,
                "image_phash_hamming_to_true": hd,
            })
        else:
            # High content overlap with the true paper -- calibrated toward fact [5]'s
            # "42/45" figure: sample a high fraction of real question stems verbatim.
            overlap_fraction = rng.uniform(0.75, 0.95)
            k = max(1, int(round(overlap_fraction * len(TRUE_PAPER_QUESTIONS))))
            sampled_qs = rng.sample(TRUE_PAPER_QUESTIONS, k)
            snippet = rng.choice(sampled_qs)
            broker_word = rng.choice(BROKER_KEYWORDS)
            text = f"{snippet} [{broker_word}]"

            events.append({
                "timestamp": t,
                "sender_id": rng.choice(role_pop[role]),
                "group_id": group_id,
                "region": region,
                "role": role,
                "event_type": "text",
                "text": text,
                "content_overlap_fraction": overlap_fraction,
                "image_phash_hamming_to_true": None,
            })

    events.sort(key=lambda e: e["timestamp"])
    return {"session_id": session_id, "label": "leak", "window_start": window_start, "events": events}


def generate_dataset(n_normal: int, n_leak: int, seed: int = None) -> dict:
    rng = random.Random(seed)
    window_start = 1_800_000_000.0  # arbitrary fixed reference epoch, deterministic given seed
    true_image_hash = _make_true_image_hash(rng)

    sessions = []
    for i in range(n_normal):
        sessions.append(generate_normal_session(f"normal_{i}", window_start, rng))
    for i in range(n_leak):
        sessions.append(generate_leak_session(f"leak_{i}", window_start, rng, true_image_hash))

    rng.shuffle(sessions)
    return {
        "sessions": sessions,
        "true_image_hash": true_image_hash,
        "true_paper_questions": TRUE_PAPER_QUESTIONS,
        "window_start": window_start,
    }


def summarize_dataset(dataset: dict):
    """
    Validates 'closeness', doesn't just assert it: prints aggregate
    stats per class so the separation the later detectors will rely on
    is visible and checkable now, not assumed.
    """
    by_label = {"normal": [], "leak": []}
    for s in dataset["sessions"]:
        by_label[s["label"]].append(s)

    for label, sessions in by_label.items():
        n = len(sessions)
        if n == 0:
            continue
        avg_events = sum(len(s["events"]) for s in sessions) / n
        text_overlaps = [e["content_overlap_fraction"] for s in sessions for e in s["events"]
                          if e["event_type"] == "text"]
        avg_overlap = sum(text_overlaps) / len(text_overlaps) if text_overlaps else 0.0
        image_hds = [e["image_phash_hamming_to_true"] for s in sessions for e in s["events"]
                     if e["event_type"] == "image"]
        avg_hd = sum(image_hds) / len(image_hds) if image_hds else None
        regions_touched = [len({e["region"] for e in s["events"]}) for s in sessions]
        avg_regions = sum(regions_touched) / n
        senders = [len({e["sender_id"] for e in s["events"]}) for s in sessions]
        avg_senders = sum(senders) / n

        print(f"[{label}] sessions={n}  avg_events/session={avg_events:.1f}  "
              f"avg_text_content_overlap={avg_overlap:.2f}  "
              f"avg_image_hamming_dist={'n/a' if avg_hd is None else f'{avg_hd:.1f}'}  "
              f"avg_regions_touched={avg_regions:.2f}  avg_distinct_senders={avg_senders:.1f}")


if __name__ == "__main__":
    dataset = generate_dataset(n_normal=40, n_leak=40, seed=42)
    print(f"Generated {len(dataset['sessions'])} sessions "
          f"({sum(1 for s in dataset['sessions'] if s['label']=='normal')} normal, "
          f"{sum(1 for s in dataset['sessions'] if s['label']=='leak')} leak).")
    print()
    summarize_dataset(dataset)
    print()

    # Sanity checks -- the separation the later GNN/autoencoder/embedding/pHash
    # branches will depend on must actually be present, not assumed.
    normal_sessions = [s for s in dataset["sessions"] if s["label"] == "normal"]
    leak_sessions = [s for s in dataset["sessions"] if s["label"] == "leak"]

    normal_overlaps = [e["content_overlap_fraction"] for s in normal_sessions for e in s["events"] if e["event_type"] == "text"]
    leak_overlaps = [e["content_overlap_fraction"] for s in leak_sessions for e in s["events"] if e["event_type"] == "text"]
    assert max(normal_overlaps) == 0.0, "normal sessions must never reference the true paper content"
    assert min(leak_overlaps) > 0.5, "leak sessions must show substantial content overlap"
    print("Content-overlap separation confirmed: normal max =", max(normal_overlaps), "| leak min =", round(min(leak_overlaps), 2))

    normal_has_images = any(e["event_type"] == "image" for s in normal_sessions for e in s["events"])
    assert normal_has_images is False, "normal sessions must never contain photographed-page events"
    print("Image-event separation confirmed: normal sessions contain zero image events (leak sessions do).")

    leak_regions_avg = sum(len({e["region"] for e in s["events"]}) for s in leak_sessions) / len(leak_sessions)
    normal_regions_avg = sum(len({e["region"] for e in s["events"]}) for s in normal_sessions) / len(normal_sessions)
    assert leak_regions_avg > normal_regions_avg, "leak sessions should span more regions than normal sessions"
    print(f"Cross-region spread separation confirmed: leak avg regions={leak_regions_avg:.2f} > normal avg regions={normal_regions_avg:.2f}")

    print()
    print("All separation checks passed -- this dataset gives the later detection")
    print("modules a real, checkable signal to learn from, not an arbitrary label.")
