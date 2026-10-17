"""
HARD, realistic session generator ("hard mode") -- replaces the easy
generator for all evaluation.

Why the old generator gave F1 = 1.00 on every seed: every leak session
had verbatim paper content + photos + multi-region spread + broker
words, and every normal session had none of those. Any one signal could
separate the classes on its own. Real monitoring data doesn't look like
that. This generator deliberately builds the confusers that make the
real problem hard, and uses REAL exam text (exam_corpus.py), REAL-looking
phone photos scored by the real image matchers (image_pool.py), and the
realistic ways questions get rewritten (leak_text.py).

NORMAL scenarios (label 0) -- including ones that LOOK like leaks:
  study_chatter          ordinary prep chat, sometimes across states
  past_paper_discussion  students sharing/solving PAST papers verbatim,
                         with photos and answer keys -- same syllabus,
                         same layout, same exam body as the real paper
  coaching_broadcast     a coaching channel fanning out mock tests,
                         fee/payment messages and photos to many groups
                         in several states, intensifying before the exam
  fake_leak_scam         "100% GENUINE PAPER, pay Rs 5000 advance" --
                         broker language, forwarding cascades, fake
                         answer keys and old papers passed off as proof.
                         Very common in reality; it is NOT a leak of the
                         committed paper, and flagging it is a false alarm.

LEAK scenarios (label 1) -- including ones that DON'T look like leaks:
  full_leak         the big multi-hop, multi-state cascade (the NEET-2026 shape)
  partial_leak      only 2-6 questions escape
  paraphrased_leak  questions only ever appear reworded / in Hinglish
  answer_key_leak   only "Q3-B, Q4-AC, Q11-281" lists -- no question text
  photo_leak        only phone photos, with uninformative captions
  quiet_leak        tiny closed group, one city, no broker talk, few messages
  buried_leak       1-3 leaked items hidden inside a large, normal group chat
  coordination_only_leak
                    a REAL leak operation whose monitored channels only carry
                    the coordination ("paper aa gaya", payments, "DM me") --
                    the paper itself moves in private chats we cannot see.
                    It looks almost exactly like a scam. This is the
                    deliberately near-impossible case: it caps achievable
                    recall and tests whether propagation structure alone
                    (the GNN) carries any signal.

Every leak session also carries ordinary chatter, and every event keeps
ground truth (`is_leak_event`, `qid`) so explanations can be checked.

Assumptions (stated, not hidden): scenario mix, volumes, timing and
cascade delays are modelling choices -- no public dataset quantifies
them for exam leaks. The CITED facts from synthetic_data.py still shape
the full_leak scenario (multi-hop role chain, multi-state spread,
photos + text, ~10-day window).
"""

import math
import random

from src.detection.exam_corpus import load_papers, TARGET_PAPERS, past_papers
from src.detection import leak_text
from src.detection.image_pool import build_image_pool

REGIONS = {
    "Rajasthan": ["Kota", "Sikar", "Jaipur"],
    "Maharashtra": ["Latur", "Pune", "Nashik"],
    "Bihar": ["Patna"],
    "Telangana": ["Hyderabad"],
    "Delhi": ["Delhi"],
    "Uttar Pradesh": ["Lucknow", "Prayagraj"],
}

NORMAL_MIX = {"study_chatter": 0.35, "past_paper_discussion": 0.25, "coaching_broadcast": 0.2, "fake_leak_scam": 0.2}
LEAK_MIX = {"full_leak": 0.18, "partial_leak": 0.13, "paraphrased_leak": 0.13, "answer_key_leak": 0.11,
            "photo_leak": 0.12, "quiet_leak": 0.11, "buried_leak": 0.12, "coordination_only_leak": 0.10}

WINDOW_DAYS = 10

# One shared vocabulary of "leak-sounding" talk, used by real leak networks AND by scammers.
# (A first version gave scams and real leaks different phrases; a supervised text baseline
# then "detected" leaks just by learning which phrase list a message came from -- an artifact
# of the generator, not a real signal. Real scammers copy real brokers' language.)
LEAK_TALK = leak_text.BROKER_LINES + leak_text.SCAM_POSTS + [
    "paper aa gaya, DM", "centre se nikla hai, genuine", "payment ke baad personal pe bhejenge",
    "sirf trusted logon ko", "kal subah tak final set", "{fee} advance, baaki exam ke baad"]
LEAK_PREFIXES = ["confirmed: ", "same set - ", "", "tomorrow's paper: "]
REACTIONS = ["is this real?", "same qs aaye kya?", "thanks bro", "fake hai ye", "source?", "bhai aur bhejo",
             "kitne ka hai?", "don't share outside", "ye to last year ka hai", "ok", "legit?", "saved"]


class _Ctx:
    """Everything one session is built from: its population, groups, clock, and the event list."""

    def __init__(self, sid, rng, exam_paper, window_start, pool):
        self.sid, self.rng, self.exam_paper, self.window_start = sid, rng, exam_paper, window_start
        self.events, self.pool = [], pool
        self.people = {}      # sender_id -> (role, region)
        self.groups = {}      # group_id -> region

    # population ----------------------------------------------------------
    def add_people(self, role, n, regions):
        ids = []
        for i in range(n):
            # neutral ids: a real monitor sees account handles, not roles -- the role is kept only
            # as hidden ground truth on each event and is never used by any model
            pid = f"user{self.rng.randrange(10**6):06d}_{self.sid}"
            while pid in self.people:
                pid = f"user{self.rng.randrange(10**6):06d}_{self.sid}"
            self.people[pid] = (role, self.rng.choice(regions))
            ids.append(pid)
        return ids

    def add_groups(self, kind, n, regions):
        ids = []
        for i in range(n):
            region = self.rng.choice(regions)
            city = self.rng.choice(REGIONS[region])
            gid = f"{city}_group{self.rng.randrange(10**4):04d}_{self.sid}"   # neutral name (kind is not visible)
            self.groups[gid] = region
            ids.append(gid)
        return ids

    # clock ------------------------------------------------------------------
    def time(self, ramp=1.0, night=False):
        """Days-before-exam skewed towards the exam by `ramp`; hour from a diurnal profile."""
        u = self.rng.random() ** ramp
        day = int(u * WINDOW_DAYS)
        if night and self.rng.random() < 0.6:
            hour = self.rng.choice([21, 22, 23, 0, 1, 2])
        else:
            hour = self.rng.choices(range(24), weights=[1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 4, 5, 6, 7, 8, 8, 7, 5, 2])[0]
        t = self.window_start - (day + 1) * 86400 + hour * 3600 + self.rng.uniform(0, 3600)
        return min(t, self.window_start - 60)

    # events -------------------------------------------------------------------
    def post(self, t, sender, group, text=None, image=None, leak=False, render=None, qid=None):
        role, _ = self.people[sender]
        self.events.append({
            "timestamp": t, "sender_id": sender, "group_id": group, "region": self.groups[group], "role": role,
            "event_type": "image" if image else "text",
            "text": text if not image else (text or ""),
            "image_id": image["image_id"] if image else None,
            "image_kind": image["kind"] if image else None,
            "is_leak_event": bool(leak), "render": render, "qid": qid,
            # legacy fields, kept so the original modules still accept these sessions
            "content_overlap_fraction": None if image else (1.0 if leak else 0.0),
            "image_phash_hamming_to_true": image["scores"][self.exam_paper]["phash_hamming"] if image else None,
        })

    def cascade(self, t0, sender, group, forwarders, target_groups, mean_delay_s, p_forward, **content):
        """Post once, then let forwarders re-post the same item into other groups after random delays."""
        self.post(t0, sender, group, **content)
        for g in target_groups:
            if g == group or self.rng.random() > p_forward:
                continue
            t = min(t0 + self.rng.expovariate(1.0 / mean_delay_s), self.window_start - 30)
            self.post(t, self.rng.choice(forwarders), g, **content)

    def chatter(self, n, members, groups, ramp=1.0, broker_prob=0.0):
        for _ in range(n):
            text = leak_text.fill(self.rng.choice(leak_text.STUDY_CHATTER), self.rng)
            if self.rng.random() < broker_prob:
                text = self.rng.choice(["koi PW batch ka PDF bhej do, payment kar dunga", "full set notes chahiye, kitne ka?",
                                        "genuine test series konsi hai?", "advance mein fees di thi coaching ki, refund nahi mila"])
            self.post(self.time(ramp), self.rng.choice(members), self.rng.choice(groups), text=text)

    def reactions(self, n, members, groups, ramp):
        for _ in range(n):
            self.post(self.time(ramp), self.rng.choice(members), self.rng.choice(groups), text=self.rng.choice(REACTIONS))

    def image(self, kinds, paper=None):
        cands = [r for r in self.pool if r["kind"] in kinds and (paper is None or r.get("paper") == paper)]
        return self.rng.choice(cands)

    def render(self, q, styles):
        style = self.rng.choice(styles)
        return style, leak_text.RENDERERS[style](q, self.rng)


def _regions(rng, lo, hi):
    return rng.sample(list(REGIONS), rng.randint(lo, hi))


# ---------------------------------------------------------------------------
# NORMAL scenarios
# ---------------------------------------------------------------------------

def study_chatter(c):
    regs = _regions(c.rng, 1, 2 if c.rng.random() < 0.25 else 1)
    students = c.add_people("student", c.rng.randint(10, 40), regs)
    groups = c.add_groups("study", c.rng.randint(1, 3), regs)
    c.chatter(c.rng.randint(15, 70), students, groups, ramp=c.rng.uniform(1.0, 1.6), broker_prob=0.05)


def past_paper_discussion(c):
    regs = _regions(c.rng, 1, 2)
    students = c.add_people("student", c.rng.randint(10, 35), regs)
    groups = c.add_groups("study", c.rng.randint(1, 3), regs)
    ramp = c.rng.uniform(1.3, 2.2)                 # revision intensifies before the exam, like a leak does
    c.chatter(c.rng.randint(15, 50), students, groups, ramp=ramp)
    pps = c.rng.sample(past_papers(), c.rng.randint(1, 2))
    for _ in range(c.rng.randint(5, 20)):
        q = c.rng.choice(load_papers()[c.rng.choice(pps)])
        style, text = c.render(q, ["verbatim", "excerpt", "typed", "ocr", "paraphrase", "hinglish"])
        c.post(c.time(ramp), c.rng.choice(students), c.rng.choice(groups), text=text, render=f"past_{style}")
    for _ in range(c.rng.randint(0, 2)):          # "2019 P1 answer key" -- a past paper's key
        qs = c.rng.sample(load_papers()[c.rng.choice(pps)], 6)
        c.post(c.time(ramp), c.rng.choice(students), c.rng.choice(groups), text=leak_text.answer_key(qs, c.rng), render="past_answer_key")
    for _ in range(c.rng.randint(0, 4)):
        c.post(c.time(ramp), c.rng.choice(students), c.rng.choice(groups),
               text=c.rng.choice(leak_text.GENERIC_PHOTO_CAPTIONS).format(n=c.rng.randint(1, 9)),
               image=c.image(("past_photo",)), render="past_photo")
    c.reactions(c.rng.randint(2, 10), students, groups, ramp)


def coaching_broadcast(c):
    regs = _regions(c.rng, 2, 3)
    admin = c.add_people("coaching_admin", 1, regs)
    faculty = c.add_people("faculty", c.rng.randint(1, 3), regs)
    students = c.add_people("student", c.rng.randint(20, 60), regs)
    groups = c.add_groups("coaching", c.rng.randint(3, 6), regs)
    ramp = c.rng.uniform(1.5, 2.5)
    for _ in range(c.rng.randint(6, 18)):          # announcements fanned out to every batch group
        c.cascade(c.time(ramp), admin[0], groups[0], faculty + admin, groups, 7200, 0.8,
                  text=leak_text.fill(c.rng.choice(leak_text.COACHING_BROADCAST), c.rng), render="coaching_ad")
    for _ in range(c.rng.randint(3, 10)):          # "mock test" questions (really past-paper questions)
        q = c.rng.choice(load_papers()[c.rng.choice(past_papers())])
        style, text = c.render(q, ["verbatim", "typed", "paraphrase", "hinglish"])
        c.cascade(c.time(ramp), c.rng.choice(faculty), c.rng.choice(groups), faculty, groups, 3600, 0.5,
                  text=c.rng.choice(["Mock Q: ", "Expected Q: ", "", "PYQ: "]) + text, render=f"mock_{style}")
    for _ in range(c.rng.randint(2, 8)):
        c.cascade(c.time(ramp), c.rng.choice(faculty), c.rng.choice(groups), faculty, groups, 3600, 0.4,
                  text="mock sheet", image=c.image(("coaching_photo", "notes_photo")), render="coaching_photo")
    c.chatter(c.rng.randint(15, 60), students, groups, ramp=ramp, broker_prob=0.1)
    c.reactions(c.rng.randint(5, 20), students, groups, ramp)


def fake_leak_scam(c):
    regs = _regions(c.rng, 2, 4)
    brokers = c.add_people("broker", c.rng.randint(1, 3), regs)
    operators = c.add_people("operator", c.rng.randint(2, 8), regs)
    students = c.add_people("student", c.rng.randint(15, 50), regs)
    groups = c.add_groups("leakgroup", c.rng.randint(2, 5), regs)
    ramp = c.rng.uniform(2.0, 3.0)
    for _ in range(c.rng.randint(8, 25)):
        c.cascade(c.time(ramp, night=True), c.rng.choice(brokers), c.rng.choice(groups), operators, groups, 1200, 0.6,
                  text=leak_text.fill(c.rng.choice(LEAK_TALK), c.rng), render="scam_post")
    for _ in range(c.rng.randint(2, 8) if c.rng.random() < 0.7 else 0):   # "proof": an OLD paper passed off as this year's
        q = c.rng.choice(load_papers()[c.rng.choice(past_papers())])
        style, text = c.render(q, ["verbatim", "excerpt", "typed", "hinglish"])
        c.cascade(c.time(ramp, night=True), c.rng.choice(brokers + operators), c.rng.choice(groups), operators, groups,
                  1200, 0.5, text=c.rng.choice(LEAK_PREFIXES) + text, render=f"scam_proof_{style}")
    for _ in range(c.rng.randint(0, 5)):
        c.cascade(c.time(ramp, night=True), c.rng.choice(operators), c.rng.choice(groups), operators, groups, 1200, 0.5,
                  text="proof pic", image=c.image(("past_photo", "coaching_photo")), render="scam_photo")
    if c.rng.random() < 0.5:                       # a fabricated answer key for THIS exam (random answers)
        qs = c.rng.sample(load_papers()[c.exam_paper], 10)
        fake = [dict(q, gold=c.rng.choice(["A", "B", "C", "D", "AB", "ACD", str(c.rng.randint(0, 99))])) for q in qs]
        c.cascade(c.time(ramp, night=True), c.rng.choice(brokers), c.rng.choice(groups), operators, groups, 1200, 0.6,
                  text=leak_text.answer_key(fake, c.rng), render="scam_fake_key")
    c.reactions(c.rng.randint(10, 30), students, groups, ramp)
    c.chatter(c.rng.randint(5, 25), students, groups, ramp=1.2)


# ---------------------------------------------------------------------------
# LEAK scenarios
# ---------------------------------------------------------------------------

def _leak_network(c, n_groups, n_regions, chain=True):
    regs = _regions(c.rng, *n_regions)
    net = {"regs": regs}
    net["experts"] = c.add_people("subject_expert", c.rng.randint(1, 2), regs) if chain else []
    net["brokers"] = c.add_people("broker", c.rng.randint(1, 4), regs) if chain else []
    net["operators"] = c.add_people("operator", c.rng.randint(2, 8), regs) if chain else []
    net["students"] = c.add_people("student", c.rng.randint(10, 40), regs)
    net["groups"] = c.add_groups("leakgroup" if chain else "study", c.rng.randint(*n_groups), regs)
    return net


def _leak_items(c, net, questions, styles, ramp, fanout=0.6, prefix_broker=True):
    fwd = net["operators"] or net["students"]
    src_pool = net["brokers"] + net["experts"] or net["students"]
    for q in questions:
        style, text = c.render(q, styles)
        if prefix_broker and c.rng.random() < 0.3:
            text = c.rng.choice(LEAK_PREFIXES) + text
        c.cascade(c.time(ramp, night=True), c.rng.choice(src_pool), c.rng.choice(net["groups"]), fwd, net["groups"],
                  900, fanout, text=text, leak=True, render=style, qid=q["qid"])


def _leak_photos(c, net, n, ramp, fanout=0.6, captions=None):
    fwd = net["operators"] or net["students"]
    src_pool = net["brokers"] + net["experts"] or net["students"]
    for _ in range(n):
        img = c.image(("target_photo", "target_digital"), paper=c.exam_paper)
        cap = c.rng.choice(captions or leak_text.GENERIC_PHOTO_CAPTIONS + leak_text.BROKER_LINES).format(n=c.rng.randint(1, 9), fee=leak_text.fee(c.rng))
        c.cascade(c.time(ramp, night=True), c.rng.choice(src_pool), c.rng.choice(net["groups"]), fwd, net["groups"],
                  900, fanout, text=cap, image=img, leak=True, render=f"photo_{img['kind']}",
                  qid=(img["scores"][c.exam_paper]["qids"] or [None])[0])


def _broker_talk(c, net, n, ramp):
    for _ in range(n):
        c.cascade(c.time(ramp, night=True), c.rng.choice(net["brokers"]), c.rng.choice(net["groups"]), net["operators"],
                  net["groups"], 1200, 0.4, text=leak_text.fill(c.rng.choice(LEAK_TALK), c.rng), render="broker_line")


def full_leak(c):
    net = _leak_network(c, (2, 4), (2, 4))
    ramp = c.rng.uniform(2.0, 3.0)
    qs = c.rng.sample(load_papers()[c.exam_paper], max(3, int(len(load_papers()[c.exam_paper]) * c.rng.uniform(0.3, 0.8))))
    _leak_items(c, net, qs, ["verbatim", "verbatim", "excerpt", "ocr", "typed", "paraphrase", "hinglish"], ramp)
    _leak_photos(c, net, c.rng.randint(2, 10), ramp)
    _broker_talk(c, net, c.rng.randint(5, 15), ramp)
    c.reactions(c.rng.randint(10, 30), net["students"], net["groups"], ramp)
    c.chatter(c.rng.randint(10, 40), net["students"], net["groups"], ramp=1.3)


def partial_leak(c):
    net = _leak_network(c, (1, 2), (1, 2))
    ramp = c.rng.uniform(1.5, 3.0)
    qs = c.rng.sample(load_papers()[c.exam_paper], c.rng.randint(2, 6))
    _leak_items(c, net, qs, ["verbatim", "excerpt", "ocr", "typed", "paraphrase", "hinglish"], ramp, fanout=0.4)
    if c.rng.random() < 0.4:
        _leak_photos(c, net, c.rng.randint(1, 2), ramp, fanout=0.3)
    _broker_talk(c, net, c.rng.randint(0, 5), ramp)
    c.reactions(c.rng.randint(3, 12), net["students"], net["groups"], ramp)
    c.chatter(c.rng.randint(15, 50), net["students"], net["groups"], ramp=1.3)


def paraphrased_leak(c):
    net = _leak_network(c, (1, 3), (1, 3))
    ramp = c.rng.uniform(1.5, 3.0)
    qs = c.rng.sample(load_papers()[c.exam_paper], c.rng.randint(3, 10))
    _leak_items(c, net, qs, ["paraphrase", "hinglish"], ramp, fanout=0.4)
    _broker_talk(c, net, c.rng.randint(0, 6), ramp)
    c.reactions(c.rng.randint(3, 12), net["students"], net["groups"], ramp)
    c.chatter(c.rng.randint(15, 50), net["students"], net["groups"], ramp=1.3)


def answer_key_leak(c):
    net = _leak_network(c, (1, 3), (1, 3))
    ramp = c.rng.uniform(2.0, 3.0)
    paper = load_papers()[c.exam_paper]
    for _ in range(c.rng.randint(1, 3)):
        qs = sorted(c.rng.sample(paper, c.rng.randint(6, min(20, len(paper)))), key=lambda q: q["number"])
        noisy = [dict(q, gold=(c.rng.choice("ABCD") if c.rng.random() < 0.15 else q["gold"])) for q in qs]   # a few errors
        c.cascade(c.time(ramp, night=True), c.rng.choice(net["brokers"] + net["experts"]), c.rng.choice(net["groups"]),
                  net["operators"], net["groups"], 900, 0.6, text=leak_text.answer_key(noisy, c.rng), leak=True,
                  render="answer_key", qid=qs[0]["qid"])
    _broker_talk(c, net, c.rng.randint(2, 10), ramp)
    c.reactions(c.rng.randint(3, 15), net["students"], net["groups"], ramp)
    c.chatter(c.rng.randint(10, 40), net["students"], net["groups"], ramp=1.3)


def photo_leak(c):
    net = _leak_network(c, (1, 3), (1, 3))
    ramp = c.rng.uniform(2.0, 3.0)
    _leak_photos(c, net, c.rng.randint(2, 10), ramp, captions=leak_text.GENERIC_PHOTO_CAPTIONS)
    _broker_talk(c, net, c.rng.randint(0, 6), ramp)
    c.reactions(c.rng.randint(3, 15), net["students"], net["groups"], ramp)
    c.chatter(c.rng.randint(10, 40), net["students"], net["groups"], ramp=1.3)


def quiet_leak(c):
    """A small, careful insider group: one city, a handful of people, no broker language, flat timing."""
    net = _leak_network(c, (1, 1), (1, 1), chain=False)
    net["students"] = net["students"][: c.rng.randint(3, 6)]
    qs = c.rng.sample(load_papers()[c.exam_paper], c.rng.randint(1, 4))
    _leak_items(c, net, qs, ["typed", "excerpt", "verbatim", "hinglish"], ramp=1.0, fanout=0.0, prefix_broker=False)
    if c.rng.random() < 0.3:
        _leak_photos(c, net, 1, 1.0, fanout=0.0, captions=["", "see", "ye dekh"])
    c.chatter(c.rng.randint(6, 20), net["students"], net["groups"], ramp=1.0)


def buried_leak(c):
    """A big ordinary study group where one person drops 1-3 leaked items among normal chat."""
    regs = _regions(c.rng, 1, 2)
    students = c.add_people("student", c.rng.randint(20, 50), regs)
    groups = c.add_groups("study", c.rng.randint(1, 3), regs)
    c.chatter(c.rng.randint(60, 150), students, groups, ramp=c.rng.uniform(1.0, 1.6))
    net = {"students": students, "groups": groups, "brokers": [], "experts": [], "operators": []}
    n_items = c.rng.randint(1, 3)
    if c.rng.random() < 0.5:
        _leak_photos(c, net, 1, 1.5, fanout=0.0, captions=["", "see this", "kal ka?", "ye dekho"])
        n_items -= 1
    if n_items > 0:
        qs = c.rng.sample(load_papers()[c.exam_paper], n_items)
        _leak_items(c, net, qs, ["typed", "excerpt", "hinglish", "verbatim"], ramp=1.5, fanout=0.0, prefix_broker=False)
    c.reactions(c.rng.randint(2, 8), students, groups, 1.3)


def coordination_only_leak(c):
    net = _leak_network(c, (2, 4), (2, 4))
    ramp = c.rng.uniform(2.0, 3.0)
    # the real chain: expert -> broker -> operators -> students, with multi-hop forwarding
    for _ in range(c.rng.randint(8, 25)):
        src = c.rng.choice(net["experts"] + net["brokers"])
        c.cascade(c.time(ramp, night=True), src, c.rng.choice(net["groups"]), net["operators"], net["groups"], 900, 0.6,
                  text=leak_text.fill(c.rng.choice(LEAK_TALK), c.rng), leak=True, render="coordination")
    for _ in range(c.rng.randint(0, 4)):          # payment screenshots / unrelated pics -- NOT the paper
        c.cascade(c.time(ramp, night=True), c.rng.choice(net["brokers"]), c.rng.choice(net["groups"]), net["operators"],
                  net["groups"], 900, 0.4, text=c.rng.choice(["payment ss", "proof", "", "done"]),
                  image=c.image(("notes_photo", "coaching_photo")), leak=True, render="coordination_image")
    c.reactions(c.rng.randint(10, 30), net["students"], net["groups"], ramp)
    c.chatter(c.rng.randint(5, 25), net["students"], net["groups"], ramp=1.2)


SCENARIOS = {
    "study_chatter": study_chatter, "past_paper_discussion": past_paper_discussion,
    "coaching_broadcast": coaching_broadcast, "fake_leak_scam": fake_leak_scam,
    "full_leak": full_leak, "partial_leak": partial_leak, "paraphrased_leak": paraphrased_leak,
    "answer_key_leak": answer_key_leak, "photo_leak": photo_leak, "quiet_leak": quiet_leak, "buried_leak": buried_leak,
    "coordination_only_leak": coordination_only_leak,
}


def generate_session(sid: str, scenario: str, rng: random.Random, pool: list, exam_paper: str = None,
                     window_start: float = 1_800_000_000.0) -> dict:
    exam_paper = exam_paper or rng.choice(TARGET_PAPERS)
    c = _Ctx(sid, rng, exam_paper, window_start, pool)
    SCENARIOS[scenario](c)
    c.events.sort(key=lambda e: e["timestamp"])
    label = "leak" if scenario in LEAK_MIX else "normal"
    return {"session_id": sid, "label": label, "scenario": scenario, "exam_paper": exam_paper,
            "window_start": window_start, "events": c.events}


def generate_hard_dataset(n_sessions: int = 1200, leak_fraction: float = 0.25, seed: int = 0, pool: list = None) -> dict:
    """
    Class balance is deliberately NOT 50/50: leaks are the rare class in
    real monitoring, so the default is 25% leak sessions -- which is why
    PR-AUC, not accuracy, is the headline metric.
    """
    rng = random.Random(seed)
    pool = pool or build_image_pool()
    n_leak = int(round(n_sessions * leak_fraction))
    scen = rng.choices(list(NORMAL_MIX), weights=list(NORMAL_MIX.values()), k=n_sessions - n_leak)
    scen += rng.choices(list(LEAK_MIX), weights=list(LEAK_MIX.values()), k=n_leak)
    rng.shuffle(scen)
    sessions = [generate_session(f"s{i:05d}", s, random.Random(rng.randrange(1 << 30)), pool) for i, s in enumerate(scen)]
    return {"sessions": sessions, "pool": pool, "seed": seed}


if __name__ == "__main__":
    import collections
    ds = generate_hard_dataset(n_sessions=300, seed=1)
    ss = ds["sessions"]
    print(f"{len(ss)} sessions:", dict(collections.Counter(s['label'] for s in ss)))
    print("scenarios:", dict(collections.Counter(s['scenario'] for s in ss)))
    by = collections.defaultdict(list)
    for s in ss:
        by[s["scenario"]].append(s)
    print(f"\n{'scenario':24s} {'events':>7s} {'regions':>8s} {'images':>7s} {'leak_ev':>8s}")
    for k, v in by.items():
        mean = lambda f: sum(f(s) for s in v) / len(v)
        print(f"{k:24s} {mean(lambda s: len(s['events'])):7.1f} {mean(lambda s: len({e['region'] for e in s['events']})):8.2f} "
              f"{mean(lambda s: sum(e['event_type']=='image' for e in s['events'])):7.1f} "
              f"{mean(lambda s: sum(e['is_leak_event'] for e in s['events'])):8.1f}")
    assert all(s["events"] for s in ss)
    assert all(not any(e["is_leak_event"] for e in s["events"]) for s in ss if s["label"] == "normal")
    assert all(any(e["is_leak_event"] for e in s["events"]) for s in ss if s["label"] == "leak")
    print("\nrealistic_data self-test passed (labels consistent with event-level ground truth).")
