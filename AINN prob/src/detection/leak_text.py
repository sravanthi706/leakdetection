"""
How exam questions actually look when they travel through chat groups.

A leaked paper almost never circulates as clean, identical text. The
real NEET-UG 2026 chargesheet reporting describes PDF copies, phone
photos, and questions retyped by brokers and students (see citations in
synthetic_data.py). So a REAL question from exam_corpus.py is turned
into a chat message in one of several ways, from easiest to hardest to
detect:

  verbatim     copy-pasted from a PDF (plain text, LaTeX stripped)
  excerpt      only the first part of the question
  ocr          text pulled out of a phone photo by OCR, with the typical
               OCR character confusions (l/1, O/0, rn/m, dropped spaces)
  typed        a student retyping it in shorthand
  paraphrase   same meaning, different words (hand-written evaluation
               set for JEE 2023, rule-based fallback for other papers)
  hinglish     Hindi-English code-mixed retype, often with the answer
  answer_key   no question text at all -- just "Q3-B, Q4-AC, Q11-281"

The same renderers are applied to PAST papers inside NORMAL sessions,
because students legitimately share and discuss old papers every day.
That is what makes content matching genuinely hard here: the detector
has to tell "this year's committed paper" apart from "an old paper on
the same syllabus in the same format".
"""

import json
import random
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARAPHRASE_JSON = ROOT / "data" / "annotations" / "leak_paraphrases_jee2023.json"

# ---------------------------------------------------------------------
# Hand-written paraphrases (evaluation resource)
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def authored_paraphrases() -> dict:
    """{qid: {"paraphrase_en": text, "student_hinglish": text}} -- empty if the file is missing."""
    if not PARAPHRASE_JSON.exists():
        return {}
    out = {}
    for item in json.load(open(PARAPHRASE_JSON, encoding="utf-8"))["items"]:
        out.setdefault(item["qid"], {})[item["style"]] = item["text"]
    return out


# ---------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------

def verbatim(q: dict, rng: random.Random) -> str:
    prefix = rng.choice(["", f"Q{q['number']}. ", f"{q['number']}) ", f"Q.{q['number']} "])
    return prefix + q["plain"]


def excerpt(q: dict, rng: random.Random) -> str:
    words = q["stem_plain"].split()
    k = min(len(words), rng.randint(10, 22))
    return f"Q{q['number']} " + " ".join(words[:k]) + " ..."


_OCR_SUBS = [("l", "1"), ("O", "0"), ("o", "0"), ("rn", "m"), ("e", "c"), ("i", "l"),
             ("S", "5"), ("B", "8"), ("h", "b"), ("t", "f"), ("a", "o"), ("n", "ri")]


def ocr_noise(text: str, rng: random.Random, rate: float = 0.06) -> str:
    """Character-level corruption with the confusion pairs OCR engines typically make on phone photos."""
    out = text
    n_edits = max(1, int(len(out) * rate))
    for _ in range(n_edits):
        op = rng.random()
        if op < 0.55:
            src, dst = rng.choice(_OCR_SUBS)
            idxs = [m.start() for m in re.finditer(re.escape(src), out)]
            if idxs:
                i = rng.choice(idxs)
                out = out[:i] + dst + out[i + len(src):]
        elif op < 0.8:
            spaces = [m.start() for m in re.finditer(" ", out)]
            if spaces:
                i = rng.choice(spaces)
                out = out[:i] + out[i + 1:]            # merged words
        else:
            if len(out) > 2:
                i = rng.randrange(len(out))
                out = out[:i] + out[i + 1:]            # dropped character
    return out


def ocr(q: dict, rng: random.Random) -> str:
    return ocr_noise(q["plain"], rng, rate=rng.uniform(0.03, 0.10))


_ABBREV = {"question": "ques", "which": "wch", "following": "foll", "statements": "stmts",
           "statement": "stmt", "correct": "crct", "number": "no", "function": "fn",
           "value": "val", "maximum": "max", "minimum": "min", "probability": "prob",
           "equation": "eqn", "respectively": "resp", "between": "btw", "because": "coz",
           "temperature": "temp", "velocity": "vel", "acceleration": "acc", "with": "w/"}
_STOP = {"the", "a", "an", "of", "be", "is", "are", "that", "let", "then", "such", "and"}


def typed(q: dict, rng: random.Random) -> str:
    """A student retyping the question quickly: lowercase, abbreviations, dropped function words, truncated."""
    words = q["stem_plain"].lower().split()
    out = []
    for w in words:
        if w in _STOP and rng.random() < 0.6:
            continue
        out.append(_ABBREV.get(w, w))
    k = min(len(out), rng.randint(12, 30))
    lead = rng.choice(["", "qs: ", f"q{q['number']} ", "this one - ", "ye wala: "])
    return lead + " ".join(out[:k])


_SYN = [("Let", "Suppose"), ("Consider", "Take"), ("find", "determine"), ("value", "number"),
        ("which of the following", "which of these"), ("is(are) true", "hold"),
        ("is(are)", "are"), ("function", "map"), ("real numbers", "reals"),
        ("the number of", "how many"), ("probability", "chance"), ("respectively", "in that order"),
        ("distinct", "different"), ("ratio", "proportion"), ("correct", "right")]


def rule_paraphrase(q: dict, rng: random.Random) -> str:
    """Fallback paraphrase for questions without a hand-written one: synonym swaps + clause reordering."""
    t = q["stem_plain"]
    for a, b in _SYN:
        if a in t and rng.random() < 0.8:
            t = t.replace(a, b)
    sents = [s.strip() for s in re.split(r"(?<=[.?])\s+", t) if s.strip()]
    if len(sents) > 2 and rng.random() < 0.5:
        # move the question sentence to the front ("Find X, given that ...")
        sents = [sents[-1]] + sents[:-1]
    return " ".join(sents)


def paraphrase(q: dict, rng: random.Random) -> str:
    auth = authored_paraphrases().get(q["qid"], {})
    return auth.get("paraphrase_en") or rule_paraphrase(q, rng)


def hinglish(q: dict, rng: random.Random) -> str:
    auth = authored_paraphrases().get(q["qid"], {})
    if "student_hinglish" in auth:
        return auth["student_hinglish"]
    return f"bhai ye wala {typed(q, rng)} ... ans {q['gold']} hai kya"


def answer_key(questions: list, rng: random.Random) -> str:
    """Answer-key leak: only (question number, answer) pairs -- no question text at all."""
    fmt = rng.choice(["Q{n}-{a}", "{n}. {a}", "Q{n}) {a}", "{n}-{a}"])
    sep = rng.choice([", ", " | ", "\n", "  "])
    body = sep.join(fmt.format(n=q["number"], a=q["gold"]) for q in questions)
    head = rng.choice(["", "ANS KEY ", "answers: ", "final key ", "key -> "])
    return head + body


RENDERERS = {
    "verbatim": verbatim,
    "excerpt": excerpt,
    "ocr": ocr,
    "typed": typed,
    "paraphrase": paraphrase,
    "hinglish": hinglish,
}


# ---------------------------------------------------------------------
# Non-question chatter
# ---------------------------------------------------------------------

STUDY_CHATTER = [
    "anyone have notes for chapter {n}?", "what time is the mock test tomorrow",
    "can someone explain rotational mechanics doubt from ex {n}", "good luck everyone for the exam",
    "revising organic today, named reactions are killing me", "is the syllabus same as last year",
    "does anyone have last year's cutoff", "kal ka test kitne baje hai?",
    "integration by parts wala doubt clear kar do koi", "physics mein modern physics weightage kitna hai",
    "i am getting 180 in mocks, enough for top 5k?", "how to revise inorganic in 3 days",
    "anyone solving cengage or black book for maths", "admit card download ho gaya sabka?",
    "exam centre kaha aaya sabka", "please share formula sheet for electrostatics",
    "mera thermodynamics weak hai kya karu", "sleep schedule fix karo guys, exam morning shift hai",
    "which calculator is allowed? none right", "coordinate geometry practice set {n} done",
    "chemistry ncert line by line padh lo", "anyone from kota here?",
    "is the paper pattern changing this year?", "what is the marking scheme for multiple correct",
    "koi group study karega tonight?", "bro limits and continuity mein PYQs kaafi hai",
]

COACHING_BROADCAST = [
    "Mock Test {n} PDF uploaded -- full syllabus, solutions tonight",
    "Last 3 days crash course, pay {fee} for full set of notes + tests",
    "Final revision series live at 7 pm, link in bio",
    "All India Test Series rank list out, check your rank",
    "Join our premium batch, payment link below, only {fee}",
    "Formula booklet PDF (physics + chem + maths) free for first 500",
    "Expected questions for JEE Advanced based on PYQ trends -- PDF",
    "Doubt session tomorrow 10 am, bring your mock test sheets",
]

SCAM_POSTS = [
    "JEE ADV PAPER AVAILABLE 100% GENUINE, pay {fee} advance, DM fast",
    "confirmed paper tomorrow night, limited slots, payment first",
    "full set with answer key, genuine source inside centre, {fee} only",
    "last year also we gave paper, proof below, DM for deal",
    "PDF + answer key, pay via UPI, no refunds, genuine",
    "paper leak group link -- joining fee {fee}",
]

BROKER_LINES = [
    "confirmed, same set", "send the rest after payment", "full set coming tonight",
    "don't forward outside this group", "delete after reading", "advance received, sending",
    "genuine hai, centre se aaya hai", "{fee} final, no bargain", "answer key bhi milega",
    "check pics", "pics bhej raha hu", "same paper hai confirm",
]

GENERIC_PHOTO_CAPTIONS = ["", "see this", "pic", "page {n}", "check", "ye dekho", "notes"]


def fee(rng: random.Random) -> str:
    return rng.choice(["Rs 499", "Rs 999", "Rs 5000", "Rs 15,000", "Rs 2 lakh", "₹5000", "₹25k", "₹499"])


def fill(template: str, rng: random.Random) -> str:
    return template.format(n=rng.randint(1, 30), fee=fee(rng))


if __name__ == "__main__":
    from src.detection.exam_corpus import load_papers
    rng = random.Random(0)
    q = load_papers()["JEE Adv 2023 Paper 1"][10]
    print("REAL QUESTION:", q["plain"][:140], "\n")
    for name, fn in RENDERERS.items():
        print(f"{name:10s} ->", fn(q, rng)[:140])
    print(f"{'answer_key':10s} ->", answer_key(load_papers()["JEE Adv 2023 Paper 1"][9:13], rng))
    print(f"\nHand-written paraphrases loaded for {len(authored_paraphrases())} real questions.")
