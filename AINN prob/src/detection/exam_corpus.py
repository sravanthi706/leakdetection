"""
Real exam-paper corpus -- replaces the old "[synthetic placeholder
question stem #i]" text with REAL published JEE-Advanced questions.

Source: JEEBench (Arora, Singh & Mausam, EMNLP 2023, MIT licence) --
515 questions from JEE Advanced 2016-2023, each tagged with the exact
paper it appeared in ("JEE Adv 2023 Paper 1", ...). Download it with
`python scripts/download_data.py`.

Why this matters for the ML: with placeholder text, "leak" messages and
"normal" messages had almost no vocabulary in common, so the content
matcher's job was trivially easy. Real papers are hard in exactly the
way the real problem is hard:
  * past papers (legitimately shared by students every day) use the SAME
    physics/chemistry/maths vocabulary as the paper being protected;
  * questions are LaTeX-heavy, and students retype them in shorthand,
    so exact string matching breaks.

How the corpus is used:
  TARGET_PAPERS -- the "upcoming exam" whose paper is committed to the
                   ledger. A leak session leaks one of these.
  every other paper -- a PAST paper. Sharing and discussing it is normal,
                   legitimate behaviour, and must NOT raise an alert.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JEEBENCH_JSON = ROOT / "data" / "external" / "jeebench" / "dataset.json"

TARGET_PAPERS = [
    "JEE Adv 2022 Paper 1",
    "JEE Adv 2022 Paper 2",
    "JEE Adv 2023 Paper 1",
    "JEE Adv 2023 Paper 2",
]

_GREEK = ["alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "theta", "lambda", "mu",
          "nu", "pi", "rho", "sigma", "tau", "phi", "varphi", "omega", "Omega", "Delta", "eta",
          "kappa", "chi", "psi", "xi", "zeta", "Gamma", "Lambda", "Sigma", "Phi", "Psi"]

_SYMBOLS = {
    r"\times": " x ", r"\cdot": "*", r"\cdots": "...", r"\ldots": "...", r"\infty": "infinity",
    r"\leq": "<=", r"\geq": ">=", r"\le": "<=", r"\ge": ">=", r"\neq": "!=", r"\ne": "!=",
    r"\in": " in ", r"\notin": " not in ", r"\cup": " U ", r"\cap": " n ", r"\subset": " subset ",
    r"\subseteq": " subset ", r"\rightarrow": "->", r"\to": "->", r"\Rightarrow": "=>",
    r"\approx": "~", r"\pm": "+-", r"\circ": "deg", r"\prime": "'", r"\partial": "d",
    r"\sum": "sum", r"\int": "integral", r"\lim": "lim", r"\log": "log", r"\ln": "ln",
    r"\sin": "sin", r"\cos": "cos", r"\tan": "tan", r"\cot": "cot", r"\sec": "sec", r"\csc": "cosec",
    r"\max": "max", r"\min": "min", r"\det": "det", r"\mid": "|", r"\quad": " ", r"\qquad": " ",
    r"\,": " ", r"\;": " ", r"\!": "", r"\%": "%", r"\{": "{", r"\}": "}", r"\mathrm": "",
    r"\mathbf": "", r"\mathbb": "", r"\text": "", r"\operatorname": "", r"\hat": "", r"\vec": "",
    r"\bar": "", r"\overline": "", r"\left": "", r"\right": "", r"\displaystyle": "", r"\limits": "",
}


def latex_to_plain(text: str) -> str:
    """
    Convert LaTeX math into the plain text a student would type or a
    PDF copy-paste would yield: $\\frac{a}{b}$ -> (a)/(b), \\sqrt{x} -> sqrt(x),
    \\theta -> theta, ^{2} -> ^2. Deterministic; no information added.
    """
    t = text.replace("$", " ")
    t = re.sub(r"\\(begin|end)\s*\{[^{}]*\}(\{[^{}]*\})?", " ", t)   # environments: cases, array ...
    t = re.sub(r"\\(textbf|mathbf|bf|textit|mathit|mathrm|mathbb|text|operatorname)\s*", "", t)
    # Resolve structures from the innermost braces outwards, so nested forms like
    # \frac{x^{2}}{6} or \sqrt{\frac{a}{b}} convert correctly.
    frac = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    for _ in range(8):
        before = t
        t = re.sub(r"\^\s*\{([^{}]*)\}", r"^(\1)", t)
        t = re.sub(r"_\s*\{([^{}]*)\}", r"_\1", t)
        t = frac.sub(r"(\1)/(\2)", t)
        t = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}", r"root\1(\2)", t)
        t = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", t)
        t = re.sub(r"\{([^{}]*)\}", r"\1", t) if t == before else t
        if t == before:
            break
    # single-token exponents look nicer without brackets: x^(2) -> x^2
    t = re.sub(r"\^\((-?[A-Za-z0-9.]+)\)", r"^\1", t)
    for g in sorted(_GREEK, key=len, reverse=True):
        t = re.sub(r"\\" + g + r"(?![A-Za-z])", g, t)
    for k in sorted(_SYMBOLS, key=len, reverse=True):
        t = t.replace(k, _SYMBOLS[k])
    t = re.sub(r"\\[A-Za-z]+", " ", t)          # any remaining command
    t = t.replace("{", "").replace("}", "").replace("\\", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def split_stem_and_options(text: str):
    """Returns (stem, {"A": ..., "B": ...}) -- options only if the question is MCQ-style."""
    parts = re.split(r"\(([A-D])\)", text)
    stem = parts[0].strip()
    options = {}
    for i in range(1, len(parts) - 1, 2):
        options[parts[i]] = parts[i + 1].strip()
    return stem, options


@lru_cache(maxsize=1)
def load_papers(path: str = str(JEEBENCH_JSON)) -> dict:
    """
    Returns {paper_name: [question, ...]} in the paper's original order.
    Each question: {qid, number, subject, type, raw, plain, stem_plain, options_plain, gold}.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found -- run `python scripts/download_data.py` from the project root first."
        )
    records = json.load(open(p, encoding="utf-8"))
    papers = {}
    for r in records:
        papers.setdefault(r["description"], []).append(r)
    out = {}
    for name, qs in papers.items():
        qs = sorted(qs, key=lambda r: int(r["index"]))
        items = []
        for number, r in enumerate(qs, start=1):
            stem, opts = split_stem_and_options(r["question"])
            items.append({
                "qid": f"{name}#Q{number}",
                "paper": name,
                "number": number,
                "subject": r["subject"],
                "type": r["type"],
                "raw": r["question"],
                "plain": latex_to_plain(r["question"]),
                "stem_plain": latex_to_plain(stem),
                "options_plain": {k: latex_to_plain(v) for k, v in opts.items()},
                "gold": str(r["gold"]),
            })
        out[name] = items
    return out


def past_papers() -> list:
    return [name for name in load_papers() if name not in TARGET_PAPERS]


if __name__ == "__main__":
    papers = load_papers()
    print(f"Loaded {sum(len(v) for v in papers.values())} real questions from {len(papers)} papers.")
    print("Target (protected) papers:", TARGET_PAPERS)
    print("Past papers (legitimately shared):", len(past_papers()))
    q = papers["JEE Adv 2023 Paper 1"][10]
    print("\nRaw LaTeX :", q["raw"][:160])
    print("Plain text:", q["plain"][:160])
    assert "\\" not in q["plain"] and "$" not in q["plain"]
    bad = [x["qid"] for v in papers.values() for x in v if "\\" in x["plain"]]
    print(f"\nQuestions with leftover LaTeX backslashes after conversion: {len(bad)}")
    assert not bad
    print("exam_corpus self-test passed.")
