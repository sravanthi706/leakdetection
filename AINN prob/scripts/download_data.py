"""
Downloads the REAL, public datasets the upgraded ML layer is evaluated on.

Every file is pinned to an exact git commit and verified by SHA-256, so
results are reproducible and nobody can silently swap the data under you.
Nothing here is scraped from a live platform -- all three are published
research datasets, used for academic evaluation only.

  1. Twitter15 / Twitter16 rumour-propagation trees  (Ma et al., ACL 2017;
     pre-processed by Bian et al., BiGCN, AAAI 2020)
     -> REAL propagation cascades, the standard GNN rumour-detection
        benchmark. Used to prove the GNN learns real spreading structure.
  2. JEEBench (Arora et al., EMNLP 2023, MIT licence)
     -> 515 REAL JEE-Advanced questions, 2016-2023, grouped by paper.
        Used as the committed question papers instead of placeholders.
  3. Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005)
     -> 5,801 human-labelled paraphrase pairs. Standard benchmark for
        checking whether the content matcher survives rewording.

Usage (from the project root):
    python scripts/download_data.py
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"

BIGCN = "https://raw.githubusercontent.com/TianBian95/BiGCN/5779f2cf86a4649651f2466b43c73caa6ed7463e/data"
JEEB = "https://raw.githubusercontent.com/dair-iitd/jeebench/a9af8bf354887043bd280a617c8716ef54e7aaec"
MSRP = ("https://raw.githubusercontent.com/wasiahmad/paraphrase_identification/"
        "88b208d0d094ab945d2401d9393531be1fec0076/dataset/msr-paraphrase-corpus")

FILES = [
    # (url, local path relative to data/external, sha256)
    (f"{BIGCN}/Twitter15/Twitter15_label_All.txt", "twitter/Twitter15_label_All.txt",
     "df9a941df2260225017a58d089a783e64cf8384ef5747f242b9de3471c1d918f"),
    (f"{BIGCN}/Twitter16/Twitter16_label_All.txt", "twitter/Twitter16_label_All.txt",
     "808849df5f482592b773ca63a0780e3fc9a1f954cd017b776016556d6d299257"),
    # Twitter15 and Twitter16 share one tree file (identical bytes in the BiGCN repo)
    (f"{BIGCN}/Twitter15/data.TD_RvNN.vol_5000.txt", "twitter/data.TD_RvNN.vol_5000.txt",
     "a3ce2e8bc18721fb41180d47d2ce7117f7da1c9d641480d5c11f0254f512559d"),
    (f"{JEEB}/data.zip", "jeebench/data.zip",
     "9db3aa734a3e5b7e02cead16c17b6687ff363d8a4d1015395f04f28ace33a07a"),
    (f"{MSRP}/msr_paraphrase_train.txt", "msrp/msr_paraphrase_train.txt",
     "61a88818ec384d8465297dda1883b469f1b22d442ab90183d13e06233b180591"),
    (f"{MSRP}/msr_paraphrase_test.txt", "msrp/msr_paraphrase_test.txt",
     "0360d0d3427a0ae14882e2b3a799df0fab6cb5203d5e8ee481949283da62acb6"),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected_sha: str):
    if dest.exists() and _sha256(dest) == expected_sha:
        print(f"  ok (cached)  {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading  {dest.relative_to(ROOT)} ...", flush=True)
    urllib.request.urlretrieve(url, tmp)
    got = _sha256(tmp)
    if got != expected_sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {dest.name}: expected {expected_sha}, got {got}")
    tmp.replace(dest)
    print(f"  verified     {dest.relative_to(ROOT)}")


def main():
    print(f"Downloading pinned public datasets into {EXTERNAL.relative_to(ROOT)}/")
    for url, rel, sha in FILES:
        fetch(url, EXTERNAL / rel, sha)

    # JEEBench ships as a zip; only dataset.json is needed.
    dataset_json = EXTERNAL / "jeebench" / "dataset.json"
    if not dataset_json.exists():
        with zipfile.ZipFile(EXTERNAL / "jeebench" / "data.zip") as z:
            with z.open("data/dataset.json") as src, open(dataset_json, "wb") as dst:
                dst.write(src.read())
        print(f"  extracted    {dataset_json.relative_to(ROOT)}")
    print("All datasets present and checksum-verified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # clear message instead of a raw traceback for the common "no internet" case
        print(f"\nDownload failed: {e}\nCheck your internet connection and re-run; completed files are kept.")
        sys.exit(1)
