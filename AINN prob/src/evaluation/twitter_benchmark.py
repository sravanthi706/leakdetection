"""
REAL-DATA benchmark for the propagation-graph idea: Twitter15 / Twitter16.

The question this answers for the viva: "Your GNN was only ever trained
on data you generated. Does learning from propagation structure work on
REAL social-media cascades at all?"

Data: real Twitter rumour cascades (Ma et al., ACL 2017), as pre-processed
and released with BiGCN (Bian et al., AAAI 2020). Each sample is one
source tweet plus its whole retweet/reply tree; nodes carry a 5000-word
bag-of-words; label is one of {non-rumor, false, true, unverified}.
A leak spreading through Telegram/WhatsApp groups is the same kind of
object -- an item cascading through a network -- which is why this is
the standard, closest public stand-in.

Models compared (all trained here, same folds, same protocol):
  root_text_lr      logistic regression on the SOURCE tweet's words only
                    (no propagation at all)
  cascade_text_lr   logistic regression on the words of the whole cascade,
                    pooled (content of the spread, but no structure)
  structure_gnn     GraphSAGE on STRUCTURE ONLY -- depth, fan-out, subtree
                    size, position -- no words at all. This is the analogue
                    of AINN's GNN branch, which is deliberately text-blind.
  gcn               2-layer GCN over the tree with word features
  bigcn             our re-implementation of BiGCN (top-down + bottom-up
                    GCNs with root-feature enhancement and DropEdge)

Protocol: stratified 5-fold cross-validation; early stopping on a
validation slice carved out of the TRAINING fold. (The public BiGCN code
early-stops on the test fold itself, which inflates its reported numbers
-- so a proper-protocol re-implementation scoring somewhat lower than
the paper is expected, not a bug.) Word features: the full 5000-word
TF-IDF vector of every tweet (IDF fitted on the training fold only),
fed through a learned sparse embedding layer -- mathematically the same
as BiGCN's first dense layer, but it runs on a laptop CPU because it
never materialises the 5000-wide dense matrix.

Run:  python -m src.evaluation.twitter_benchmark            (both datasets)
      python -m src.evaluation.twitter_benchmark --quick    (Twitter16, 2 folds)
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.utils import dropout_edge

ROOT = Path(__file__).resolve().parents[2]
TW = ROOT / "data" / "external" / "twitter"
OUT = ROOT / "reports" / "results"
CLASSES = ["non-rumor", "false", "true", "unverified"]
VOCAB = 5000
EMB = 64


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(name: str):
    rows = defaultdict(list)
    for line in open(TW / "data.TD_RvNN.vol_5000.txt", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        rows[p[0]].append(p)
    samples = []
    for line in open(TW / f"{name}_label_All.txt", encoding="utf-8"):
        lab, _, eid = line.strip().split("\t")[:3]
        tree = rows.get(eid, [])
        if len(tree) < 2:                          # same filter as BiGCN: need at least one response
            continue
        index = {int(r[2]): i for i, r in enumerate(sorted(tree, key=lambda r: int(r[2])))}
        tree = sorted(tree, key=lambda r: int(r[2]))
        n = len(tree)
        bow = sp.lil_matrix((n, VOCAB), dtype=np.float32)
        parents = [-1] * n
        for r in tree:
            i = index[int(r[2])]
            if r[1] != "None":
                parents[i] = index[int(r[1])]
            if r[5]:
                for pair in r[5].split(" "):
                    w, c = pair.split(":")
                    if int(w) < VOCAB:
                        bow[i, int(w)] = float(c)
        root = parents.index(-1) if -1 in parents else 0
        samples.append({"eid": eid, "label": CLASSES.index(lab), "bow": bow.tocsr(), "parents": parents, "root": root})
    return samples


def structure_features(parents: list, root: int) -> np.ndarray:
    """Text-blind node features: depth, fan-out, subtree size, relative position, root/leaf flags."""
    n = len(parents)
    children = defaultdict(list)
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
    depth = [0] * n
    order = [root]
    for u in order:
        for v in children[u]:
            depth[v] = depth[u] + 1
            order.append(v)
    sub = [1] * n
    for u in reversed(order):
        for v in children[u]:
            sub[u] += sub[v]
    maxd = max(depth) or 1
    x = np.zeros((n, 7), np.float32)
    for i in range(n):
        x[i] = [depth[i] / maxd, min(len(children[i]), 50) / 50, np.log1p(sub[i]) / np.log1p(n),
                i / max(n - 1, 1), float(i == root), float(not children[i]), np.log1p(n) / 7]
    return x


def to_graphs(samples, tfidf):
    """Each graph keeps its words sparsely: (node, word, tf-idf weight) triples. `bow_node_index`
    contains 'index' in its name, so PyG shifts it by the node offset when graphs are batched."""
    graphs = []
    for s in samples:
        n = len(s["parents"])
        td = [(p, i) for i, p in enumerate(s["parents"]) if p >= 0]
        td = torch.tensor(td, dtype=torch.long).t().contiguous() if td else torch.zeros((2, 0), dtype=torch.long)
        W = tfidf.transform(s["bow"]).tocoo()
        graphs.append(Data(xs=torch.from_numpy(structure_features(s["parents"], s["root"])),
                           bow_node_index=torch.from_numpy(W.row.astype(np.int64)),
                           bow_word=torch.from_numpy(W.col.astype(np.int64)),
                           bow_val=torch.from_numpy(W.data.astype(np.float32)),
                           edge_index=td, y=torch.tensor([s["label"]]), root=torch.tensor([s["root"]]), num_nodes=n))
    return graphs


class SparseTextEmbedding(nn.Module):
    """x_node = sum_w tfidf(node, w) * E[w] -- a Linear(5000 -> d) applied to sparse input."""

    def __init__(self, vocab=VOCAB, d=EMB):
        super().__init__()
        self.E = nn.Parameter(torch.randn(vocab, d) * 0.05)
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, b):
        out = torch.zeros(b.num_nodes, self.E.size(1))
        out.index_add_(0, b.bow_node_index, self.E[b.bow_word] * b.bow_val[:, None])
        return out + self.b


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GCN(nn.Module):
    def __init__(self, hid=64, n_cls=4):
        super().__init__()
        self.emb = SparseTextEmbedding()
        self.c1, self.c2 = GCNConv(EMB, hid), GCNConv(hid, hid)
        self.out = nn.Linear(hid, n_cls)

    def forward(self, b):
        x = self.emb(b)
        ei = torch.cat([b.edge_index, b.edge_index.flip(0)], 1)          # undirected
        x = F.dropout(F.relu(self.c1(x, ei)), 0.5, self.training)
        x = F.relu(self.c2(x, ei))
        return self.out(global_mean_pool(x, b.batch))


class StructSAGE(nn.Module):
    """Structure-only GraphSAGE -- the Twitter analogue of AINN's text-blind GNN branch."""

    def __init__(self, d_in=7, hid=64, n_cls=4):
        super().__init__()
        self.c1, self.c2 = SAGEConv(d_in, hid), SAGEConv(hid, hid)
        self.out = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, n_cls))

    def forward(self, b):
        ei = torch.cat([b.edge_index, b.edge_index.flip(0)], 1)
        x = F.dropout(F.relu(self.c1(b.xs, ei)), 0.3, self.training)
        x = F.relu(self.c2(x, ei))
        return self.out(torch.cat([global_mean_pool(x, b.batch), global_max_pool(x, b.batch)], 1))


class _DirGCN(nn.Module):
    """One direction of BiGCN: GCN with root-feature enhancement after each layer."""

    def __init__(self, d_in, hid, out):
        super().__init__()
        self.c1, self.c2 = GCNConv(d_in, hid), GCNConv(hid + d_in, out)

    def forward(self, x, ei, batch, root_idx):
        x1 = x
        h = self.c1(x, ei)
        h2 = h
        h = torch.cat([h, x1[root_idx][batch]], 1)
        h = F.dropout(F.relu(h), 0.5, self.training)
        h = F.relu(self.c2(h, ei))
        h = torch.cat([h, h2[root_idx][batch]], 1)
        return global_mean_pool(h, batch)


class BiGCN(nn.Module):
    def __init__(self, hid=64, out=64, n_cls=4, drop_edge=0.2):
        super().__init__()
        self.emb = SparseTextEmbedding()
        self.td, self.bu = _DirGCN(EMB, hid, out), _DirGCN(EMB, hid, out)
        self.fc = nn.Linear(2 * (out + hid), n_cls)
        self.drop_edge = drop_edge

    def forward(self, b):
        x = self.emb(b)
        root_idx = b.ptr[:-1] + b.root                   # global index of each graph's root node
        ei = b.edge_index
        if self.training and self.drop_edge > 0:
            ei, _ = dropout_edge(ei, p=self.drop_edge)
        return self.fc(torch.cat([self.td(x, ei, b.batch, root_idx), self.bu(x, ei.flip(0), b.batch, root_idx)], 1))


def train_torch(model, tr, va, epochs=150, lr=5e-4, patience=20, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for b in DataLoader(tr, batch_size=64, shuffle=True):
            opt.zero_grad()
            F.cross_entropy(model(b), b.y).backward()
            opt.step()
        vloss = val_loss(model, va)
        if vloss < best - 1e-4:
            best, bad, best_state = vloss, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def val_loss(model, graphs):
    model.eval()
    tot = n = 0
    for b in DataLoader(graphs, batch_size=256):
        tot += F.cross_entropy(model(b), b.y, reduction="sum").item()
        n += b.num_graphs
    return tot / n


@torch.no_grad()
def predict_torch(model, graphs):
    model.eval()
    ys, ps = [], []
    for b in DataLoader(graphs, batch_size=256):
        ys.append(b.y.numpy())
        ps.append(model(b).argmax(1).numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run(name: str, folds: int = 5, seed: int = 0, models=None):
    models = models or ["root_text_lr", "cascade_text_lr", "structure_gnn", "gcn", "bigcn"]
    samples = load_dataset(name)
    y = np.array([s["label"] for s in samples])
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    res = defaultdict(lambda: {"acc": [], "macro_f1": [], "per_class_f1": []})
    for k, (tr_i, te_i) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        rng = np.random.default_rng(seed + k)
        tr_i = rng.permutation(tr_i)
        n_va = len(tr_i) // 9
        va_i, trn_i = tr_i[:n_va], tr_i[n_va:]
        # text weighting (IDF) fitted on the TRAINING fold only
        tfidf = TfidfTransformer(sublinear_tf=True).fit(sp.vstack([samples[i]["bow"] for i in trn_i]))
        G = {"tr": to_graphs([samples[i] for i in trn_i], tfidf), "va": to_graphs([samples[i] for i in va_i], tfidf),
             "te": to_graphs([samples[i] for i in te_i], tfidf)}
        for m in models:
            if m in ("root_text_lr", "cascade_text_lr"):
                def feat(idx):
                    if m == "root_text_lr":
                        return sp.vstack([tfidf.transform(samples[i]["bow"][samples[i]["root"]]) for i in idx])
                    return sp.vstack([sp.csr_matrix(tfidf.transform(samples[i]["bow"]).mean(axis=0)) for i in idx])
                clf = LogisticRegression(C=10.0, max_iter=3000).fit(feat(np.concatenate([trn_i, va_i])), y[np.concatenate([trn_i, va_i])])
                yt, yp = y[te_i], clf.predict(feat(te_i))
            else:
                net = {"structure_gnn": lambda: StructSAGE(), "gcn": lambda: GCN(), "bigcn": lambda: BiGCN()}[m]()
                train_torch(net, G["tr"], G["va"], seed=seed + k)
                yt, yp = predict_torch(net, G["te"])
            res[m]["acc"].append(accuracy_score(yt, yp))
            res[m]["macro_f1"].append(f1_score(yt, yp, average="macro"))
            res[m]["per_class_f1"].append(f1_score(yt, yp, average=None, labels=range(4)).tolist())
        print(f"  {name} fold {k + 1}/{folds} done in {time.time() - t0:.0f}s: " +
              ", ".join(f"{m}={res[m]['acc'][-1]:.3f}" for m in models), flush=True)
    summary = {}
    for m in models:
        pc = np.array(res[m]["per_class_f1"])
        summary[m] = {"acc_mean": float(np.mean(res[m]["acc"])), "acc_std": float(np.std(res[m]["acc"])),
                      "macro_f1_mean": float(np.mean(res[m]["macro_f1"])), "macro_f1_std": float(np.std(res[m]["macro_f1"])),
                      "per_class_f1_mean": dict(zip(CLASSES, pc.mean(axis=0).round(3).tolist()))}
    return {"dataset": name, "n_samples": len(samples), "folds": folds, "chance_acc": float(np.max(np.bincount(y)) / len(y)),
            "models": summary}


# Published reference points (different protocol -- see module docstring).
PUBLISHED = {
    "Twitter15": {"BiGCN (Bian et al., AAAI 2020, as reproduced in arXiv:2603.22854)": 0.844},
    "Twitter16": {"BiGCN (Bian et al., AAAI 2020, as reproduced in arXiv:2603.22854)": 0.880},
}


def to_markdown(results: list) -> str:
    lines = ["| Dataset | Model | Accuracy (mean ± std) | Macro-F1 (mean ± std) |", "|---|---|---|---|"]
    for r in results:
        for m, s in r["models"].items():
            lines.append(f"| {r['dataset']} | {m} | {s['acc_mean']:.3f} ± {s['acc_std']:.3f} | "
                         f"{s['macro_f1_mean']:.3f} ± {s['macro_f1_std']:.3f} |")
        lines.append(f"| {r['dataset']} | majority-class chance | {r['chance_acc']:.3f} | – |")
        for k, v in PUBLISHED[r["dataset"]].items():
            lines.append(f"| {r['dataset']} | {k} | {v:.3f} (published, different protocol) | – |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Twitter16 only, 2 folds")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    datasets, folds = (["Twitter16"], 2) if args.quick else (["Twitter15", "Twitter16"], 5)
    torch.set_num_threads(2)
    results = []
    for d in datasets:
        print(f"== {d} ==", flush=True)
        results.append(run(d, folds=folds))
    json.dump(results, open(OUT / "twitter_benchmark.json", "w"), indent=1)
    md = to_markdown(results)
    open(OUT / "twitter_benchmark.md", "w").write(md + "\n")
    print("\n" + md)
