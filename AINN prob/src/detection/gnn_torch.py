"""
PyTorch Geometric version of the propagation-graph GNN (optional backend).

The NumPy GNN in gnn.py stays as the dependency-free default (and as a
line-by-line teaching implementation). This module is the "real
tooling" version the original spec asked for, used when PyTorch +
PyTorch Geometric are installed:

    pip install torch torch_geometric

Differences from the NumPy version, all still CONTENT-AGNOSTIC (the GNN
must never read message text or compare anything with the paper -- that
is the content branch's job; mixing them was the bug caught earlier):

  * 8 node features instead of 4: the original 4 structural ones, plus
    node type (account vs group), how close to the exam the node was
    active, share of its activity at night, and how many of its posts
    were FORWARDS -- the same item (identical text or image id) appearing
    in more than one group. Forwarding is pure propagation structure:
    it is detected by identity of the item, not by what it says.
  * GraphSAGE layers with mean + max pooling (max pooling lets one small
    hot sub-cascade stand out inside a big, calm graph -- the buried-leak
    case).
  * Mini-batch Adam training with class weighting and early stopping on
    a held-out slice of the training data.
  * Node-level explanations with PyG's GNNExplainer: which accounts and
    groups the model relied on.
"""

import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

IN_DIM = 8


def session_to_pyg(session: dict, label: int = 0) -> Data:
    events = session["events"]
    ws = session.get("window_start", max(e["timestamp"] for e in events) if events else 0)
    senders = sorted({e["sender_id"] for e in events})
    groups = sorted({e["group_id"] for e in events})
    names = senders + groups
    idx = {n: i for i, n in enumerate(names)}
    N = max(len(names), 1)

    # identity of each posted item -> the groups it appeared in (forwarding = same item, several groups)
    item_groups = defaultdict(set)
    for e in events:
        key = e.get("image_id") or (e.get("text") or "").strip().lower()
        if key:
            item_groups[key].add(e["group_id"])

    adj = defaultdict(set)
    node_events = defaultdict(list)
    for e in events:
        s, g = idx[e["sender_id"]], idx[e["group_id"]]
        adj[s].add(g)
        adj[g].add(s)
        node_events[s].append(e)
        node_events[g].append(e)

    x = np.zeros((N, IN_DIM), np.float32)
    for n, i in idx.items():
        evs = node_events[i]
        deg = len(adj[i])
        x[i, 0] = min(deg, 20) / 20
        x[i, 1] = min(math.log1p(len(evs)) / 5, 1.0)
        x[i, 2] = min(len({e["region"] for e in evs}), 3) / 3
        x[i, 3] = min(np.mean([len(adj[j]) for j in adj[i]]) if adj[i] else 0, 20) / 20
        x[i, 4] = 1.0 if i >= len(senders) else 0.0
        if evs:
            hours_before = np.array([(ws - e["timestamp"]) / 3600 for e in evs])
            x[i, 5] = float(np.mean(hours_before < 48))
            hrs = np.array([int((e["timestamp"] / 3600) % 24) for e in evs])
            x[i, 6] = float(np.mean((hrs >= 21) | (hrs <= 3)))
            fw = [e for e in evs if len(item_groups[e.get("image_id") or (e.get("text") or "").strip().lower()]) > 1]
            x[i, 7] = min(len(fw), 10) / 10
    edges = [(a, b) for a in adj for b in adj[a]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=torch.from_numpy(x), edge_index=edge_index, y=torch.tensor([float(label)]), num_nodes=N,
                names=names)


class _SAGENet(nn.Module):
    def __init__(self, hidden=32, layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(IN_DIM if l == 0 else hidden, hidden) for l in range(layers)])
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, self.training)
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.head(g).squeeze(-1)


class TorchCascadeGNN:
    def __init__(self, hidden=32, layers=2, epochs=60, lr=5e-3, batch_size=32, patience=10, seed=0):
        self.hidden, self.layers, self.epochs, self.lr = hidden, layers, epochs, lr
        self.batch_size, self.patience, self.seed = batch_size, patience, seed
        self.net = None

    def fit(self, sessions, labels):
        torch.manual_seed(self.seed)
        rng = random.Random(self.seed)
        data = [session_to_pyg(s, int(y)) for s, y in zip(sessions, labels)]
        idx = list(range(len(data)))
        rng.shuffle(idx)
        n_val = max(1, len(idx) // 8)
        val = [data[i] for i in idx[:n_val]]
        tr = [data[i] for i in idx[n_val:]]
        pos = sum(int(d.y.item()) for d in tr)
        pos_weight = torch.tensor([(len(tr) - pos) / max(pos, 1)])
        self.net = _SAGENet(self.hidden, self.layers)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(self.epochs):
            self.net.train()
            for batch in DataLoader(tr, batch_size=self.batch_size, shuffle=True):
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(self.net(batch.x, batch.edge_index, batch.batch), batch.y,
                                                          pos_weight=pos_weight)
                loss.backward()
                opt.step()
            vloss = self._loss(val, pos_weight)
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _loss(self, data, pos_weight):
        self.net.eval()
        tot, n = 0.0, 0
        for batch in DataLoader(data, batch_size=128):
            tot += F.binary_cross_entropy_with_logits(self.net(batch.x, batch.edge_index, batch.batch), batch.y,
                                                      pos_weight=pos_weight, reduction="sum").item()
            n += batch.num_graphs
        return tot / max(n, 1)

    @torch.no_grad()
    def predict_proba(self, sessions) -> np.ndarray:
        self.net.eval()
        data = [session_to_pyg(s) for s in sessions]
        out = []
        for batch in DataLoader(data, batch_size=128):
            out.append(torch.sigmoid(self.net(batch.x, batch.edge_index, batch.batch)).numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def explain_nodes(self, session: dict, top_k: int = 5) -> list:
        """GNNExplainer node-importance mask: which accounts/groups the graph score relied on."""
        from torch_geometric.explain import Explainer, GNNExplainer
        d = session_to_pyg(session)
        self.net.eval()
        explainer = Explainer(model=self.net, algorithm=GNNExplainer(epochs=100),
                              explanation_type="model", node_mask_type="object", edge_mask_type=None,
                              model_config=dict(mode="binary_classification", task_level="graph", return_type="raw"))
        exp = explainer(d.x, d.edge_index)
        imp = exp.node_mask.view(-1).detach().numpy()
        order = np.argsort(-imp)[:top_k]
        return [{"node": d.names[i], "importance": float(imp[i])} for i in order]


if __name__ == "__main__":
    from src.detection.realistic_data import generate_hard_dataset
    from sklearn.metrics import roc_auc_score
    ds = generate_hard_dataset(n_sessions=400, seed=5)
    ss = ds["sessions"]
    y = np.array([s["label"] == "leak" for s in ss], dtype=int)
    tr, te = ss[:300], ss[300:]
    m = TorchCascadeGNN(seed=0).fit(tr, y[:300])
    p = m.predict_proba(te)
    print(f"Torch GNN (structure only) held-out ROC-AUC on hard sessions: {roc_auc_score(y[300:], p):.3f}")
    leak = next(s for s in te if s["label"] == "leak")
    print("GNNExplainer top nodes for one leak session:", m.explain_nodes(leak, 3))
