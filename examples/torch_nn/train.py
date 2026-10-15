#!/usr/bin/env python3
"""Train the signal-vs-background network of examples/nn_fit with PyTorch and export it as text for the pipeline.

    python3 train.py [datasets/public/seed_000/mc/pool.npz] [--out model.json]

The same 10-feature multilayer perceptron as examples/nn_fit (two hidden layers of 32 and 16, ReLU), trained
with PyTorch instead of scikit-learn and written two ways: model.json in nn_fit's layout (the feature list,
the standardisation, the weight matrices), which fit.py, examples/nn_fit's unchanged, evaluates with numpy
alone; and model_state.json, the state_dict as JSON lists, which fit_torch.py loads back into the same
nn.Module and evaluates with torch at run time, the route for a model whose forward pass you would rather
not rewrite by hand. No binary model files in a submission either way. torch is imported inside main(), so
fit.py's `from train import features` needs numpy only. Training takes a few minutes on a laptop CPU:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hww as H  # noqa: E402

FEATURES = ["lep_pt_0", "lep_pt_1", "met_et", "dphi", "mll", "ptll", "mt", "jet_n", "deta", "same_flavour"]


def features(ev):
    """The feature matrix of a set of events, in the order of FEATURES."""
    k = H.kinematics(ev)
    cols = {"lep_pt_0": ev["lep_pt_0"], "lep_pt_1": ev["lep_pt_1"], "met_et": ev["met_et"], "dphi": k["dphi"],
            "mll": k["mll"], "ptll": k["ptll"], "mt": k["mt"], "jet_n": ev["jet_n"],
            "deta": np.abs(ev["lep_eta_0"].astype(float) - ev["lep_eta_1"].astype(float)),
            "same_flavour": (ev["lep_type_0"] == ev["lep_type_1"]).astype(float)}
    return np.column_stack([np.asarray(cols[f], float) for f in FEATURES])


def auc(y, p, w=None):
    """The area under the ROC curve, from the weighted true- and false-positive rates in order of score."""
    w = np.ones(len(p)) if w is None else np.asarray(w, float)
    order = np.argsort(-p)
    tp, fp = np.cumsum(w[order] * (y[order] == 1)), np.cumsum(w[order] * (y[order] == 0))
    tpr, fpr = np.concatenate([[0.0], tp / tp[-1]]), np.concatenate([[0.0], fp / fp[-1]])
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", nargs="?", default=str(HERE.parent.parent / "datasets" / "public" / "seed_000" / "mc" / "pool.npz"))
    ap.add_argument("--out", default=str(HERE / "model.json"))
    ap.add_argument("--n", type=int, default=120000, help="events per class to train on (default 120000)")
    ap.add_argument("--epochs", type=int, default=40, help="passes over the training events at most (default 40; stops early)")
    a = ap.parse_args()
    import torch
    torch.manual_seed(1)
    pool = H.load(a.pool)
    rng = np.random.default_rng(1)
    idx = np.concatenate([rng.choice(np.where(pool["label"] == c)[0], size=min(a.n, int((pool["label"] == c).sum())), replace=False) for c in (1, 0)])
    rng.shuffle(idx)
    X, y, w = features(H.select(pool, idx)), pool["label"][idx].astype(int), pool["weight"][idx]
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = ((X - mean) / std).astype(np.float32)
    n_tr = int(0.8 * len(y))
    Xt, yt = torch.from_numpy(Xs), torch.from_numpy(y.astype(np.float32))
    net = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 16), torch.nn.ReLU(),
                              torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(a.epochs):
        net.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, 256):
            j = perm[i:i + 256]
            opt.zero_grad()
            loss_fn(net(Xt[j])[:, 0], yt[j]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(loss_fn(net(Xt[n_tr:])[:, 0], yt[n_tr:]))
        print(f"epoch {epoch + 1:2d}: validation loss {val:.4f}")
        if val < best - 1e-4:
            best, best_state, bad = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:            # early stopping, as the scikit-learn example
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        p = torch.sigmoid(net(Xt[n_tr:])[:, 0]).numpy()
    print(f"trained on {n_tr} events; AUC on the held-out fifth: {auc(y[n_tr:], p):.3f} (weighted {auc(y[n_tr:], p, w[n_tr:]):.3f})")
    # torch keeps a layer's weight as (out, in); the pipeline computes x @ W + b, so W is its transpose
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    model = {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "activation": "relu",
             "layers": [{"W": m.weight.detach().numpy().T.tolist(), "b": m.bias.detach().numpy().tolist()} for m in linears]}
    Path(a.out).write_text(json.dumps(model))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e3:.0f} kB)")
    state = {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist(),
             "state_dict": {k: v.detach().numpy().tolist() for k, v in net.state_dict().items()}}
    out_state = Path(a.out).with_name("model_state.json")
    out_state.write_text(json.dumps(state))
    print(f"wrote {out_state} ({out_state.stat().st_size / 1e3:.0f} kB): the state_dict as JSON, for fit_torch.py")


if __name__ == "__main__":
    main()
