#!/usr/bin/env python3
"""Train the classifier offline and export it as model.json (text) for fit.py. numpy only.

    python3 train.py <pool.npz> [--out model.json] [--hidden 64,64] [--epochs 60] [--seed 1]

A multilayer perceptron with weighted binary cross-entropy and Adam, early stopping on a validation fifth. It trains
on the even-numbered pool events; the odd ones give the held-out AUC. Every signal event is shifted to a random mass in
120-130 GeV (hww.morph), so the score holds across the masses the fit scans. Each class's weights are clipped at their
99th percentile and the two classes carry equal total weight.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hww as H  # noqa: E402
from features import FEATURES, features  # noqa: E402


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def forward(Ws, bs, X):
    acts = [X]
    for i, (W, b) in enumerate(zip(Ws, bs)):
        z = acts[-1] @ W + b
        acts.append(np.maximum(z, 0.0) if i < len(Ws) - 1 else z)
    return acts


def shifted(sig, rng, lo=120.0, hi=130.0):
    """Each signal event at a random mass in [lo, hi]: lepton pT and missing ET scaled by m / 125."""
    k = rng.uniform(lo, hi, len(sig["weight"])) / H.M_REF
    out = dict(sig)
    for c in H.SCALED:
        out[c] = sig[c].astype(float) * k
    return out


def wauc(y, s, w):
    o = np.argsort(s, kind="mergesort")
    y, w = y[o], w[o]
    neg = np.cumsum(w * (1 - y))
    return float(np.sum(w * y * neg) / ((w * y).sum() * (w * (1 - y)).sum()))


def train(X, y, w, Xv, yv, wv, hidden, epochs, seed, batch=4096, lr=2e-3, l2=1e-5, patience=6):
    rng = np.random.default_rng(seed)
    sizes = [X.shape[1], *hidden, 1]
    Ws = [rng.normal(0.0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
    bs = [np.zeros(b) for b in sizes[1:]]
    m1 = [np.zeros_like(p) for p in Ws + bs]
    m2 = [np.zeros_like(p) for p in Ws + bs]
    best, t = (np.inf, None, -1), 0

    def vloss():
        p = np.clip(sigmoid(forward(Ws, bs, Xv)[-1][:, 0]), 1e-7, 1 - 1e-7)
        return float(-np.sum(wv * (yv * np.log(p) + (1 - yv) * np.log(1 - p))) / wv.sum())

    for ep in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            j = order[s:s + batch]
            acts = forward(Ws, bs, X[j])
            d = ((sigmoid(acts[-1][:, 0]) - y[j]) * w[j] / w[j].sum())[:, None]
            gW, gb = [None] * len(Ws), [None] * len(Ws)
            for i in range(len(Ws) - 1, -1, -1):
                gW[i], gb[i] = acts[i].T @ d + l2 * Ws[i], d.sum(0)
                if i:
                    d = (d @ Ws[i].T) * (acts[i] > 0)
            t += 1
            for k, (p, g) in enumerate(zip(Ws + bs, gW + gb)):
                m1[k] = 0.9 * m1[k] + 0.1 * g
                m2[k] = 0.999 * m2[k] + 0.001 * g * g
                p -= lr * (m1[k] / (1 - 0.9 ** t)) / (np.sqrt(m2[k] / (1 - 0.999 ** t)) + 1e-8)
        loss = vloss()
        if loss < best[0] - 1e-6:
            best = (loss, ([W.copy() for W in Ws], [b.copy() for b in bs]), ep)
        elif ep - best[2] >= patience:
            break
    return best[1][0], best[1][1], ep + 1, best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool")
    ap.add_argument("--out", default=str(HERE / "model.json"))
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-shift", action="store_true", help="train on the 125 GeV signal only")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    pool = H.load(a.pool)
    even = np.arange(len(pool["label"])) % 2 == 0
    tr_ev, held = H.select(pool, even), H.select(pool, ~even)
    sig, bkg = H.select(tr_ev, tr_ev["label"] == 1), H.select(tr_ev, tr_ev["label"] == 0)
    if not a.no_shift:
        sig = shifted(sig, rng)
    X = np.concatenate([features(sig), features(bkg)])
    y = np.concatenate([np.ones(len(sig["weight"])), np.zeros(len(bkg["weight"]))])
    w = np.concatenate([sig["weight"], bkg["weight"]]).astype(float)
    for c in (0, 1):
        s = y == c
        w[s] = np.minimum(w[s], np.quantile(w[s], 0.99))
        w[s] *= 0.5 / w[s].sum()
    w *= len(y)
    mean, std = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mean) / std
    perm = rng.permutation(len(y))
    va, tr = perm[:len(y) // 5], perm[len(y) // 5:]
    hidden = tuple(int(h) for h in a.hidden.split(","))
    Ws, bs, ne, loss = train(Xs[tr], y[tr], w[tr], Xs[va], y[va], w[va], hidden, a.epochs, a.seed)
    p = sigmoid(forward(Ws, bs, (features(held) - mean) / std)[-1][:, 0])
    print(f"{ne} epochs, validation loss {loss:.5f}; held-out weighted AUC {wauc(held['label'].astype(float), p, held['weight'].astype(float)):.4f}")
    model = {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "activation": "relu", "mass_shifted": not a.no_shift,
             "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in zip(Ws, bs)]}
    Path(a.out).write_text(json.dumps(model))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
