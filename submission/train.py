#!/usr/bin/env python3
"""The tutorial's network, trained on the pool and exported as text for the pipeline.

    python3 train.py [datasets/public/seed_000/mc/pool.npz] [--out model.json]

This is the classifier of the ML tutorial, transcribed: the same six raw columns (the missing ET and its
direction, the two lepton pTs and their directions), the same three hidden layers (12, 60, 32, relu) and a
sigmoid output, trained unweighted on the labels. The only changes are the library (scikit-learn, so the
weights come out as plain arrays) and standardised inputs. It is the starting point, not the goal: every row
of GUIDE.md's table (features, weights, binning) is a way past it. fit.py evaluates model.json with numpy
alone, as the contract requires.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hww as H  # noqa: E402

FEATURES = ["met_et", "met_phi", "lep_pt_0", "lep_pt_1", "lep_phi_0", "lep_phi_1"]


def features(ev):
    """The feature matrix of a set of events, in the order of FEATURES: the raw columns, nothing derived."""
    return np.column_stack([np.asarray(ev[f], float) for f in FEATURES])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", nargs="?", default=str(HERE.parent.parent / "datasets" / "public" / "seed_000" / "mc" / "pool.npz"))
    ap.add_argument("--out", default=str(HERE / "model.json"))
    ap.add_argument("--n", type=int, default=120000, help="events per class to train on (default 120000)")
    a = ap.parse_args()
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    pool = H.load(a.pool)
    rng = np.random.default_rng(1)
    idx = np.concatenate([rng.choice(np.where(pool["label"] == c)[0], size=min(a.n, int((pool["label"] == c).sum())), replace=False) for c in (1, 0)])
    rng.shuffle(idx)
    X, y, w = features(H.select(pool, idx)), pool["label"][idx].astype(int), pool["weight"][idx]
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = (X - mean) / std
    n_tr = int(0.8 * len(y))
    clf = MLPClassifier(hidden_layer_sizes=(12, 60, 32), activation="relu", max_iter=60, early_stopping=True, random_state=1)
    clf.fit(Xs[:n_tr], y[:n_tr])
    p = clf.predict_proba(Xs[n_tr:])[:, 1]
    print(f"trained on {n_tr} events; AUC on the held-out fifth: {roc_auc_score(y[n_tr:], p):.3f} (weighted {roc_auc_score(y[n_tr:], p, sample_weight=w[n_tr:]):.3f})")
    model = {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "activation": "relu",
             "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in zip(clf.coefs_, clf.intercepts_)]}
    Path(a.out).write_text(json.dumps(model))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
