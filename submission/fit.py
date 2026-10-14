#!/usr/bin/env python3
"""The tutorial network plugged into the measurement: a template fit of its score distribution.

    python3 fit.py <dataset_dir> <output_dir>      (run.sh calls this; model.json from train.py sits beside it)

The network (model.json) is evaluated with numpy on the data and on the pool; the score, "how Higgs-like", is
the fitted observable: 20 bins from 0 to 1, the signal and background shapes from the pool, two parameters, mu
and the background normalisation. Identical to examples/nn_fit/fit.py: only train.py (the features, the network) differs.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hww as H  # noqa: E402
from train import features  # noqa: E402

EDGES = np.linspace(0.0, 1.0, 21)


def load_model(path):
    m = json.loads(Path(path).read_text())
    return {"mean": np.array(m["mean"]), "std": np.array(m["std"]), "layers": [(np.array(l["W"]), np.array(l["b"])) for l in m["layers"]]}


def score(model, ev):
    """The network's output for every event: relu hidden layers, a sigmoid output."""
    x = (features(ev) - model["mean"]) / model["std"]
    for i, (W, b) in enumerate(model["layers"]):
        x = x @ W + b
        if i < len(model["layers"]) - 1:
            x = np.maximum(x, 0.0)
    return 1.0 / (1.0 + np.exp(-x[:, 0]))


def histogram(model, ev, weights=None):
    return np.histogram(score(model, ev), EDGES, weights=weights)[0]


def fit(n, s, b_nom):
    """Return (mu, b, err_mu, err_b, Z) for counts n against the templates s (mu = 1) and b_nom (b = 1)."""
    def nll(mu, b):
        lam = np.clip(mu * s + b * b_nom, 1e-9, None)
        return float(np.sum(lam - n * np.log(lam)))
    r = minimize(lambda p: nll(*p), [1.0, 1.0], method="L-BFGS-B", bounds=[(-2.0, 20.0), (0.2, 5.0)])
    mu_hat, b_hat = map(float, r.x)
    h = np.array([0.02, 0.01]); x = np.array([mu_hat, b_hat]); hess = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h[i], np.eye(2)[j] * h[j]
            hess[i, j] = (nll(*(x + ei + ej)) - nll(*(x + ei - ej)) - nll(*(x - ei + ej)) + nll(*(x - ei - ej))) / (4 * h[i] * h[j])
    try:
        err = np.sqrt(np.diag(np.linalg.inv(hess)))
        if not np.all(np.isfinite(err)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        err = np.array([1.0, 0.1])
    r0 = minimize(lambda p: nll(0.0, p[0]), [b_hat], method="L-BFGS-B", bounds=[(0.2, 5.0)])
    z = float(np.sqrt(max(0.0, 2.0 * (r0.fun - r.fun)))) if mu_hat > 0 else 0.0
    return mu_hat, b_hat, float(max(err[0], 1e-3)), float(err[1]), z


def main(ds: Path, out: Path):
    model = load_model(HERE / "model.json")
    data = H.load(ds / "data" / "events.npz")
    pool = H.load(ds / "mc" / "pool.npz")
    sig, bkg = H.select(pool, pool["label"] == 1), H.select(pool, pool["label"] == 0)
    n = histogram(model, data)
    s, b_nom = histogram(model, sig, sig["weight"]), histogram(model, bkg, bkg["weight"])
    mu, b, mu_err, b_err, z = fit(n, s, b_nom)
    result = {
        "pois": [{"name": "mu", "value": mu, "stat": mu_err, "syst": 0.0, "total": mu_err}],
        "significance_obs": z,
        "discovery": bool(z >= 5.0),
        "note": f"template fit of the network score, 20 bins; background normalisation {b:.3f} ± {b_err:.3f}",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1))
    print(f"mu = {mu:.3f} ± {mu_err:.3f}  b = {b:.3f} ± {b_err:.3f}  Z = {z:.2f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
