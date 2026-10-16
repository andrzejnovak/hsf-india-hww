#!/usr/bin/env python3
"""The organisers' reference pipeline: a binned profile-likelihood fit of the classifier score and the masses.

    python3 fit.py <dataset_dir> <output_dir>      (run.sh calls this; model.json from train.py sits beside it)

The network (model.json, trained offline by train.py on mass-shifted signal) scores every event; the model in each bin
of (score, m_T, m_ll) is

    lambda = mu * S_m + b * B ,

S_m the signal at mass m (the 125 GeV simulation shifted to m with hww.morph and scored again, every 1 GeV, linear in
between), B the background, mu the signal strength and b the background rate relative to nominal, held to it by a 10 %
Gaussian. The mass is profiled over 123-127 GeV, the range the rules give for every scored dataset. Templates come from
the whole simulation, the population the data are drawn from; a background bin the simulation leaves empty holds half
its smallest event, so one data event there cannot pass for a signal. mu >= 0. Its error bar is where -ln L, the mass
and b profiled, rises by 1/2 on either side, times SCALE_MU, the pull width measured on suites of our own (not the
sealed one). numpy and scipy only, deterministic.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hww as H  # noqa: E402
from features import features  # noqa: E402

AXES = 3                                          # 3: (score, m_T, m_ll); 2: (score, m_T)
M_LO, M_HI = 123.0, 127.0
M_TEMPLATES = np.arange(M_LO, M_HI + 1e-9, 1.0)
M_FINE = np.arange(M_LO, M_HI + 1e-9, 0.25)
SCORE_EDGES = np.array([0.0, 0.3, 0.5, 0.65, 0.75, 0.85, 0.92, 0.96, 1.0])
MT_EDGES = np.linspace(0.0, 250.0, 26)
MLL_EDGES = np.array([0.0, 15, 25, 35, 45, 55, 70, 90, 150])
B_PRIOR = 0.10
MU_MAX = 10.0
SCALE_MU = 1.0


def load_model(path):
    m = json.loads(Path(path).read_text())
    return {"mean": np.array(m["mean"]), "std": np.array(m["std"]),
            "layers": [(np.array(l["W"]), np.array(l["b"])) for l in m["layers"]]}


def score(model, ev):
    x = (features(ev) - model["mean"]) / model["std"]
    for i, (W, b) in enumerate(model["layers"]):
        x = x @ W + b
        if i < len(model["layers"]) - 1:
            x = np.maximum(x, 0.0)
    return 1.0 / (1.0 + np.exp(-np.clip(x[:, 0], -30.0, 30.0)))


def hist(model, ev, w=None):
    s, k = score(model, ev), H.kinematics(ev)
    if AXES == 3:
        h = np.histogramdd([s, k["mt"], k["mll"]], [SCORE_EDGES, MT_EDGES, MLL_EDGES], weights=w)[0]
    else:
        h = np.histogram2d(s, k["mt"], [SCORE_EDGES, MT_EDGES], weights=w)[0]
    return h.ravel()


def templates(pool, model):
    sig, bkg = H.select(pool, pool["label"] == 1), H.select(pool, pool["label"] == 0)
    S = np.stack([hist(model, H.morph(sig, m), sig["weight"]) for m in M_TEMPLATES])
    B = np.maximum(hist(model, bkg, bkg["weight"]), 0.5 * float(bkg["weight"].min()))
    return S, B


def at_masses(S, ms):
    """The signal templates at the masses ms, linearly between the two neighbouring templates."""
    j = np.clip(np.searchsorted(M_TEMPLATES, ms, side="right") - 1, 0, len(M_TEMPLATES) - 2)
    f = (ms - M_TEMPLATES[j]) / (M_TEMPLATES[j + 1] - M_TEMPLATES[j])
    return (1 - f)[:, None] * S[j] + f[:, None] * S[j + 1]


def prof_b(sm, B, n, iters=15):
    """-ln L with b profiled (Newton, convex in b), for every signal template in the leading axes of sm."""
    ip = 1.0 / B_PRIOR ** 2
    b = np.ones(sm.shape[:-1])
    for _ in range(iters):
        lam = sm + b[..., None] * B
        g = B.sum() - (n * B / lam).sum(-1) + (b - 1.0) * ip
        h = (n * B * B / lam ** 2).sum(-1) + ip
        b = np.clip(b - g / h, 0.3, 3.0)
    lam = sm + b[..., None] * B
    return (lam - n * np.log(lam)).sum(-1) + 0.5 * (b - 1.0) ** 2 * ip, b


def surface(Sf, B, n, mus):
    """-ln L on a (mu, mass) grid, b profiled."""
    out = np.empty((len(mus), len(Sf)))
    for i in range(0, len(mus), 40):                  # in slices: memory
        out[i:i + 40] = prof_b(mus[i:i + 40, None, None] * Sf[None], B, n)[0]
    return out


def crossing(x, y, y0, side):
    """Where the curve y(x) first reaches y0 walking from its minimum to the given side; None past the grid."""
    i0 = int(np.argmin(y))
    rng = range(i0, len(y) - 1) if side > 0 else range(i0, 0, -1)
    for i in rng:
        j = i + side
        if y[j] >= y0 > y[i]:
            return x[i] + (y0 - y[i]) / (y[j] - y[i]) * (x[j] - x[i])
    return None


def main():
    ds, out = Path(sys.argv[1]), Path(sys.argv[2])
    model = load_model(HERE / "model.json")
    data, pool = H.load(ds / "data" / "events.npz"), H.load(ds / "mc" / "pool.npz")
    n = hist(model, data)
    S, B = templates(pool, model)
    Sf = at_masses(S, M_FINE)

    coarse = np.arange(0.0, MU_MAX + 1e-9, 0.05)
    f = surface(Sf, B, n, coarse)
    mu0 = coarse[np.unravel_index(np.argmin(f), f.shape)[0]]
    lo, hi = max(0.0, mu0 - 0.6), mu0 + 0.6
    while True:                                       # widen until both 1/2 crossings (or mu = 0) are inside
        mus = np.arange(lo, hi + 1e-9, 0.004)
        f = surface(Sf, B, n, mus)
        fp = f.min(1)
        up, dn = crossing(mus, fp, fp.min() + 0.5, +1), crossing(mus, fp, fp.min() + 0.5, -1)
        if (up is not None and (dn is not None or lo == 0.0)) or hi > 3 * MU_MAX:
            break
        lo, hi = max(0.0, lo - 0.6), hi + 0.6
    i = int(np.argmin(fp))
    if 0 < i < len(mus) - 1:                          # the minimum between grid points: a parabola
        a, b_, c = fp[i - 1], fp[i], fp[i + 1]
        den = a - 2 * b_ + c
        mu_hat = mus[i] + (0.5 * (a - c) / den * (mus[1] - mus[0]) if den > 0 else 0.0)
    else:
        mu_hat = mus[i]
    f_min = fp.min()
    mu_err = (0.5 * (up - dn) if dn is not None else up - mu_hat) * SCALE_MU

    m_hat = float(M_FINE[int(np.argmin(f.min(0)))])   # the mass, mu profiled: printed, not reported (optional, held to the window)

    f0 = float(prof_b(0.0 * Sf[:1], B, n)[0][0])
    z = float(np.sqrt(max(0.0, 2.0 * (f0 - f_min)))) if mu_hat > 0 else 0.0
    result = {"pois": [{"name": "mu", "value": float(mu_hat), "stat": float(mu_err), "syst": 0.0, "total": float(mu_err)}],
              "significance_obs": z, "discovery": bool(z >= 5.0),
              "note": f"profile-likelihood fit of (score, m_T{', m_ll' if AXES == 3 else ''}), mass profiled over {M_LO:g}-{M_HI:g} GeV, "
                      f"background rate held to 10 %; error bars x {SCALE_MU:g}"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1))
    print(f"mu = {mu_hat:.3f} +- {mu_err:.3f}  (best mass on the grid {m_hat:.2f})  Z = {z:.2f}")


if __name__ == "__main__":
    main()
