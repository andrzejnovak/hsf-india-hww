#!/usr/bin/env python3
"""Make a suite of H -> WW pseudo-datasets, the truth beside each, from the MC pool.

    python3 make_suite.py suites/mine --n 10 --seed 7 [--pool datasets/public/seed_000/mc/pool.npz] [--design JSON]

seed_000 is at the public truth (m_H = 125 GeV, mu = 1, nominal background); the others draw mu in [0, 3] with a
fifth at 0 and the background normalisation within 10 % of nominal, and m_H in [100, 150], except that the first
half of them keep it within 2 GeV of 125 (truth.json: "mass_fixed"): those, with seed_000, are the datasets the
leaderboard's main table scores mu on; the table with the mass folded under it scores every dataset. Each dataset holds
data/events.npz (the pseudo-data: two leptons, missing ET, jets; no labels), mc/pool.npz and mc/pool_meta.json
(the pool, hard-linked), and truth.json. manifest.json lists the truths. --design fixes truths per sealed seed
(a JSON list of dicts; unset keys are drawn). The sealed suite is made the same way with a seed you don't have.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

import hww as H

PRIORS = {"mH_range": [100.0, 150.0], "mu_range": [0.0, 3.0], "null_fraction": 0.2, "bkg_norm_range": [0.9, 1.1],
          "fixed_mass_fraction": 0.5, "fixed_mass_range": [123.0, 127.0]}   # score_suite.MASS_FIXED_RANGE is the same window
PUBLIC_TRUTH = {"mH": 125.0, "mu": 1.0, "bkg_norm": 1.0, "mass_fixed": True}
RECIPE = 2   # bump it when the way a suite is drawn changes: the sealed run's result cache is keyed on it
DEFAULT_POOL = Path(__file__).resolve().parent / "datasets" / "public" / "seed_000" / "mc" / "pool.npz"


class Pool:
    """The MC pool, split into signal and background with sampling probabilities proportional to the weights."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        ev = H.load(self.path)
        self.meta = json.loads((self.path.parent / "pool_meta.json").read_text())
        self.sig = H.select(ev, ev["label"] == 1)
        self.bkg = H.select(ev, ev["label"] == 0)
        self.S_exp = float(self.sig["weight"].sum())
        self.B_exp = float(self.bkg["weight"].sum())
        self.sig_p = self.sig["weight"] / self.sig["weight"].sum()
        self.bkg_p = self.bkg["weight"] / self.bkg["weight"].sum()
        self._templates = None

    def templates(self):
        if self._templates is None:
            self._templates = H.templates({**self.sig, **{k: np.concatenate([self.sig[k], self.bkg[k]]) for k in self.sig}})
        return self._templates


def draw_truth(rng, priors, fixed=None, mass_fixed=False):
    """One dataset's truth; mass_fixed keeps m_H within priors["fixed_mass_range"] (a main-table dataset)."""
    f = fixed or {}
    mass_fixed = bool(f.get("mass_fixed", mass_fixed))
    t = {"mH": float(f.get("mH", rng.uniform(*(priors["fixed_mass_range"] if mass_fixed else priors["mH_range"]))))}
    if "mu" in f:
        t["mu"] = float(f["mu"])
    else:
        t["mu"] = 0.0 if rng.random() < priors["null_fraction"] else float(rng.uniform(*priors["mu_range"]))
    t["bkg_norm"] = float(f.get("bkg_norm", rng.uniform(*priors["bkg_norm_range"])))
    t["mass_fixed"] = mass_fixed
    return t


def generate(truth, rng, d: Path, pool: Pool):
    (d / "data").mkdir(parents=True, exist_ok=True)
    (d / "mc").mkdir(parents=True, exist_ok=True)
    n_sig = int(rng.poisson(truth["mu"] * pool.S_exp))
    n_bkg = int(rng.poisson(truth["bkg_norm"] * pool.B_exp))
    si = rng.choice(len(pool.sig_p), size=n_sig, p=pool.sig_p)
    bi = rng.choice(len(pool.bkg_p), size=n_bkg, p=pool.bkg_p)
    sig = H.morph(H.select(pool.sig, si), truth["mH"])
    bkg = H.select(pool.bkg, bi)
    perm = rng.permutation(n_sig + n_bkg)
    ev = {c: np.concatenate([sig[c], bkg[c]])[perm] for c in H.EVENT_COLS}
    np.savez_compressed(d / "data" / "events.npz", **ev)
    target = d / "mc" / "pool.npz"
    if target.exists():
        target.unlink()
    try:
        os.link(pool.path, target)
    except OSError:
        shutil.copy(pool.path, target)
    shutil.copy(pool.path.parent / "pool_meta.json", d / "mc" / "pool_meta.json")
    S, B = pool.templates()
    z = H.asimov_significance(truth["mu"] * H.interpolate(S, truth["mH"]), truth["bkg_norm"] * B)
    full = {**truth, "n_signal_realized": n_sig, "n_background_realized": n_bkg,
            "expected_significance": round(z, 3), "lumi_fb": pool.meta["lumi_fb"]}
    (d / "truth.json").write_text(json.dumps(full, indent=1))
    return full


def build(out: Path, n: int = 10, seed: int = 1, pool_path=DEFAULT_POOL, design=None, log=print):
    """Write seed_000 (the public truth) and n - 1 drawn datasets under out/, truth.json beside each and a
    manifest.json listing them; returns the list of truths (seed name included)."""
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; pick another name")
    pool = Pool(pool_path)
    n = len(design) + 1 if design else n
    n_fixed = round(PRIORS["fixed_mass_fraction"] * (n - 1))   # seed_001 .. seed_<n_fixed> keep the mass at 125
    truths = []
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        truth = dict(PUBLIC_TRUTH) if i == 0 else draw_truth(rng, PRIORS, design[i - 1] if design else None, mass_fixed=i <= n_fixed)
        d = out / f"seed_{i:03d}"
        full = generate(truth, rng, d, pool)
        truths.append({"seed": d.name, **full})
        tag = ("public" if i == 0 else f"Z_exp={full['expected_significance']:.1f}") + ("  mass fixed" if truth["mass_fixed"] else "")
        log(f"{d.name}  mH={truth['mH']:.2f}  mu={truth['mu']:.2f}  bkg_norm={truth['bkg_norm']:.3f}  "
            f"events {full['n_signal_realized'] + full['n_background_realized']}  {tag}")
    (out / "manifest.json").write_text(json.dumps({"task": "hww", "master_seed": seed, "recipe": RECIPE, "pool": str(pool.path),
                                                   "priors": PRIORS, "seeds": truths}, indent=1))
    return truths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="suite directory to create")
    ap.add_argument("--n", type=int, default=10, help="datasets including the public seed_000 (default 10)")
    ap.add_argument("--seed", type=int, default=1, help="master seed; change it for another suite")
    ap.add_argument("--pool", default=str(DEFAULT_POOL), help="the MC pool to draw from (default: the public one)")
    ap.add_argument("--design", help="JSON list of truths to fix, one per sealed seed (unset keys are drawn)")
    a = ap.parse_args()
    truths = build(Path(a.out), a.n, a.seed, Path(a.pool), json.loads(a.design) if a.design else None)
    print(f"wrote {len(truths)} datasets under {a.out}/ (truth.json beside each; seed_000 is at the public truth)")


if __name__ == "__main__":
    main()
