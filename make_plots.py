#!/usr/bin/env python3
"""The figures of the README and the slides, from the kit itself:

    python3 make_plots.py [--out figs] [--fast]

  data.png       the public dataset in m_T and m_ll, with the templates at the public truth
  morph.png      the signal templates at 100, 125 and 150 GeV, by morphing the simulated sample, on the background
  fit.png        the profile likelihood in m_H for the public dataset: the valley whose width is the error bar
  recovery.png   reported vs planted m_H and mu for the reference pipeline on a suite of 30 (skipped with --fast)
  pulls.png      the pulls of that run
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hww as H  # noqa: E402

PUBLIC = HERE / "datasets" / "public" / "seed_000"
MT_C = 0.5 * (H.MT_EDGES[1:] + H.MT_EDGES[:-1])
MLL_C = 0.5 * (H.MLL_EDGES[1:] + H.MLL_EDGES[:-1])


def style(plt):
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def fig_data(plt, out):
    data = H.load(PUBLIC / "data" / "events.npz"); pool = H.load(PUBLIC / "mc" / "pool.npz")
    n = H.template2d(data); S, B = H.templates(pool); s = S[125.0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, axis, c, edges, lab in ((axes[0], 1, MT_C, H.MT_EDGES, "transverse mass m_T [GeV]"), (axes[1], 0, MLL_C, H.MLL_EDGES, "dilepton mass m_ll [GeV]")):
        d, ss, bb = n.sum(axis=axis), s.sum(axis=axis), B.sum(axis=axis)
        ax.errorbar(c, d, yerr=np.sqrt(d), fmt="o", ms=3, color="k", label=f"the data: {int(n.sum())} events")
        ax.stairs(bb, edges, color="#2f6db5", label="background template (nominal)")
        ax.stairs(bb + ss, edges, color="#d98a1e", label="background + signal at m_H = 125, μ = 1")
        ax.stairs(ss, edges, color="#b8322f", label="the signal alone")
        ax.set_xlabel(lab); ax.set_ylabel("events / bin"); ax.legend(frameon=False, fontsize=8)
    axes[0].set_title("the public dataset: 300 fb⁻¹ of H → WW → ℓνℓν on the WW continuum", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "data.png"); plt.close(fig)


def fig_morph(plt, out):
    pool = H.load(PUBLIC / "mc" / "pool.npz"); S, B = H.templates(pool)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, axis, c, edges, lab in ((axes[0], 1, MT_C, H.MT_EDGES, "m_T [GeV]"), (axes[1], 0, MLL_C, H.MLL_EDGES, "m_ll [GeV]")):
        for m, col in ((100.0, "#2f6db5"), (125.0, "#b8322f"), (150.0, "#3a9d5d")):
            ax.stairs(S[m].sum(axis=axis), edges, color=col, label=f"signal at m_H = {m:.0f} GeV (μ = 1)")
        ax.stairs(B.sum(axis=axis) / 10, edges, color="0.5", ls="--", label="background / 10")
        ax.set_xlabel(lab); ax.set_ylabel("events / bin"); ax.legend(frameon=False, fontsize=8)
    axes[0].set_title("a Higgs of mass m: the 125 GeV sample with lepton pT and missing ET scaled by m/125", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "morph.png"); plt.close(fig)


def fig_fit(plt, out):
    from scipy.optimize import minimize
    data = H.load(PUBLIC / "data" / "events.npz"); pool = H.load(PUBLIC / "mc" / "pool.npz")
    n = H.template2d(data); S, B = H.templates(pool)

    def nll(m, mu, b):
        lam = np.clip(mu * H.interpolate(S, m) + b * B, 1e-9, None)
        return float(np.sum(lam - n * np.log(lam)))
    ms = np.arange(100.0, 150.01, 1.0)
    prof = np.array([minimize(lambda p: nll(m, p[0], p[1]), [1.0, 1.0], method="L-BFGS-B", bounds=[(-2, 20), (0.2, 5)]).fun for m in ms])
    prof -= prof.min()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ms, prof, color="#b8322f")
    ax.axhline(0.5, color="0.4", ls=":", lw=1); ax.text(101, 0.55, "½ unit worse: the 1σ interval", fontsize=8, color="0.3")
    inside = ms[prof <= 0.5]
    ax.axvspan(inside.min(), inside.max(), color="#d98a1e", alpha=0.25, label=f"m_H in [{inside.min():.0f}, {inside.max():.0f}] GeV")
    ax.axvline(125, color="k", ls="--", lw=1, label="planted: 125 GeV")
    ax.set_xlabel("m_H hypothesis [GeV]"); ax.set_ylabel("−ln L − min, μ and b profiled"); ax.set_ylim(0, min(prof.max(), 12))
    ax.set_title("a measurement is a fit: the likelihood valley in the mass, public dataset", fontsize=10); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fit.png"); plt.close(fig)


def fig_recovery(plt, out):
    import make_suite as M, score_suite as S
    import shutil, tempfile
    suite = Path(tempfile.mkdtemp(prefix="hww_suite_"))
    shutil.rmtree(suite); M.build(suite, 31, 2026, log=lambda s: None)
    r = S.score(HERE / "examples" / "template_fit", suite, 600, log=lambda s: None)
    S.plot(r["rows"], out / "recovery.png", title="the reference template fit on 30 planted datasets")
    ok = [x for x in r["rows"] if x["ok"] and x["truth"]["expected_significance"] >= S.Z_APPLICABLE]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, name in zip(axes, ("mH", "mu")):
        p = np.array([x[f"pull_{name}"] for x in ok])
        ax.hist(p, bins=np.linspace(-4, 4, 17), color="#2f6db5", alpha=0.8)
        xs = np.linspace(-4, 4, 200); ax.plot(xs, len(p) * 0.5 * np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="width 1")
        ax.set_xlabel(f"pull of {name}: (reported − planted) / error"); ax.set_title(f"width {np.sqrt(np.mean(p**2)):.2f}, mean {p.mean():+.2f}", fontsize=10); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "pulls.png"); plt.close(fig)
    shutil.rmtree(suite, ignore_errors=True)
    a = r["agg"]
    print("reference on 30: mH slope %.2f width %.2f | mu slope %.2f width %.2f | FDR %s" % (a["pois"]["mH"]["slope"], a["pois"]["mH"]["width"], a["pois"]["mu"]["slope"], a["pois"]["mu"]["width"], a["fdr"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "figs"))
    ap.add_argument("--fast", action="store_true", help="skip the recovery run (30 fits)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in (fig_data, fig_morph, fig_fit) + (() if a.fast else (fig_recovery,)):
        f(plt, out); print("wrote", out / (f.__name__[4:] + ".png"))


if __name__ == "__main__":
    main()
