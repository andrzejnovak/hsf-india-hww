#!/usr/bin/env python3
"""The figures of GUIDE.md and the slides, from the kit itself:

    python3 make_plots.py [--out figs] [--fast]

  data.png       the public dataset in m_T and m_ll, with the templates at 125 GeV, mu = 1
  morph.png      the signal templates at 100, 125 and 150 GeV, by morphing the simulated sample, on the background
  score.png      the network score for simulated signal, simulated background and the data
  fit.png        the likelihood valleys of the public dataset, in mu and in the mass, for the three example
                 pipelines: the valley whose width is the error bar
  recovery.png   reported vs planted mass and mu for the three pipelines on a suite of 30 (skipped with --fast)
  pulls.png      the pulls of that run
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hww as H  # noqa: E402

PIPELINES = ("template_fit", "tutorial_nn", "nn_fit")


def network(name):
    """An example's fit module and its model, imported from its own folder (each has its own train.features);
    (None, None) for the template fit."""
    if name == "template_fit":
        return None, None
    import importlib
    for mod in ("fit", "train"):
        sys.modules.pop(mod, None)
    d = HERE / "examples" / name
    sys.path.insert(0, str(d))
    try:
        NN = importlib.import_module("fit")
        return NN, NN.load_model(d / "model.json")
    finally:
        sys.path.remove(str(d))

PUBLIC = HERE / "datasets" / "public" / "seed_000"
MT_C = 0.5 * (H.MT_EDGES[1:] + H.MT_EDGES[:-1])
MLL_C = 0.5 * (H.MLL_EDGES[1:] + H.MLL_EDGES[:-1])


def style(plt):
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def pieces():
    data = H.load(PUBLIC / "data" / "events.npz"); pool = H.load(PUBLIC / "mc" / "pool.npz")
    sig, bkg = H.select(pool, pool["label"] == 1), H.select(pool, pool["label"] == 0)
    return data, sig, bkg


def fig_data(plt, out):
    data, sig, bkg = pieces()
    n, s, b = H.template2d(data), H.template2d(sig, sig["weight"]), H.template2d(bkg, bkg["weight"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, axis, c, edges, lab in ((axes[0], 1, MT_C, H.MT_EDGES, "transverse mass m_T [GeV]"), (axes[1], 0, MLL_C, H.MLL_EDGES, "dilepton mass m_ll [GeV]")):
        d, ss, bb = n.sum(axis=axis), s.sum(axis=axis), b.sum(axis=axis)
        ax.errorbar(c, d, yerr=np.sqrt(d), fmt="o", ms=3, color="k", label="the data")
        ax.stairs(bb, edges, color="#2f6db5", label="simulated background (nominal rate)")
        ax.stairs(bb + ss, edges, color="#d98a1e", label="background + signal at μ = 1")
        ax.stairs(ss, edges, color="#b8322f", label="the signal alone")
        ax.set_xlabel(lab); ax.set_ylabel("events / bin"); ax.legend(frameon=False, fontsize=8)
    axes[0].set_title(f"the public dataset: {len(data['lep_pt_0']):,} events, 1000 fb⁻¹ of H → WW → ℓνℓν on the WW continuum", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "data.png"); plt.close(fig)


def fig_morph(plt, out):
    data, sig, bkg = pieces()
    S, B = H.templates({k: np.concatenate([sig[k], bkg[k]]) for k in sig})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, axis, c, edges, lab in ((axes[0], 1, MT_C, H.MT_EDGES, "transverse mass m_T [GeV]"), (axes[1], 0, MLL_C, H.MLL_EDGES, "dilepton mass m_ll [GeV]")):
        for m, col in ((100.0, "#2f6db5"), (125.0, "#b8322f"), (150.0, "#3a9d5d")):
            ax.stairs(S[m].sum(axis=axis), edges, color=col, label=f"signal at m_H = {m:.0f} GeV (μ = 1)")
        ax.stairs(B.sum(axis=axis) / 10, edges, color="0.5", ls="--", label="background / 10")
        ax.set_xlabel(lab); ax.set_ylabel("events / bin"); ax.legend(frameon=False, fontsize=8)
    axes[0].set_title("a Higgs of mass m: the 125 GeV sample with lepton pT and missing ET scaled by m / 125", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "morph.png"); plt.close(fig)


def templates_of(name, data, sig, bkg):
    """(n, S, B) for an example pipeline: the data histogram, the signal templates per mass of the grid, the
    background template, on that pipeline's own axes."""
    NN, model = network(name)
    if NN is None:
        pool = {k: np.concatenate([sig[k], bkg[k]]) for k in sig}
        S, B = H.templates(pool)
        return H.template2d(data), S, B
    S = {float(m): NN.template(model, H.morph(sig, m), sig["weight"]) for m in H.MASS_GRID}
    return NN.template(model, data), S, NN.template(model, bkg, bkg["weight"])


def fig_score(plt, out):
    NN, model = network("nn_fit")
    data, sig, bkg = pieces()
    e = np.linspace(0.0, 1.0, 21); c = 0.5 * (e[1:] + e[:-1])
    ps, pb, pd = NN.score(model, sig), NN.score(model, bkg), NN.score(model, data)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.hist(ps, e, weights=sig["weight"] / sig["weight"].sum(), histtype="step", color="#b8322f", lw=1.5, label="simulated signal")
    ax.hist(pb, e, weights=bkg["weight"] / bkg["weight"].sum(), histtype="step", color="#2f6db5", lw=1.5, label="simulated background")
    ax.set_xlabel("network score: how Higgs-like"); ax.set_ylabel("fraction of events"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("a classifier trained on the simulation (examples/nn_fit/train.py)", fontsize=10)
    ax = axes[1]
    nd = np.histogram(pd, e)[0]; s = np.histogram(ps, e, weights=sig["weight"])[0]; b = np.histogram(pb, e, weights=bkg["weight"])[0]
    ax.errorbar(c, nd, yerr=np.sqrt(nd), fmt="o", ms=3, color="k", label="the data")
    ax.stairs(b, e, color="#2f6db5", label="background (nominal rate)"); ax.stairs(b + s, e, color="#d98a1e", label="background + signal at μ = 1")
    ax.stairs(s, e, color="#b8322f", label="the signal alone")
    ax.set_xlabel("network score"); ax.set_ylabel("events / bin"); ax.set_yscale("log"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("the same score on the public dataset: the fitted observable", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "score.png"); plt.close(fig)


def fig_fit(plt, out):
    from scipy.optimize import minimize
    data, sig, bkg = pieces()
    labels = {"template_fit": "fit of (m_T, m_ll) (examples/template_fit)", "tutorial_nn": "fit of (tutorial score, m_T) (examples/tutorial_nn)",
              "nn_fit": "the same with ten physics features (examples/nn_fit)"}
    mus, ms = np.linspace(0.3, 1.7, 71), np.arange(100.0, 150.01, 1.0)
    curves = {}
    for name in PIPELINES:
        n, S, B = templates_of(name, data, sig, bkg)
        def nll(m, mu, b):
            lam = np.clip(mu * H.interpolate(S, m) + b * B, 1e-9, None); return float(np.sum(lam - n * np.log(lam)))
        ms_fine = np.arange(100.0, 150.01, 0.5)     # the mass profiled on a grid: the interpolated template is piecewise linear in m, a gradient method stalls on its kinks
        prof_mu = np.array([min(minimize(lambda p: nll(m, mu, p[0]), [1.0], method="L-BFGS-B", bounds=[(0.2, 5)]).fun for m in ms_fine) for mu in mus])
        prof_m = np.array([minimize(lambda p: nll(m, p[0], p[1]), [1.0, 1.0], method="L-BFGS-B", bounds=[(-2, 20), (0.2, 5)]).fun for m in ms])
        curves[name] = (prof_mu - prof_mu.min(), prof_m - prof_m.min())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for (name, (pmu, pm)), col in zip(curves.items(), ("#2f6db5", "#d98a1e", "#b8322f")):
        i_mu, i_m = mus[pmu <= 0.5], ms[pm <= 0.5]
        axes[0].plot(mus, pmu, color=col, label=f"{labels[name]}: μ in [{i_mu.min():.2f}, {i_mu.max():.2f}]")
        axes[1].plot(ms, pm, color=col, label=f"m_H in [{i_m.min():.0f}, {i_m.max():.0f}] GeV")
    for ax, x0, xl, lab in ((axes[0], 1.0, "signal strength μ hypothesis", "planted: μ = 1"), (axes[1], 125.0, "m_H hypothesis [GeV]", "planted: 125 GeV")):
        ax.axhline(0.5, color="0.4", ls=":", lw=1); ax.axvline(x0, color="k", ls="--", lw=1, label=lab)
        ax.set_xlabel(xl); ax.set_ylabel("−ln L − min, the other knobs profiled"); ax.set_ylim(0, 8); ax.legend(frameon=False, fontsize=7.5, loc="upper center")
    axes[0].text(0.32, 0.56, "½ unit worse: the 1σ interval", fontsize=8, color="0.3")
    axes[0].set_title("the likelihood valley in μ, public dataset (the mass and the background rate re-fitted at each point)", fontsize=9)
    axes[1].set_title("the valley in the mass (μ and the background rate re-fitted at each point)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fit.png"); plt.close(fig)


def fig_recovery(plt, out):
    import make_suite as M, score_suite as S
    import shutil, tempfile
    suite = Path(tempfile.mkdtemp(prefix="hww_suite_"))
    shutil.rmtree(suite); M.build(suite, 31, 2026, log=lambda s: None)
    res = {name: S.score(HERE / "examples" / name, suite, 600, log=lambda s: None) for name in PIPELINES}
    pois = (("mH", "m_H [GeV]", (98, 152)), ("mu", "μ", (-0.1, 3.1)))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    for row, (poi, lab, lim) in zip(axes, pois):
        for ax, (name, r) in zip(row, res.items()):
            for x in r["rows"]:
                if not x["ok"]:
                    continue
                strong = x["truth"]["expected_significance"] >= S.Z_APPLICABLE
                ax.errorbar(x["truth"][poi], x[poi], yerr=x[poi + "_e"], fmt="o", ms=4, capsize=2, color="C0" if strong else "0.6", alpha=0.9 if strong else 0.6)
            ax.plot(lim, lim, "k--", lw=1, label="reported = planted"); ax.set_xlim(*lim); ax.set_ylim(*lim)
            a = r["agg"]["pois"][poi]
            ax.set_title(f"{name}: slope {a['slope']:.2f}, pull width {a['width']:.2f}, error ≈ {np.mean([x[poi + '_e'] for x in r['rows'] if x['ok'] and x['truth']['expected_significance'] >= S.Z_APPLICABLE]):.2f}", fontsize=9)
            ax.set_xlabel(f"planted {lab}"); ax.legend(frameon=False, fontsize=8, loc="upper left")
        row[0].set_ylabel(f"reported {lab}")
    fig.suptitle("the three example pipelines on 30 planted datasets (blue: Z_exp ≥ 5, grey: weak or empty)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "recovery.png"); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharey=True)
    for row, (poi, lab, _) in zip(axes, pois):
        for ax, (name, r) in zip(row, res.items()):
            ok = [x for x in r["rows"] if x["ok"] and x["truth"]["expected_significance"] >= S.Z_APPLICABLE]
            p = np.array([x[f"pull_{poi}"] for x in ok])
            ax.hist(p, bins=np.linspace(-4, 4, 17), color="#2f6db5", alpha=0.8)
            xs = np.linspace(-4, 4, 200); ax.plot(xs, len(p) * 0.5 * np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="width 1")
            ax.set_xlabel(f"pull of {lab}: (reported − planted) / error"); ax.set_title(f"{name}: width {np.sqrt(np.mean(p**2)):.2f}, mean {p.mean():+.2f}", fontsize=10); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "pulls.png"); plt.close(fig)
    shutil.rmtree(suite, ignore_errors=True)
    for name, r in res.items():
        a = r["agg"]
        print(f"{name} on 30: " + " | ".join(f"{p} slope {a['pois'][p]['slope']:.2f} width {a['pois'][p]['width']:.2f}" for p in ("mH", "mu")) + f" | FDR {a['fdr']} power {a['power']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "figs"))
    ap.add_argument("--fast", action="store_true", help="skip the recovery run (93 fits, 15 min)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in (fig_data, fig_morph, fig_score, fig_fit) + (() if a.fast else (fig_recovery,)):
        f(plt, out); print("wrote", f.__name__[4:])


if __name__ == "__main__":
    main()
