#!/usr/bin/env python3
"""The figures of GUIDE.md and the slides, from the kit itself:

    python3 make_plots.py [--out figs] [--fast]

  data.png       the public dataset in m_T and m_ll, with the templates at mu = 1
  score.png      the network score for simulated signal, simulated background and the data
  fit.png        the likelihood in mu for the public dataset, fitting m_T and fitting the score: the valley
                 whose width is the error bar, narrower with the classifier
  recovery.png   reported vs planted mu for both pipelines on a suite of 30 (skipped with --fast)
  pulls.png      the pulls of that run
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "examples" / "nn_fit"))
import hww as H  # noqa: E402

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


def fig_score(plt, out):
    import fit as NN
    model = NN.load_model(HERE / "examples" / "nn_fit" / "model.json")
    data, sig, bkg = pieces()
    e = NN.EDGES; c = 0.5 * (e[1:] + e[:-1])
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
    import fit as NN
    model = NN.load_model(HERE / "examples" / "nn_fit" / "model.json")
    data, sig, bkg = pieces()
    curves = {}
    for name, hist in (("fit of m_T (examples/template_fit)", lambda ev, w=None: np.histogram(H.kinematics(ev)["mt"], np.linspace(0, 250, 26), weights=w)[0]),
                       ("fit of the network score (examples/nn_fit)", lambda ev, w=None: NN.histogram(model, ev, w))):
        n, s, b = hist(data), hist(sig, sig["weight"]), hist(bkg, bkg["weight"])
        def nll(mu, bn):
            lam = np.clip(mu * s + bn * b, 1e-9, None); return float(np.sum(lam - n * np.log(lam)))
        mus = np.linspace(0.3, 1.7, 141)
        prof = np.array([minimize(lambda p: nll(m, p[0]), [1.0], method="L-BFGS-B", bounds=[(0.2, 5)]).fun for m in mus])
        curves[name] = (mus, prof - prof.min())
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for (name, (mus, prof)), col in zip(curves.items(), ("#2f6db5", "#b8322f")):
        inside = mus[prof <= 0.5]
        ax.plot(mus, prof, color=col, label=f"{name}: μ in [{inside.min():.2f}, {inside.max():.2f}]")
    ax.axhline(0.5, color="0.4", ls=":", lw=1); ax.text(0.32, 0.56, "½ unit worse: the 1σ interval", fontsize=8, color="0.3")
    ax.axvline(1.0, color="k", ls="--", lw=1, label="planted: μ = 1")
    ax.set_xlabel("signal strength μ hypothesis"); ax.set_ylabel("−ln L − min, background rate profiled"); ax.set_ylim(0, 8)
    ax.set_title("a measurement is a fit: the likelihood valley in μ, public dataset", fontsize=10); ax.legend(frameon=False, fontsize=8, loc="upper center")
    fig.tight_layout(); fig.savefig(out / "fit.png"); plt.close(fig)


def fig_recovery(plt, out):
    import make_suite as M, score_suite as S
    import shutil, tempfile
    suite = Path(tempfile.mkdtemp(prefix="hww_suite_"))
    shutil.rmtree(suite); M.build(suite, 31, 2026, log=lambda s: None)
    res = {name: S.score(HERE / "examples" / name, suite, 600, log=lambda s: None) for name in ("template_fit", "nn_fit")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, (name, r) in zip(axes, res.items()):
        for x in r["rows"]:
            if not x["ok"]:
                continue
            strong = x["truth"]["expected_significance"] >= S.Z_APPLICABLE
            ax.errorbar(x["truth"]["mu"], x["mu"], yerr=x["mu_e"], fmt="o", ms=4, capsize=2, color="C0" if strong else "0.6", alpha=0.9 if strong else 0.6)
        ax.plot([-0.1, 3.1], [-0.1, 3.1], "k--", lw=1, label="reported = planted"); ax.set_xlim(-0.1, 3.1)
        a = r["agg"]["pois"]["mu"]
        ax.set_title(f"{name}: slope {a['slope']:.2f}, pull width {a['width']:.2f}, error ≈ {np.mean([x['mu_e'] for x in r['rows'] if x['ok']]):.2f}", fontsize=10)
        ax.set_xlabel("planted μ"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("reported μ")
    fig.suptitle("the two example pipelines on 30 planted datasets (blue: Z_exp ≥ 5, grey: weak or empty)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "recovery.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (name, r) in zip(axes, res.items()):
        ok = [x for x in r["rows"] if x["ok"] and x["truth"]["expected_significance"] >= S.Z_APPLICABLE]
        p = np.array([x["pull_mu"] for x in ok])
        ax.hist(p, bins=np.linspace(-4, 4, 17), color="#2f6db5", alpha=0.8)
        xs = np.linspace(-4, 4, 200); ax.plot(xs, len(p) * 0.5 * np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="width 1")
        ax.set_xlabel("pull of μ: (reported − planted) / error"); ax.set_title(f"{name}: width {np.sqrt(np.mean(p**2)):.2f}, mean {p.mean():+.2f}", fontsize=10); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "pulls.png"); plt.close(fig)
    shutil.rmtree(suite, ignore_errors=True)
    for name, r in res.items():
        a = r["agg"]
        print(f"{name} on 30: slope {a['pois']['mu']['slope']:.2f} width {a['pois']['mu']['width']:.2f} FDR {a['fdr']} power {a['power']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "figs"))
    ap.add_argument("--fast", action="store_true", help="skip the recovery run (60 fits)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in (fig_data, fig_score, fig_fit) + (() if a.fast else (fig_recovery,)):
        f(plt, out); print("wrote", f.__name__[4:])


if __name__ == "__main__":
    main()
