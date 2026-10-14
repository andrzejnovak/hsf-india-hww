#!/usr/bin/env python3
"""Look at one dataset, and at what a pipeline said about it:

    python3 explore.py datasets/public/seed_000
    python3 explore.py suites/mine/seed_003 --result out/RESULT.json --out seed_003.png

Left and middle: the data in m_T and m_ll against the background template at its nominal normalisation and the
signal template at the planted mu (truth.json beside the dataset) or at the reported one (--result). Right:
the data in the (m_T, m_ll) plane, with the signal template's contours. Prints the counts, the truth if known,
and the pulls if a RESULT.json is given.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hww as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--result", help="a RESULT.json your run.sh wrote for this dataset")
    ap.add_argument("--out", default="explore.png")
    a = ap.parse_args()
    d = Path(a.dataset)
    data = H.load(d / "data" / "events.npz"); pool = H.load(d / "mc" / "pool.npz"); meta = H.read_meta(d)
    truth = json.loads((d / "truth.json").read_text()) if (d / "truth.json").is_file() else None
    res = json.loads(Path(a.result).read_text()) if a.result else None
    pois = {p["name"]: p for p in res.get("pois", [])} if res else {}
    n_ev = len(data["lep_pt_0"])
    print(f"{d}: {n_ev} events; expected at nominal: {meta['signal_expected']:.0f} signal (mu = 1) + {meta['background_expected']:.0f} background, {meta['lumi_fb']:.0f} fb-1")
    if truth:
        print(f"  truth: mu {truth['mu']:.2f}  bkg_norm {truth['bkg_norm']:.3f}  signal events {truth.get('n_signal_realized')}  Z_exp {truth.get('expected_significance')}")
    if res:
        line = "  result:"
        for k in ("mu",):
            if k in pois:
                line += f" {k} {pois[k]['value']:.3f} ± {pois[k]['total']:.3f}"
                if truth:
                    line += f" (pull {(pois[k]['value'] - truth[k]) / pois[k]['total']:+.2f})"
        print(line + f"  Z {res.get('significance_obs', float('nan')):.2f}  discovery {res.get('discovery')}")
    mark = 125.0
    mu = truth["mu"] if truth else (pois.get("mu") or {}).get("value", 1.0)
    S, B = H.templates(pool, masses=[mark])
    s = mu * S[mark]; n = H.template2d(data); kd = H.kinematics(data)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, axis, edges, lab in ((axes[0], 1, H.MT_EDGES, "m_T [GeV]"), (axes[1], 0, H.MLL_EDGES, "m_ll [GeV]")):
        c = 0.5 * (edges[1:] + edges[:-1]); dd = n.sum(axis=axis)
        ax.errorbar(c, dd, yerr=np.sqrt(dd), fmt="o", ms=3, color="k", label=f"data, {n_ev} events")
        ax.stairs(B.sum(axis=axis), edges, color="#2f6db5", label="background, nominal")
        ax.stairs(B.sum(axis=axis) + s.sum(axis=axis), edges, color="#d98a1e", label=f"+ signal at μ = {mu:.2f}" + (" (planted)" if truth else " (reported)" if pois else ""))
        ax.set_xlabel(lab); ax.set_ylabel("events / bin"); ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    ax.hist2d(kd["mt"], kd["mll"], [H.MT_EDGES, H.MLL_EDGES], cmap="Greys")
    X, Y = np.meshgrid(0.5 * (H.MT_EDGES[1:] + H.MT_EDGES[:-1]), 0.5 * (H.MLL_EDGES[1:] + H.MLL_EDGES[:-1]), indexing="ij")
    if s.max() > 0:
        ax.contour(X, Y, s, levels=4, colors="#b8322f", linewidths=1)
    ax.set_xlabel("m_T [GeV]"); ax.set_ylabel("m_ll [GeV]"); ax.set_title("the data (grey) and the signal template (red contours)", fontsize=9)
    fig.suptitle(str(d), fontsize=10); fig.tight_layout(); fig.savefig(a.out, dpi=130)
    print("plot:", a.out)


if __name__ == "__main__":
    main()
