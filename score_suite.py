#!/usr/bin/env python3
"""Run a submission on a suite you made with make_suite.py and score it the way the sealed evaluation does.

    python3 score_suite.py <submission_dir> <suite_dir> [--plot recovery.png] [--plot-main mu.png] [--timeout 600] [--truths DIR]

Runs `run.sh <dataset> <out>` once per seed (no sandbox: the real evaluation has one), reads RESULT.json, and
compares with truth.json: a table of planted vs reported values with pulls, then the aggregates of
EVAL_CONTRACT.md section 3 -- sensitivity (slope of reported vs planted), bias (mean pull), calibration (pull
width), recovery, false-discovery rate and discovery power -- twice: as the leaderboard's main table does, mu
alone on the datasets whose mass stays at 125 GeV (truth.json "mass_fixed", half of a make_suite.py suite),
and as the table folded under it does, mu and the mass on every dataset.  Pulls and slopes use the seeds whose
injected signal has expected local significance >= 5; null seeds count towards the false-discovery rate.  --truths reads
each seed's truth from DIR/<seed>.json instead of <seed>/truth.json (the sealed run keeps them apart).
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import math

import numpy as np

Z_APPLICABLE = 5.0
POIS = ["mu"]                      # required: what truth.json and RESULT.json are compared on
OPTIONAL_POIS = ["mH"]             # scored on the seeds where it is reported (a bonus on the leaderboard)
ALL_POIS = POIS + OPTIONAL_POIS
UNITS = {"mH": " [GeV]", "mu": ""}


def run_seed(sub: Path, ds: Path, timeout: float, as_user: str = None):
    """One run of sub/run.sh on the dataset, with a scrubbed environment (PATH, a scratch HOME, LANG: nothing of
    yours) and, with as_user, as that user through sudo -n (the sealed run: the truths stay unreadable)."""
    out = Path(tempfile.mkdtemp(prefix="score_"))
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(out), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    cmd = ["bash", str(sub / "run.sh"), str(ds.resolve()), str(out)]
    if as_user:
        os.chmod(out, 0o1777)
        cmd = ["sudo", "-n", "-u", as_user, "--", "env", "-i"] + [f"{k}={v}" for k, v in env.items()] + cmd
    t = time.time()
    try:
        p = subprocess.run(cmd, cwd=sub, env=env, capture_output=True, text=True, timeout=timeout)
        status, log = p.returncode, (p.stdout + p.stderr)[-800:]
    except subprocess.TimeoutExpired:
        status, log = "timeout", ""
    res = None
    if status == 0 and (out / "RESULT.json").is_file():
        try:
            res = json.loads((out / "RESULT.json").read_text())
        except (json.JSONDecodeError, PermissionError) as e:
            status, log = "malformed", str(e)
    if as_user:
        subprocess.run(["sudo", "-n", "rm", "-rf", str(out)], capture_output=True)
    shutil.rmtree(out, ignore_errors=True)
    return status, res, time.time() - t, log


def poi(res, name):
    for p in res.get("pois", []):
        if p.get("name") == name:
            return float(p["value"]), float(p["total"])
    raise KeyError(name)



MASS_FIXED_RANGE = (123.0, 127.0)   # make_suite.PRIORS["fixed_mass_range"]: a truth without the mass_fixed flag counts by its mass


def mass_fixed(truth):
    """Whether a dataset is one the main table scores mu on: its mass stays at 125 GeV. make_suite flags it in
    truth.json; a truth without the flag (the public seed_000 as committed) counts when its mass lies in the window."""
    if "mass_fixed" in truth:
        return bool(truth["mass_fixed"])
    return MASS_FIXED_RANGE[0] <= float(truth["mH"]) <= MASS_FIXED_RANGE[1]


WIDTH_OK = (0.5, 1.5)   # honest pipelines sit at 0.6-1.0 on this toy (ORGANISER.md): the window catches padded and overconfident error bars, not that


def load_truth(ds: Path, truths: Path = None):
    f = (truths / f"{ds.name}.json") if truths else (ds / "truth.json")
    return json.loads(f.read_text()) if f.is_file() else None


def score(sub: Path, suite: Path, timeout: float = 600, truths: Path = None, log=print, as_user: str = None, budget: float = None):
    """Run sub/run.sh on every seed of the suite and compare with the truths; prints the per-seed table through
    log and returns {"rows": [...], "agg": aggregate(rows)}. With a budget (seconds for the whole suite), the
    seeds left when it runs out count as not run."""
    sub, suite = Path(sub).resolve(), Path(suite).resolve()
    if not (sub / "run.sh").is_file():
        raise SystemExit(f"no run.sh in {sub}")
    seeds = [d for d in sorted(suite.glob("seed_*")) if load_truth(d, truths)]
    if not seeds:
        raise SystemExit(f"no truths for seed_* under {suite}: make a suite with make_suite.py")
    rows = []
    log(f"{'seed':9s} {'status':8s} {'wall':>6s}  " + "  ".join(f"{p + ' planted':>10s} {'reported':>16s} {'pull':>6s}" for p in ALL_POIS)
        + f"  {'Z_exp':>5s} {'Z_obs':>5s} disc")
    t0 = time.time()
    for ds in seeds:
        truth = load_truth(ds, truths)
        if budget is not None and time.time() - t0 > budget:
            rows.append({"seed": ds.name, "truth": truth, "ok": False, "status": "budget", "wall": 0.0})
            log(f"{ds.name:9s} {'budget':8s}   0.0s  the submission's time for the whole suite ({budget:.0f} s) ran out")
            continue
        status, res, wall, errlog = run_seed(sub, ds, timeout, as_user)
        row = {"seed": ds.name, "truth": truth, "ok": False, "status": str(status), "wall": wall}
        if res is not None:
            try:
                vals = {p: poi(res, p) for p in POIS}
                for p in OPTIONAL_POIS:
                    try:
                        vals[p] = poi(res, p)
                    except (KeyError, TypeError, ValueError):
                        pass
                row.update(ok=True, **{p: v for p, (v, e) in vals.items()}, **{p + "_e": e for p, (v, e) in vals.items()},
                           z=float(res.get("significance_obs", 0.0)), disc=bool(res.get("discovery", False)))
            except (KeyError, TypeError, ValueError) as e:
                status = f"bad:{type(e).__name__}"
                row["status"] = status
        zexp = truth.get("expected_significance", 0.0)
        if row["ok"]:
            for p in ALL_POIS:
                if p in row:
                    row[f"pull_{p}"] = (row[p] - truth[p]) / row[p + "_e"]
            log(f"{ds.name:9s} {'ok':8s} {wall:5.1f}s  " + "  ".join(
                f"{truth[p]:10.2f} {row[p]:8.2f} ± {row[p + '_e']:5.2f} {row['pull_' + p]:6.2f}" if p in row else f"{truth[p]:10.2f} {'not reported':>16s} {'-':>6s}" for p in ALL_POIS)
                + f"  {zexp:5.1f} {row['z']:5.1f} {'yes' if row['disc'] else 'no'}")
        else:
            log(f"{ds.name:9s} {str(status):8s} {wall:5.1f}s  " + "  ".join(f"{truth[p]:10.2f} {'-':>16s} {'-':>6s}" for p in ALL_POIS)
                + f"  {zexp:5.1f}" + (f"\n    {errlog.strip()[-300:]}" if errlog.strip() else ""))
        rows.append(row)
    agg_main, agg = aggregate(rows, mass_fixed, POIS, "the datasets whose mass stays at 125 GeV"), aggregate(rows)
    log("\n== the main table: mu on the datasets whose mass stays at 125 GeV")
    report(agg_main, log)
    log("\n== folded under it, with the mass: every dataset")
    report(agg, log)
    return {"rows": rows, "agg": agg, "agg_main": agg_main}


def aggregate(rows, pick=None, pois=ALL_POIS, selection="every dataset"):
    """The contract's aggregates from the per-seed rows: recovery (over every dataset), and per POI in pois the
    sensitivity slope, bias and calibration width over the strong seeds, the false-discovery rate on null seeds,
    the discovery power on strong, all over the datasets pick (a predicate on the truth) selects: the main table
    is aggregate(rows, mass_fixed, POIS), the table with the mass aggregate(rows)."""
    ok = [r for r in rows if r["ok"]]
    picked = [r for r in ok if pick is None or pick(r["truth"])]
    strong = [r for r in picked if r["truth"].get("expected_significance", 0) >= Z_APPLICABLE]
    null = [r for r in picked if r["truth"]["mu"] == 0]
    agg = {"n": len(rows), "n_ok": len(ok), "recovery": len(ok) / max(len(rows), 1), "selection": selection,
           "n_picked": sum(1 for r in rows if pick is None or pick(r["truth"])),
           "n_strong": len(strong), "n_null": len(null), "pois": {}}
    for name in pois:
        have = [r for r in strong if name in r]       # an optional POI counts on the seeds where it is reported
        if len(have) < 3 or (name in OPTIONAL_POIS and len(have) < len(strong)):
            agg["pois"][name] = None
            continue
        x = np.array([r["truth"][name] for r in have]); y = np.array([r[name] for r in have])
        pulls = np.array([r[f"pull_{name}"] for r in have])
        slope = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else float("nan")
        width = float(np.sqrt(np.mean(pulls ** 2)))
        verdict = ("OK" if WIDTH_OK[0] <= width <= WIDTH_OK[1] else
                   ("UNDER-covering: error bars too small" if width > WIDTH_OK[1] else "OVER-covering: error bars too large"))
        err = float(np.mean([r[f"{name}_e"] for r in have]))              # the mean error bar quoted
        err_cal = err * max(1.0, width) if math.isfinite(width) else float("nan")   # what it would have to be to be honest
        agg["pois"][name] = {"slope": slope, "bias": float(pulls.mean()), "width": width, "verdict": verdict,
                             "err": err, "err_cal": err_cal}
    agg["false_discoveries"] = sum(r["disc"] for r in null)
    agg["fdr"] = agg["false_discoveries"] / len(null) if null else None
    agg["discoveries"] = sum(r["disc"] for r in strong)
    agg["power"] = agg["discoveries"] / len(strong) if strong else None
    return agg


def level_of(agg, poi="mu"):
    """The leaderboard's level for a POI: 0 no measurement (a dataset that did not run, no slope, or slope < 0.8),
    1 follows the truth (slope >= 0.8), 2 and honest error bars (pull width within WIDTH_OK; for mu also no
    discovery claimed on an empty dataset). Level 3 is judged in the room, not here."""
    a = agg["pois"].get(poi)
    if agg["recovery"] < 0.9 or a is None or not math.isfinite(a["slope"]) or a["slope"] < 0.8:
        return 0
    honest = WIDTH_OK[0] <= a["width"] <= WIDTH_OK[1] and (poi != "mu" or not agg["false_discoveries"])
    return 2 if honest else 1


def report(agg, log=print):
    log(f"\nrecovery: {agg['n_ok']}/{agg['n']} seeds completed with a valid RESULT.json")
    log(f"scored for {', '.join(p for p in agg['pois'] if p in POIS)}"
        + (f" (and {', '.join(p for p in agg['pois'] if p in OPTIONAL_POIS)} if reported)" if any(p in OPTIONAL_POIS for p in agg["pois"]) else "")
        + f" on {agg.get('selection', 'every dataset')} ({agg.get('n_picked', agg['n'])} of {agg['n']}): "
        f"{agg['n_strong']} seeds with expected significance >= {Z_APPLICABLE:g}; {agg['n_null']} null seeds")
    for name in agg["pois"]:
        a = agg["pois"][name]
        if a is None:
            log(f"  {name}: " + ("not reported on every scorable seed (optional: no score)" if name in OPTIONAL_POIS else "fewer than 3 scorable seeds, no slope"))
            continue
        log(f"  {name}: sensitivity slope {a['slope']:5.2f} (1 = faithful, 0 = constant) | bias (mean pull) {a['bias']:+5.2f} "
            f"| calibration (pull RMS) {a['width']:4.2f} -> {a['verdict']}")
        log(f"  {name}: mean error bar {a['err']:.3g}, calibrated {a['err_cal']:.3g} (times the pull width above 1: the leaderboard's precision)")
    if agg["fdr"] is not None:
        log(f"  false-discovery rate: {agg['fdr']:.2f} ({agg['false_discoveries']} of {agg['n_null']} null seeds claimed)")
    log(f"  level: {level_of(agg)} on mu" + (f", {level_of(agg, 'mH')} on the mass (optional)" if agg["pois"].get("mH") else
                                            "; the mass not reported (optional)" if "mH" in agg["pois"] else ""))
    if agg["power"] is not None:
        log(f"  discovery power: {agg['power']:.2f} ({agg['discoveries']} of {agg['n_strong']} strong seeds claimed)")


def plot(rows, path, title=None, which="full"):
    """The recovery plot, reported against planted, every scored dataset coloured by how far it sits from the
    truth in units of its own error bar (|pull|: green within one error bar, red at three or more), which is what
    inflates the pull width and bends the slope. which="main": the mu panel alone, on the fixed-mass datasets the
    README's table scores, the moving-mass ones left out; "full": every reported POI on every dataset, filled
    markers the fixed-mass datasets, hollow the moving-mass ones (LEADERBOARD_MASS.md). Small grey: weak or null,
    outside the slope and the pulls. Each panel's title carries the level with the slope and pull width behind it,
    and the mean error bar: the precision of the analysis, the number to push down once the pulls are honest.
    Returns False without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
    except ImportError:
        return False
    main = which == "main"
    ok = [r for r in rows if r["ok"] and (not main or mass_fixed(r["truth"]))]
    names = ["mu"] if main else ([p for p in ALL_POIS if any(p in r for r in ok)] or ["mu"])
    agg = aggregate(rows, mass_fixed, POIS) if main else aggregate(rows)
    where = "the fixed-mass datasets" if main else "every dataset"
    cmap, norm = plt.get_cmap("RdYlGn_r"), Normalize(0.0, 3.0)
    fig, axes = plt.subplots(1, len(names), figsize=(5.8 * len(names), 4.9), squeeze=False, constrained_layout=True)
    for ax, name in zip(axes[0], names):
        unit = UNITS.get(name, "")
        for r in ok:
            if name not in r:
                continue
            strong = r["truth"].get("expected_significance", 0) >= Z_APPLICABLE
            if strong:
                c = cmap(norm(abs(r[f"pull_{name}"])))
                ax.errorbar(r["truth"][name], r[name], yerr=r[f"{name}_e"], fmt="o", ms=5.5, capsize=2, color=c, ecolor=c,
                            mec=c, mfc=c if (main or mass_fixed(r["truth"])) else "none", mew=1.4, alpha=0.95, zorder=3)
            else:
                ax.errorbar(r["truth"][name], r[name], yerr=r[f"{name}_e"], fmt="o", ms=3.5, capsize=1.5, color="0.65", alpha=0.6, zorder=2)
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="reported = planted", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(f"planted {name}{unit}"); ax.set_ylabel(f"reported {name}{unit}")
        ax.legend(loc="upper left", frameon=False)
        p = agg["pois"].get(name)
        sel = [r for r in ok if name in r and r["truth"].get("expected_significance", 0) >= Z_APPLICABLE]
        u = unit.strip(" []")
        ax.set_title((f"level {level_of(agg, name)}: slope {p['slope']:.2f}, pull width {p['width']:.2f}" if p else "no slope: not scored")
                     + f" on {where}\n" + (f"mean error bar {p['err']:.2f}{u}, calibrated {p['err_cal']:.2f}{u}" if p else "no error bar")
                     + f" over {len(sel)} scored datasets", fontsize=9)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=axes[0].tolist(), fraction=0.035, pad=0.02, label="|reported − planted| / your error bar")
    note = ("the datasets the README's table scores: mass at 125 GeV; grey: weak or null, not scored" if main else
            "filled: mass at 125 (the README's table), hollow: mass moving; grey: weak or null, not scored")
    fig.suptitle((title + "\n" if title else "") + note, fontsize=10 if title else 9)
    fig.savefig(path, dpi=130); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission")
    ap.add_argument("suite")
    ap.add_argument("--plot", help="write the recovery plot (reported vs planted, every dataset, mu and the mass) to this PNG")
    ap.add_argument("--plot-main", metavar="PNG", help="the README's view: mu alone on the fixed-mass datasets")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--truths", help="read truths from DIR/<seed>.json instead of <seed>/truth.json")
    a = ap.parse_args()
    r = score(Path(a.submission), Path(a.suite), a.timeout, Path(a.truths).resolve() if a.truths else None)
    for path, which in ((a.plot, "full"), (a.plot_main, "main")):
        if path:
            print(f"plot: {path}" if plot(r["rows"], path, which=which) else "no matplotlib: skipping the plot")


if __name__ == "__main__":
    main()
