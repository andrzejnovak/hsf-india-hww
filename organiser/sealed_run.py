#!/usr/bin/env python3
"""The sealed run, with the kit alone, anywhere with git, python3, numpy, scipy and torch (CPU): build a sealed suite from a
secret seed, clone every submitted team repository at its submitted commit, run them all, write the leaderboard.

    python3 organiser/sealed_run.py [--seed <secret>] [--n 21] [--out ~/hack-sealed] [--issue 1 | --teams FILE]
                          [--team red --team blue] [--as nobody] [--publish README.md [--plots]]
                          [--no-check] [--timeout 600]

Without --seed the suite is drawn from a seed made up on the spot and never printed: the run is then
self-contained (CI: .github/workflows/sealed_run.yml runs it on every comment on the submissions issue and
commits the table into the README). With --as USER the submissions run as that user through sudo -n, with the
truths unreadable to it (<out> is 711, <out>/truths 700) and a scrubbed environment. --publish writes the
leaderboard table between the `<!-- leaderboard -->` markers of that README; --plots also copies each team's
recovery plot beside it (results/<team>.png) and links it: the plots show the planted values per dataset, so
publish them only once the deadline has passed.

Submissions are the comments on the submissions issue (--issue, default 1 of andrzejnovak/hsf-india-hww,
read through the public GitHub API with no token), one line each: `<team>: <git url> @ <commit>`; the last
comment per team wins. --notify (CI) replies to a comment with a URL in it but no such line, which is otherwise
skipped in silence (a team name with a space, say). --teams FILE reads the same lines from a file instead (a
hand-made list, or the issue copied over). Each repository is cloned into <out>/submissions/<team>/ and checked out at that commit; the
submission is its submission/ directory (or its root when run.sh is there).

The suite goes under --out: seed_000 is the public truth and the rest are drawn from the task's priors. The truths are moved out of the dataset directories into <out>/truths/ before anything
runs, and <out> is made private, so a submission that reads its dataset directory finds no answer key. A second
run with the same --seed and --out reuses the suite (and re-clones each team); without --seed a fresh suite
replaces it.

Per team: the self-check (hygiene, the contract, determinism on the public dataset; <out>/results/<team>/check.txt),
then score_suite.score on every sealed dataset (score.txt, score.json, recovery.png). Submissions run as you,
without a sandbox: run this in a throwaway session or a VM rather than on your own account; the private
harness's `sbh eval` on the clones is the sandboxed alternative (organiser/ORGANISER.md).

Two tables. The main one scores mu alone, on the datasets whose Higgs mass stays at 125 GeV (within 2 GeV;
make_suite flags them mass_fixed: half of the drawn datasets, and seed_000), so that the mass does not pull mu
along. Folded under it (a <details> block in the README), the table with the mass: mu and mH on every dataset,
the mass moving between 100 and 150 GeV. Levels, as the pitch defines them (change `level` below if the rule changes):
  0  self-check failed, fewer than 90 % of the datasets gave a valid RESULT.json, or no slope
  1  a measurement: the slope of reported vs planted mu is at least 0.8
  2  level 1 with honest error bars: a pull width within [0.5, 1.5], and no false discovery
The mass is optional and scored independently in the folded table: a team that reports mH on every scored
dataset gets a mass level by the same rule (1: slope >= 0.8; 2: and width within [0.5, 1.5]); 0 when it is not reported.
  3  is not the machine's to judge (how the pipeline was made); say so when the teams present
Order: the main table by the mu level, then the distance |slope - 1| + |ln width| on mu, smallest first; the
folded table by its mu level, the mass level, then that distance.
Each sbh check run takes two runs of the submission and the scoring --n runs: budget about 2 minutes per team
with a pipeline like examples/template_fit, 15 with examples/tutorial_nn.

The suite: with the SEALED_SEED secret (any long random string) every run builds the same sealed suite, so
runs are comparable and a team whose commit was scored before is read from the cache (results/cache/<team>.json,
aggregates only) instead of re-run; the final run with --plots re-runs everyone. Without the secret the seed is
made up for the run and nothing is cached. The seed and the per-dataset truths never reach the log: the
per-seed tables stay in <out>/results/<team>/score.txt.
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
sys.path.insert(0, str(KIT))
import score_suite as S  # noqa: E402
import make_suite as T  # noqa: E402

PUBLIC = KIT / "datasets" / "public" / "seed_000"
ISSUE_REPO = "andrzejnovak/hsf-india-hww"
LINE = re.compile(r"^\s*`?([A-Za-z0-9_-]+)\s*:\s*(\S+?)\s*@\s*([0-9a-fA-F]{7,40})`?\s*$", re.M)


def secret_seed():
    """The master seed from the SEALED_SEED secret (any string; a long random one), and a cache key derived
    from it. None, None without the secret: a seed made up for this run, and no cache."""
    s = os.environ.pop("SEALED_SEED", None)
    if not s:
        return None, None
    seed = int.from_bytes(hashlib.sha256(("seed:" + s).encode()).digest()[:8], "big") & (2 ** 63 - 1)
    return seed, hashlib.sha256(f"cache:{s}:recipe{T.RECIPE}".encode()).hexdigest()[:12]   # a new recipe: a new suite, no stale results


def cached(cache: Path, team: str, entry, n: int, key: str):
    """The last result of this team's commit on this suite, if the cache has it."""
    if cache is None or key is None:
        return None
    f = cache / f"{team}.json"
    if not f.is_file():
        return None
    try:
        c = json.loads(f.read_text())
    except json.JSONDecodeError:
        return None
    if c.get("commit") == entry["commit"] and c.get("n") == n and c.get("key") == key and "main" in c and "full" in c:
        return c
    return None


def remember(cache: Path, e, n: int, key: str):
    """Cache what the leaderboard needs: aggregates and verdicts, never rows (they carry the planted truths)."""
    if cache is None or key is None:
        return
    cache.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in e.items() if k not in ("rows",)}
    keep.update(n=n, key=key)
    (cache / f"{e['team']}.json").write_text(json.dumps(keep, indent=1, default=str))


def score_team(team, entry, a, out: Path, suite: Path, truths: Path, cache, key):
    """Fetch, self-check and score one team; returns (entry, log lines). Nothing a submission printed and no
    per-dataset truth reaches the returned log: the per-seed table stays in results/<team>/score.txt."""
    lines = []; log = lines.append
    log(f"=== {team}")
    res = out / "results" / team; res.mkdir(parents=True, exist_ok=True)
    submitted = f"{entry['commit'][:7]} ({entry['submitted']})"
    c = None if a.plots else cached(cache, team, entry, a.n, key)
    if c is not None:
        c["rows"] = []
        log(f"cached: this commit was scored on this suite at {c.get('scored_at')}: level {c['level']}")
        return c, lines
    sub = fetch(entry, out / "submissions" / team, log)
    if sub is None:
        (res / "check.txt").write_text("not fetched\n")
        log("level 0 (not fetched)")
        return {"team": team, "check_ok": False, "submitted": submitted, "level": 0, "distance": float("inf"),
                "main": S.aggregate([], S.mass_fixed, S.POIS), "full": S.aggregate([]), "rows": [],
                "url": entry["url"], "commit": entry["commit"]}, lines
    if a.no_check:
        check_ok, note = False, "skipped (--no-check)"
    else:
        check_ok, text = self_check(sub, res / "check.txt", 2 * a.timeout + 60)
        note = "PASS" if check_ok else "FAIL: " + (text.strip().splitlines() or ["(no output)"])[-1][:120]
    log(f"self-check: {note}")
    table = []
    r = S.score(sub, suite, a.timeout, truths, log=table.append, as_user=a.as_user, budget=a.team_budget)
    (res / "score.txt").write_text("\n".join(table) + "\n")
    (res / "score.json").write_text(json.dumps(r, indent=1, default=str))
    log("main table, mu on the datasets whose mass stays at 125 GeV:"); S.report(r["agg_main"], log)
    log("with the mass, every dataset:"); S.report(r["agg"], log)
    if not S.plot(r["rows"], res / "recovery.png", title=f"{team} — sealed run"):
        log("no matplotlib here: no recovery.png (python3 -m pip install matplotlib, or plot score.json elsewhere)")
    e = {"team": team, "check_ok": check_ok, "submitted": submitted, "main": r["agg_main"], "full": r["agg"],
         "level": level(r["agg_main"], check_ok), "distance": distance(r["agg_main"]), "rows": r["rows"],
         "url": entry["url"], "commit": entry["commit"],
         "scored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    remember(cache, e, a.n, key)
    log(f"level {e['level']}  (distance {fmt(e['distance'])})")
    return e, lines


def build_suite(out: Path, seed, n: int, log):
    suite, truths = out / "suite", out / "truths"
    man = truths / "manifest.json"
    if seed is None:
        import secrets
        seed = secrets.randbelow(2 ** 31)
        if suite.exists() or truths.exists():
            shutil.rmtree(suite, ignore_errors=True); shutil.rmtree(truths, ignore_errors=True)
    elif suite.is_dir() and man.is_file():
        have = json.loads(man.read_text())["master_seed"]
        if have != seed:
            raise SystemExit(f"{suite} was built with another seed; pick another --out")
        log(f"suite: {suite} (built earlier with this seed, {len(list(suite.glob('seed_*')))} datasets)")
        return suite, truths
    if suite.exists():
        raise SystemExit(f"{suite} exists without {man}; remove it or pick another --out")
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o711)   # traversable, not listable: a submission running as another user reaches its dataset and nothing else
    log(f"building the sealed suite: {n} datasets, seed {'*' * 6}")
    T.build(suite, n, seed, KIT / "datasets" / "public" / "seed_000" / "mc" / "pool.npz", log=lambda s: None)
    truths.mkdir()
    os.chmod(truths, stat.S_IRWXU)
    for d in sorted(suite.glob("seed_*")):
        (d / "truth.json").rename(truths / f"{d.name}.json")
    (suite / "manifest.json").rename(man)
    left = [p for p in suite.rglob("*") if "truth" in p.name.lower() or p.name == "manifest.json"]
    assert not left, left
    log(f"suite: {suite}; truths in {truths} (private)")
    return suite, truths


def parse_lines(texts):
    """`<team>: <url> @ <commit>` lines, in order; the last line per team wins."""
    teams = {}
    for text, when in texts:
        for m in LINE.finditer(text):
            teams[m.group(1)] = {"team": m.group(1), "url": m.group(2), "commit": m.group(3).lower(), "submitted": when}
    return teams


def teams_from_issue(number: int, notify: bool = False, log=print):
    """The comments of the submissions issue through the GitHub API. GITHUB_TOKEN, when set (the Actions job's
    own token), authenticates the call: anonymous calls share a small rate limit with every other runner. The
    variable is removed from the environment right away, before any submission runs. With notify (CI), a
    comment with a URL in it but no submission line gets a reply saying so (reply_unparsed)."""
    token = os.environ.pop("GITHUB_TOKEN", None)
    url = f"https://api.github.com/repos/{ISSUE_REPO}/issues/{number}/comments?per_page=100"
    comments, page = [], 1
    while True:
        req = urllib.request.Request(url + f"&page={page}", headers=api_headers(token))
        with urllib.request.urlopen(req, timeout=30) as r:
            batch = json.loads(r.read().decode())
        comments += batch
        if len(batch) < 100:
            break
        page += 1
    if notify:
        reply_unparsed(number, comments, token, log)
    return parse_lines([(c["body"], f"{c.get('created_at', '')} by {c.get('user', {}).get('login', '?')}") for c in comments])


def api_headers(token):
    h = {"Accept": "application/vnd.github+json", "User-Agent": "hsf-india-hackathon"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


REPLY_TAG = "<!-- sealed-run:"                                   # on every comment the sealed run posts itself
REPLY_MARK = REPLY_TAG + " no submission line in comment {} -->"
HOW = ("this comment has no submission line in it, so nothing was run for it. The line is "
       "`<team>: <git url> @ <commit>` on a line of its own, the team name letters, digits, `-` and `_` only "
       "(no spaces), e.g. `red-team: https://github.com/alice/hack-red @ 1a2b3c4d5e6f`. Post it as a new "
       "comment; `bash submit.sh <team> <git url>` from the kit folder writes and posts it for you.")


def reply_unparsed(number: int, comments, token, log):
    """A reply under every comment with a URL in it but no `<team>: <url> @ <commit>` line (a team name with a
    space, no commit, ...): once per comment, never to our own replies, and not when its author posted a comment
    that parses later. Without it the comment is skipped in silence and the team waits for a row that never comes."""
    if not token:
        log("no GITHUB_TOKEN: a comment without a submission line gets no reply")
        return
    seen = "\n".join(c.get("body") or "" for c in comments)
    for i, c in enumerate(comments):
        body, who = c.get("body") or "", c.get("user", {}).get("login", "?")
        if REPLY_TAG in body or "://" not in body or LINE.search(body) or REPLY_MARK.format(c["id"]) in seen:
            continue                                   # our own reply (it quotes an example URL), no URL, parses, replied
        if any(LINE.search(d.get("body") or "") and d.get("user", {}).get("login") == who for d in comments[i + 1:]):
            continue
        text = f"@{who} {HOW}\n\n{REPLY_MARK.format(c['id'])}"
        req = urllib.request.Request(f"https://api.github.com/repos/{ISSUE_REPO}/issues/{number}/comments",
                                     data=json.dumps({"body": text}).encode(), method="POST",
                                     headers={**api_headers(token), "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                r.read()
            log(f"replied to the comment of {who} at {c.get('created_at')}: no submission line in it")
        except urllib.error.HTTPError as e:
            log(f"could not reply to the comment of {who} at {c.get('created_at')} (HTTP {e.code}: "
                "the workflow needs `issues: write`)")


def teams_from_file(path: Path):
    return parse_lines([(path.read_text(), f"from {path.name}")])


def fetch(entry, dest: Path, log):
    """Clone the team repository at the submitted commit into dest; returns the submission directory or None."""
    if dest.exists():
        shutil.rmtree(dest)
    try:
        subprocess.run(["git", "clone", "-q", entry["url"], str(dest)], check=True, capture_output=True, text=True, timeout=300)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", entry["commit"]], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        log(f"cannot fetch {entry['url']} @ {entry['commit']}: {(e.stderr or e.stdout).strip()[-200:]}")
        return None
    except subprocess.TimeoutExpired:
        log(f"cloning {entry['url']} timed out")
        return None
    for sub in (dest / "submission", dest):
        if (sub / "run.sh").is_file():
            for q in sub.rglob("*"):
                q.chmod(q.stat().st_mode | 0o044 | (0o011 if q.is_dir() else 0))
            sub.chmod(sub.stat().st_mode | 0o055)
            return sub
    log(f"no submission/run.sh (nor run.sh) in {entry['url']} @ {entry['commit']}")
    return None


def self_check(sub: Path, log_path: Path, timeout: float):
    cmd = [sys.executable, "-m", "sbh", "check", "--task", "hww", "--submission", str(sub),
           "--dataset", str(PUBLIC), "--determinism"]
    try:
        p = subprocess.run(cmd, cwd=KIT, capture_output=True, text=True, timeout=timeout)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "self-check timed out", 1
    log_path.write_text(out)
    return rc == 0, out


def level(agg, check_ok):
    return S.level_of(agg) if check_ok else 0


def mass_level(agg):
    """The optional mass, by the same rule (0 when it is not reported on every scored dataset)."""
    return S.level_of(agg, "mH")


def distance(agg):
    mu = agg["pois"].get("mu")
    if mu is None or not math.isfinite(mu["slope"]):
        return float("inf")
    d = abs(mu["slope"] - 1) + (abs(math.log(mu["width"])) if mu["width"] > 0 else 9.0)
    return d if math.isfinite(d) else float("inf")


def fmt(x, f="{:.2f}"):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f.format(x)


MARK = ("<!-- leaderboard -->", "<!-- /leaderboard -->")


def publish(entries, readme: Path, results: Path, plots: bool, log, budget=None):
    """The table (and, with plots, each team's recovery plot) into the README between the markers."""
    text = readme.read_text()
    if MARK[0] not in text or MARK[1] not in text:
        raise SystemExit(f"{readme} has no {MARK[0]} ... {MARK[1]} markers")
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n, k = max(e["full"]["n"] for e in entries), max(e["main"]["n_picked"] for e in entries)
    body = [f"_Sealed run of {when}: {n} datasets nobody has seen, every submission on the "
            f"[issue](https://github.com/{ISSUE_REPO}/issues/1) at its last comment. The table scores μ on the {k} datasets "
            f"whose Higgs mass stays at 125 GeV; folded under it, μ and the mass on all {n}, the mass moving between 100 and 150 GeV. "
            + ("Recovery plots: reported against planted, per team." if plots else "Recovery plots follow after the deadline.")
            + (f" A submission gets at most {budget / 60:.0f} minutes per run; the datasets left when they run out count as not run." if budget else "") + "_", "",
            table_main(entries), "", "<details>", "<summary><b>With the mass</b>: μ and the mass on every dataset, the mass moving</summary>", "",
            table_full(entries), "", "</details>"]
    if plots:
        pdir = readme.parent / "results"; pdir.mkdir(exist_ok=True)
        for e in entries:
            src = results / e["team"] / "recovery.png"
            if src.is_file():
                shutil.copy(src, pdir / f"{e['team']}.png")
                body += ["", f"**{e['team']}** (level {e['level']})", "", f"![{e['team']}](results/{e['team']}.png)"]
    head, tail = text.split(MARK[0], 1)[0], text.split(MARK[1], 1)[1]
    readme.write_text(head + MARK[0] + "\n" + "\n".join(body) + "\n" + MARK[1] + tail)
    log(f"published into {readme}" + (f" with plots under {readme.parent / 'results'}" if plots else ""))


def md_table(head, rows):
    w = [max(len(h), *(len(r[j]) for r in rows)) if rows else len(h) for j, h in enumerate(head)]
    line = lambda cells: "| " + " | ".join(c.ljust(w[j]) for j, c in enumerate(cells)) + " |"
    return "\n".join([line(head), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows])


def fdr(a):
    return f"{a['false_discoveries']}/{a['n_null']}" if a["n_null"] else "–"


def table_main(entries):
    """The main table: mu on the fixed-mass datasets, by level then distance."""
    entries.sort(key=lambda e: (-e["level"], e["distance"], e["team"]))
    head = ["#", "team", "level", "slope μ", "pull width μ", "false disc.", "self-check", "submitted"]
    rows = []
    for i, e in enumerate(entries, 1):
        a = e["main"]; mu = a["pois"].get("mu")
        rows.append([str(i), e["team"], str(e["level"]), fmt(mu and mu["slope"]), fmt(mu and mu["width"]), fdr(a),
                     "PASS" if e["check_ok"] else "FAIL", e.get("submitted") or "–"])
    return md_table(head, rows)


def table_full(entries):
    """The folded table: mu and the mass on every dataset, by the mu level, the mass level, then distance."""
    order = sorted(entries, key=lambda e: (-level(e["full"], e["check_ok"]), -mass_level(e["full"]), distance(e["full"]), e["team"]))
    head = ["#", "team", "level μ", "level mass", "slope μ", "pull width μ", "slope mass", "pull width mass", "false disc.", "self-check", "submitted"]
    rows = []
    for i, e in enumerate(order, 1):
        a = e["full"]; mu, mh = a["pois"].get("mu"), a["pois"].get("mH")
        rows.append([str(i), e["team"], str(level(a, e["check_ok"])), str(mass_level(a)) if mh else "0",
                     fmt(mu and mu["slope"]), fmt(mu and mu["width"]), fmt(mh and mh["slope"]), fmt(mh and mh["width"]),
                     fdr(a), "PASS" if e["check_ok"] else "FAIL", e.get("submitted") or "–"])
    return md_table(head, rows)


def leaderboard(entries, path_md: Path, path_json: Path, log):
    text = table_main(entries) + "\n\nWith the mass, every dataset:\n\n" + table_full(entries)
    path_md.write_text(f"# Leaderboard\n\n{text}\n\nThe first table scores μ on the datasets whose mass stays at 125 GeV, the second μ and the mass "
                       f"on every dataset. level 0: no measurement (the self-check, a dataset that did not run, or a slope of μ below 0.8); "
                       f"1: follows the truth (slope ≥ 0.8); 2: and honest error bars (pull width in [{S.WIDTH_OK[0]}, {S.WIDTH_OK[1]}], no false discovery). "
                       f"The mass (`mH` in RESULT.json) is optional and scored on its own, by the same rule, when it is reported on every scored dataset. "
                       f"Order: the level, then the distance; with the mass, the μ level, the mass level, then the distance. Level 3 is judged in the room. The full numbers are in results/.\n")
    path_json.write_text(json.dumps([{k: v for k, v in e.items() if k != "rows"} for e in entries], indent=1, default=str))
    log("\n" + text)
    log(f"\nwritten: {path_md}, {path_json}; per team: {path_md.parent}/<team>/recovery.png, score.txt, check.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, help="the secret master seed; default: made up for this run and never printed")
    ap.add_argument("--n", type=int, default=31, help="datasets in the sealed suite, public seed_000 included; half of the drawn ones keep the mass at 125 (default 31)")
    ap.add_argument("--issue", type=int, default=1, help=f"the submissions issue of {ISSUE_REPO} (default 1)")
    ap.add_argument("--teams", help="read the `<team>: <url> @ <commit>` lines from this file instead of the issue")
    ap.add_argument("--out", default="~/hack-sealed", help="private working directory: suite, truths, submissions, results (default ~/hack-sealed)")
    ap.add_argument("--team", action="append", help="only these teams (repeatable)")
    ap.add_argument("--as", dest="as_user", metavar="USER", help="run the submissions as this user through sudo -n (CI: nobody)")
    ap.add_argument("--publish", metavar="README", help="write the leaderboard between the markers of this README")
    ap.add_argument("--plots", action="store_true", help="with --publish: the recovery plots too (after the deadline only)")
    ap.add_argument("--no-check", action="store_true", help="skip the self-check (its verdict then reads FAIL)")
    ap.add_argument("--timeout", type=float, default=600, help="seconds per dataset, the contract's 600")
    ap.add_argument("--team-budget", type=float, help="seconds a submission gets for the whole suite (CI: 2400); the rest counts as not run")
    ap.add_argument("--jobs", type=int, default=1, help="teams scored at the same time (CI: 2)")
    ap.add_argument("--notify", action="store_true", help="reply on the issue to a comment with a URL but no submission line (CI; GITHUB_TOKEN with issues: write)")
    ap.add_argument("--cache", metavar="DIR", help="reuse a team's result when its commit was scored on this suite before (default with --publish: <README dir>/results/cache; needs SEALED_SEED)")
    a = ap.parse_args()
    out = Path(a.out).expanduser().resolve()
    log = print
    teams = teams_from_file(Path(a.teams)) if a.teams else teams_from_issue(a.issue, notify=a.notify, log=log)
    if a.team:
        missing = sorted(set(a.team) - set(teams))
        if missing:
            raise SystemExit(f"no submission line for {', '.join(missing)}")
        teams = {t: e for t, e in teams.items() if t in a.team}
    if not teams:
        raise SystemExit("no submissions: no `<team>: <url> @ <commit>` line " + (f"in {a.teams}" if a.teams else f"on issue {a.issue}"))
    log("submissions: " + ", ".join(f"{t} ({e['url']} @ {e['commit'][:7]})" for t, e in teams.items()))
    seed, key = secret_seed()
    if a.seed is not None:
        seed, key = a.seed, None          # a seed given by hand: no cache, it is not secret enough
    log("suite: " + ("the fixed sealed suite (SEALED_SEED)" if key else "made up for this run" if seed is None else "the seed given by hand"))
    suite, truths = build_suite(out, seed, a.n, log)
    results = out / "results"; results.mkdir(exist_ok=True)
    cache = Path(a.cache).resolve() if a.cache else (Path(a.publish).resolve().parent / "results" / "cache" if a.publish else None)
    if cache is not None and key is None:
        log("no SEALED_SEED: nothing is cached, every team is scored")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futures = {team: pool.submit(score_team, team, entry, a, out, suite, truths, cache, key) for team, entry in teams.items()}
        entries = []
        for team, fut in futures.items():
            e, lines = fut.result()
            log("\n" + "\n".join(lines))
            entries.append(e)
    leaderboard(entries, results / "leaderboard.md", results / "leaderboard.json", log)
    if a.publish:
        publish(entries, Path(a.publish).resolve(), results, a.plots, log, budget=a.team_budget)


if __name__ == "__main__":
    main()
